"""
Tests du harnais multi-capacités CQuAE (`evaluation/cquae_multicapacite.py`).

Deux familles :
  1. Séparation gold — `test_aucune_fuite_gold` prouve dynamiquement, agent
     mocké (aucun réseau, aucun Qdrant, aucun LLM), que `executer_cas()` ne
     transmet jamais rien d'autre que `cas.query`.
  2. Scoring — les fonctions `noter_*` sont pures : testées avec des objets
     synthétiques (SimpleNamespace / dataclasses réels du projet remplis à la
     main), jamais un vrai appel agent.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.cquae_multicapacite import (
    CasSmoke,
    DOCUMENT_MANQUANT_CONNU,
    FICHIER_CAS_PAR_DEFAUT,
    FICHIER_GOLD_PAR_DEFAUT,
    _reponse_couvre_gold,
    charger_cas,
    executer_cas,
    noter_anti_hallucination,
    noter_classify,
    noter_extract,
    noter_search,
    noter_summarize,
    verifier_preconditions,
)
from evaluation.common import charger_enregistrements, Enregistrement
from src.rag.generation import ReponseRAG, SourceCitee
from src.tools.base import ResultatOutil, SourceOutil


# ===========================================================================
# Fixtures synthétiques
# ===========================================================================


class _FausseEtape:
    def __init__(self, nom: str, **donnees):
        self.nom = nom
        self.donnees = donnees


class _FauxEtat:
    def __init__(self, trace):
        self.trace = trace
        self.max_tentatives = 6


class _FauxContexte:
    def __init__(self, resultats):
        self.resultats = resultats


class _FausseSession:
    def __init__(self, trace, outils):
        self.etat = _FauxEtat(trace)
        self._outils = outils

    def outils_utilises(self):
        return self._outils


def _brut(session, sortie, duree=1.0, erreur=None):
    return {"session": session, "sortie": sortie, "duree_secondes": duree, "erreur": erreur}


# ===========================================================================
# 1. Séparation gold
# ===========================================================================


def test_aucune_fuite_gold(monkeypatch):
    """
    Espionne le SEUL point d'entrée agent (`construire_session`) : quel que
    soit le cas (SEARCH, EXTRACT avec champs, CLASSIFY, SUMMARIZE), l'unique
    argument reçu doit être `cas.query`, une chaîne — jamais un objet
    contenant gold_qids/champs/source_document.
    """
    appels: list[str] = []

    class FausseSessionEspion:
        def __init__(self, requete):
            appels.append(requete)
            self.etat = _FauxEtat([])
            self.contexte = _FauxContexte([])

        def outils_utilises(self):
            return []

    def fausse_construire_session(requete, **kwargs):
        assert isinstance(requete, str), "construire_session doit recevoir une chaîne, pas un objet gold."
        return FausseSessionEspion(requete)

    class FauxGraphe:
        def invoke(self, etat, config=None):
            return {"session": etat.session, "reponse": SimpleNamespace(reponse="stub", sources=[])}

    monkeypatch.setattr("src.agent.session.construire_session", fausse_construire_session)
    monkeypatch.setattr("src.agent.graph.construire_graphe", lambda: FauxGraphe())

    cas = charger_cas(FICHIER_CAS_PAR_DEFAUT)
    assert len(cas) == 28

    for c in cas:
        appels.clear()
        executer_cas(c)
        assert len(appels) == 1
        assert appels[0] == c.query, f"{c.test_id} : requête transmise différente de cas.query"

        # Garde-fou supplémentaire : aucune valeur gold ne doit apparaître
        # verbatim dans la requête transmise (les gold_qids/labels de champs
        # ne sont jamais des sous-chaînes de query par construction, mais on
        # le revérifie explicitement).
        for qid in c.gold_qids:
            assert qid not in appels[0]
        if c.champs:
            for champ in c.champs:
                if champ.get("gold_qid"):
                    assert champ["gold_qid"] not in appels[0]


def test_executer_cas_source_ne_reference_aucun_champ_gold():
    """
    Garde-fou statique complémentaire à `test_aucune_fuite_gold` (dynamique) :
    le CODE SOURCE de `executer_cas` ne doit contenir aucune référence aux
    champs gold du cas (`gold_qids`, `champs`, `source_document`,
    `evaluation_type`) — seul `cas.query` doit y apparaître. Empêche une
    régression future qui ajouterait une lecture gold dans cette fonction
    sans que `test_aucune_fuite_gold` ne la détecte par hasard.
    """
    import ast
    import inspect

    from evaluation.cquae_multicapacite import executer_cas

    arbre = ast.parse(inspect.getsource(executer_cas))
    fonction = arbre.body[0]
    corps_sans_docstring = fonction.body[1:] if ast.get_docstring(fonction) else fonction.body
    code_executable = "\n".join(ast.unparse(noeud) for noeud in corps_sans_docstring)

    for champ_interdit in ("cas.gold_qids", "cas.champs", "cas.source_document", "cas.evaluation_type"):
        assert champ_interdit not in code_executable, (
            f"executer_cas référence {champ_interdit!r} dans son CODE (hors docstring) — fuite gold potentielle."
        )
    assert "cas.query" in code_executable


# ===========================================================================
# 2. Manifeste des 28 cas
# ===========================================================================


def test_manifeste_28_cas_coherent():
    cas = charger_cas(FICHIER_CAS_PAR_DEFAUT)
    assert len(cas) == 28

    ids = [c.test_id for c in cas]
    assert len(set(ids)) == len(ids), "test_id dupliqué"

    from collections import Counter

    repartition = Counter(c.capability for c in cas)
    assert repartition == {
        "SEARCH": 16,
        "EXTRACT": 5,
        "CLASSIFY": 3,
        "SUMMARIZE": 3,
        "ANTI_HALLUCINATION": 1,
    }

    gold_index = {e.id: e for e in charger_enregistrements(FICHIER_GOLD_PAR_DEFAUT)}
    for c in cas:
        for qid in c.gold_qids:
            assert qid in gold_index, f"{c.test_id} référence un qid gold inexistant : {qid}"
        if c.champs:
            for champ in c.champs:
                if champ.get("gold_qid"):
                    assert champ["gold_qid"] in gold_index


def test_document_manquant_exclu_du_gold():
    tous = charger_enregistrements(FICHIER_GOLD_PAR_DEFAUT)
    excluent = [e for e in tous if e.expected_document == DOCUMENT_MANQUANT_CONNU]
    assert len(excluent) == 1
    assert excluent[0].id == "cquae:test:8298"
    assert len(tous) - len(excluent) == 239


def test_preconditions_detecte_mauvais_profil_domaine(monkeypatch):
    """Reproduit l'état RÉEL actuel du dépôt (french_corpus_demo actif, pas
    histoire-culture-humaines) et vérifie que verifier_preconditions le
    détecte et retourne ok=False, sans avoir touché Qdrant."""
    import evaluation.cquae_multicapacite as module

    faux_settings = SimpleNamespace(
        active_profile="generic",
        active_domain_profile="french_corpus_demo",
        qdrant_path=Path("data/vectordb/qdrant_cquae_eval"),
    )
    faux_technique = SimpleNamespace(qdrant=SimpleNamespace(nom_collection="cquae_eval"))

    monkeypatch.setattr("src.config.get_settings", lambda: faux_settings)
    monkeypatch.setattr("src.config.get_config_technique", lambda: faux_technique)
    monkeypatch.setattr(
        "src.rag.vectorstore.get_client", lambda: SimpleNamespace()
    )
    monkeypatch.setattr(
        "src.rag.vectorstore.info_collection",
        lambda: {"existe": True, "nom": "cquae_eval", "points": 2240, "statut": "green"},
    )

    cas = charger_cas(FICHIER_CAS_PAR_DEFAUT)
    ok, messages = verifier_preconditions(cas, FICHIER_GOLD_PAR_DEFAUT)

    assert ok is False
    assert any("ACTIVE_DOMAIN_PROFILE" in m for m in messages)


def test_preconditions_ok_avec_bons_profils(monkeypatch):
    faux_settings = SimpleNamespace(
        active_profile="generic",
        active_domain_profile="histoire-culture-humaines",
        qdrant_path=Path("data/vectordb/qdrant_cquae_eval"),
    )
    faux_technique = SimpleNamespace(qdrant=SimpleNamespace(nom_collection="cquae_eval"))

    monkeypatch.setattr("src.config.get_settings", lambda: faux_settings)
    monkeypatch.setattr("src.config.get_config_technique", lambda: faux_technique)
    monkeypatch.setattr("src.rag.vectorstore.get_client", lambda: SimpleNamespace())
    monkeypatch.setattr(
        "src.rag.vectorstore.info_collection",
        lambda: {"existe": True, "nom": "cquae_eval", "points": 2240, "statut": "green"},
    )

    cas = charger_cas(FICHIER_CAS_PAR_DEFAUT)
    ok, messages = verifier_preconditions(cas, FICHIER_GOLD_PAR_DEFAUT)
    assert ok is True


# ===========================================================================
# 3. Scoring — SEARCH
# ===========================================================================


def _gold_index_reel():
    return {e.id: e for e in charger_enregistrements(FICHIER_GOLD_PAR_DEFAUT)}


def test_noter_search_pass():
    cas = CasSmoke(
        test_id="SQ-01", capability="SEARCH", query="À quelle date a eu lieu la bataille de Valmy   ?",
        expected_tool="search", native_gold=True, evaluation_type="accuracy",
        source_document="cquae_doc_219.txt", gold_qids=["cquae:test:5528"],
    )
    session = _FausseSession([_FausseEtape("intention", intention="search", desambiguisation_llm=False)], ["search"])
    reponse = ReponseRAG(
        question=cas.query,
        reponse="La bataille de Valmy s'est déroulée le 20 septembre 1792.",
        profil="generic",
        contexte_suffisant=True,
        citations_valides=True,
        citations_reparees=False,
        sources=[
            SourceCitee(
                citation="S1", source="cquae_doc_219.txt", nom_fichier="cquae_doc_219.txt",
                page=None, categorie="autre", score=0.9,
                extrait="La Bataille de Valmy, 20 septembre 1792",
            )
        ],
        recherche=None,
    )
    resultat = noter_search(cas, _brut(session, reponse), _gold_index_reel())
    assert resultat.final_verdict == "PASS"
    assert resultat.routing_status == "OK"


def test_noter_search_routing_failure():
    cas = CasSmoke(
        test_id="SQ-X", capability="SEARCH", query="peu importe",
        expected_tool="search", native_gold=True, evaluation_type="accuracy",
        source_document="cquae_doc_219.txt", gold_qids=["cquae:test:5528"],
    )
    session = _FausseSession([_FausseEtape("intention", intention="classify", desambiguisation_llm=False)], ["classify"])
    reponse = ReponseRAG(
        question=cas.query, reponse="", profil="generic", contexte_suffisant=False,
        citations_valides=True, citations_reparees=False, sources=[], recherche=None,
    )
    resultat = noter_search(cas, _brut(session, reponse), _gold_index_reel())
    assert resultat.routing_status == "ROUTING_FAILURE"
    assert resultat.final_verdict == "WRONG"
    assert resultat.failure_category == "ROUTING_FAILURE"


def test_noter_search_wrong_document_donne_answer_only():
    cas = CasSmoke(
        test_id="SQ-Y", capability="SEARCH", query="peu importe",
        expected_tool="search", native_gold=True, evaluation_type="accuracy",
        source_document="cquae_doc_219.txt", gold_qids=["cquae:test:5528"],
    )
    session = _FausseSession([_FausseEtape("intention", intention="search", desambiguisation_llm=False)], ["search"])
    reponse = ReponseRAG(
        question=cas.query,
        reponse="La bataille de Valmy s'est déroulée le 20 septembre 1792.",
        profil="generic", contexte_suffisant=True, citations_valides=True, citations_reparees=False,
        sources=[
            SourceCitee(
                citation="S1", source="cquae_doc_999.txt", nom_fichier="cquae_doc_999.txt",
                page=None, categorie="autre", score=0.9,
                extrait="La Bataille de Valmy, 20 septembre 1792 hommes",
            )
        ],
        recherche=None,
    )
    resultat = noter_search(cas, _brut(session, reponse), _gold_index_reel())
    assert resultat.retrieval_status == "FAIL"
    assert resultat.final_verdict in {"ANSWER_ONLY", "WRONG"}


# ===========================================================================
# 3bis. Scoring SEARCH — `_reponse_couvre_gold` (tolérance à la paraphrase)
# ===========================================================================


def _g(qid):
    return _gold_index_reel()[qid]


@pytest.mark.parametrize(
    "qid, reponse",
    [
        # SQ-01 : gold = « … s'est déroulée le 20 septembre 1792 », agent = « … a eu lieu le … »
        ("cquae:test:5528", "La bataille de Valmy a eu lieu le 20 septembre 1792, comme l'indique [S1]."),
        # SQ-03 : agent plus précis (jour + mois) que le gold (année seule)
        ("cquae:test:6159", "Pierre et Marie Curie se marient le 26 juillet 1895 à Sceaux [S1]."),
        # SQ-12 : gold verbeux (« soucieux de la splendeur… »), agent = « voûte … fresques … Jules II »
        (
            "cquae:test:10035",
            "Le pape qui a commandé et inauguré la voûte de la Chapelle Sixtine, dont les fresques "
            "sont célèbres, est Jules II selon [S1] ; les travaux durèrent quatre ans.",
        ),
        # SQ-13 : quantité avec séparateur de milliers
        ("cquae:test:2078", "La superficie du parc national de l'Ogooué-Leketi est de 350 000 hectares [S1]."),
    ],
)
def test_reponse_couvre_gold_paraphrase_correcte_acceptee(qid, reponse):
    assert _reponse_couvre_gold(reponse, _g(qid)) is True


@pytest.mark.parametrize(
    "qid, reponse",
    [
        ("cquae:test:5528", "Je ne dispose pas d'information fiable sur la bataille de Valmy."),
        # bon cadre de phrase, millésime faux -> rejeté par la garde numérique
        ("cquae:test:5528", "La bataille de Valmy s'est déroulée le 20 septembre 1789."),
        # mauvaise entité (mauvais pape)
        ("cquae:test:10035", "Le pape Léon X a commandé et inauguré la Chapelle Sixtine."),
        # mauvaise quantité
        ("cquae:test:2078", "La superficie du parc national de l'Ogooué-Leketi est de 120 000 hectares."),
    ],
)
def test_reponse_couvre_gold_reponse_incorrecte_rejetee(qid, reponse):
    assert _reponse_couvre_gold(reponse, _g(qid)) is False


def test_reponse_couvre_gold_court_historique_inchange():
    """Gold court / canonique : `comparer_reponse` suffit, comportement conservé."""
    gold_texte = Enregistrement(
        id="x", question="Quelle est la capitale de la France ?",
        expected_answer="Paris", evidence_text="La capitale de la France est Paris.",
    )
    assert _reponse_couvre_gold("La capitale de la France est Paris.", gold_texte) is True
    assert _reponse_couvre_gold("La capitale de la France est Lyon.", gold_texte) is False

    gold_nombre = Enregistrement(
        id="y", question="Combien d'États comptent les États-Unis ?",
        expected_answer="50", evidence_text="Les États-Unis comptent 50 États.",
    )
    assert _reponse_couvre_gold("Il y en a 50.", gold_nombre) is True
    assert _reponse_couvre_gold("Il y en a 49.", gold_nombre) is False


def test_noter_search_paraphrase_correcte_devient_pass():
    """Bout en bout : une réponse SEARCH correcte mais paraphrasée n'est plus RETRIEVAL_ONLY."""
    cas = CasSmoke(
        test_id="SQ-01", capability="SEARCH", query="À quelle date a eu lieu la bataille de Valmy   ?",
        expected_tool="search", native_gold=True, evaluation_type="accuracy",
        source_document="cquae_doc_219.txt", gold_qids=["cquae:test:5528"],
    )
    session = _FausseSession(
        [_FausseEtape("intention", intention="search", desambiguisation_llm=False)], ["search"]
    )
    reponse = ReponseRAG(
        question=cas.query,
        reponse="La bataille de Valmy a eu lieu le 20 septembre 1792, comme l'indique le document [S1].",
        profil="generic", contexte_suffisant=True, citations_valides=True, citations_reparees=False,
        sources=[
            SourceCitee(
                citation="S1", source="cquae_doc_219.txt", nom_fichier="cquae_doc_219.txt",
                page=None, categorie="autre", score=0.9,
                extrait="Horace Vernet, La Bataille de Valmy, 20 septembre 1792, 1826, huile sur toile.",
            )
        ],
        recherche=None,
    )
    resultat = noter_search(cas, _brut(session, reponse), _gold_index_reel())
    assert resultat.details["exactitude"] is True
    assert resultat.answer_status == "OK"
    assert resultat.final_verdict in {"PASS", "ANSWER_ONLY"}


# ===========================================================================
# 4. Scoring — EXTRACT
# ===========================================================================


def test_noter_extract_champ_trouve_pass():
    cas = CasSmoke(
        test_id="EX-01", capability="EXTRACT", query="peu importe",
        expected_tool="extract", native_gold=False, evaluation_type="accuracy",
        source_document="cquae_doc_219.txt",
        champs=[{"label": "date de la bataille de Valmy", "gold_qid": "cquae:test:5528", "trouve_attendu": True}],
    )
    session = _FausseSession([_FausseEtape("intention", intention="extract", desambiguisation_llm=False)], ["extract"])
    resultat_outil = ResultatOutil(
        outil="extract", succes=True, message="ok",
        donnees={"date bataille valmy": {"trouve": True, "valeur_unique": True, "valeur": "20 septembre 1792", "valeurs": []}},
        sources=[SourceOutil(doc_id="d1", source="s", nom_fichier="cquae_doc_219.txt", extrait="...")],
    )
    resultat = noter_extract(cas, _brut(session, resultat_outil), _gold_index_reel())
    assert resultat.final_verdict == "PASS"


def test_noter_extract_champ_absent_conforme():
    cas = CasSmoke(
        test_id="EX-04", capability="EXTRACT", query="peu importe",
        expected_tool="extract", native_gold=False, evaluation_type="accuracy",
        source_document="cquae_doc_156.txt",
        champs=[{"label": "nom du bateau de Christophe Colomb", "gold_qid": None, "trouve_attendu": False}],
    )
    session = _FausseSession([_FausseEtape("intention", intention="extract", desambiguisation_llm=False)], ["extract"])
    resultat_outil = ResultatOutil(
        outil="extract", succes=True, message="Aucune information trouvée.",
        donnees={"nom du bateau": {"trouve": False, "valeur_unique": False, "valeur": None, "valeurs": []}},
        sources=[],
    )
    resultat = noter_extract(cas, _brut(session, resultat_outil), _gold_index_reel())
    assert resultat.final_verdict == "PASS"


def test_noter_extract_champ_invente_est_wrong():
    cas = CasSmoke(
        test_id="EX-04", capability="EXTRACT", query="peu importe",
        expected_tool="extract", native_gold=False, evaluation_type="accuracy",
        source_document="cquae_doc_156.txt",
        champs=[{"label": "nom du bateau de Christophe Colomb", "gold_qid": None, "trouve_attendu": False}],
    )
    session = _FausseSession([_FausseEtape("intention", intention="extract", desambiguisation_llm=False)], ["extract"])
    resultat_outil = ResultatOutil(
        outil="extract", succes=True, message="ok",
        donnees={"nom du bateau de christophe colomb": {"trouve": True, "valeur_unique": True, "valeur": "Santa Maria", "valeurs": []}},
        sources=[SourceOutil(doc_id="d1", source="s", nom_fichier="cquae_doc_156.txt", extrait="...")],
    )
    resultat = noter_extract(cas, _brut(session, resultat_outil), _gold_index_reel())
    assert resultat.final_verdict == "WRONG"
    assert resultat.failure_category == "EXTRACTION_FAILURE"


# ===========================================================================
# 5. Scoring — CLASSIFY (contract_validation)
# ===========================================================================


def test_noter_classify_categorie_valide_est_pass():
    cas = CasSmoke(
        test_id="CL-01", capability="CLASSIFY", query="Classe le document cquae_doc_219.txt.",
        expected_tool="classify", native_gold=False, evaluation_type="contract_validation",
        source_document="cquae_doc_219.txt",
    )
    session = _FausseSession(
        [
            _FausseEtape("intention", intention="classify", desambiguisation_llm=False),
            _FausseEtape("classify", mode="document_complet", document_demande="cquae_doc_219.txt"),
        ],
        ["classify"],
    )
    resultat_outil = ResultatOutil(
        outil="classify", succes=True, message="ok",
        donnees={"categorie": "autre", "citations": ["S1"]},
        sources=[SourceOutil(doc_id="d1", source="s", nom_fichier="cquae_doc_219.txt", extrait="...")],
    )
    resultat = noter_classify(cas, _brut(session, resultat_outil), ["contrat", "facture", "rapport", "correspondance", "identite", "technique", "autre"])
    assert resultat.final_verdict == "PASS"


def test_noter_classify_categorie_hors_taxonomie_est_wrong():
    cas = CasSmoke(
        test_id="CL-01", capability="CLASSIFY", query="Classe le document cquae_doc_219.txt.",
        expected_tool="classify", native_gold=False, evaluation_type="contract_validation",
        source_document="cquae_doc_219.txt",
    )
    session = _FausseSession(
        [
            _FausseEtape("intention", intention="classify", desambiguisation_llm=False),
            _FausseEtape("classify", mode="document_complet", document_demande="cquae_doc_219.txt"),
        ],
        ["classify"],
    )
    resultat_outil = ResultatOutil(
        outil="classify", succes=True, message="ok",
        donnees={"categorie": "histoire", "citations": ["S1"]},  # catégorie inventée, hors taxonomie
        sources=[SourceOutil(doc_id="d1", source="s", nom_fichier="cquae_doc_219.txt", extrait="...")],
    )
    resultat = noter_classify(cas, _brut(session, resultat_outil), ["contrat", "facture", "rapport", "correspondance", "identite", "technique", "autre"])
    assert resultat.final_verdict == "WRONG"
    assert resultat.failure_category == "CLASSIFICATION_FAILURE"


def test_noter_classify_document_absent_abstain_correct():
    cas = CasSmoke(
        test_id="CL-03", capability="CLASSIFY", query="Classe le document cquae_doc_2262.txt.",
        expected_tool="classify", native_gold=False, evaluation_type="contract_validation",
        source_document="cquae_doc_2262.txt",
    )
    session = _FausseSession(
        [
            _FausseEtape("intention", intention="classify", desambiguisation_llm=False),
            _FausseEtape("classify", mode="document_vise_non_resolu", document_demande=None),
        ],
        [],
    )
    resultat_outil = ResultatOutil(outil="classify", succes=False, message="Document non identifié de façon fiable.")
    resultat = noter_classify(cas, _brut(session, resultat_outil), ["autre"])
    assert resultat.final_verdict == "ABSTAIN_CORRECT"
    assert resultat.failure_category == "MISSING_DOCUMENT_EXCLUDED"


def test_noter_classify_document_absent_mais_classification_produite_est_wrong():
    """Repli sur le mode contextuel malgré un document demandé absent : doit être signalé, pas absous."""
    cas = CasSmoke(
        test_id="CL-03", capability="CLASSIFY", query="Classe le document cquae_doc_2262.txt.",
        expected_tool="classify", native_gold=False, evaluation_type="contract_validation",
        source_document="cquae_doc_2262.txt",
    )
    session = _FausseSession(
        [
            _FausseEtape("intention", intention="classify", desambiguisation_llm=False),
            _FausseEtape("classify", mode="contexte_existant", document_demande=None),
        ],
        ["search", "classify"],
    )
    resultat_outil = ResultatOutil(
        outil="classify", succes=True, message="ok", donnees={"categorie": "autre", "citations": ["S1"]},
        sources=[SourceOutil(doc_id="d1", source="s", nom_fichier="cquae_doc_47.txt", extrait="...")],
    )
    resultat = noter_classify(cas, _brut(session, resultat_outil), ["autre"])
    assert resultat.final_verdict == "WRONG"
    assert "alerte" in resultat.details


# ===========================================================================
# 6. Scoring — SUMMARIZE
# ===========================================================================


def test_noter_summarize_structurel_pass_sans_contradiction():
    cas = CasSmoke(
        test_id="SU-01", capability="SUMMARIZE", query="Résume le document cquae_doc_2920.txt.",
        expected_tool="summarize", native_gold=False, evaluation_type="structural_and_faithfulness",
        source_document="cquae_doc_2920.txt", gold_qids=["cquae:test:10915"],
    )
    session = _FausseSession(
        [
            _FausseEtape("intention", intention="summarize", desambiguisation_llm=False),
            _FausseEtape("summarize", documents_demandes=["cquae_doc_2920.txt"], succes=True),
        ],
        ["summarize"],
    )
    resultat_outil = ResultatOutil(
        outil="summarize", succes=True, message="ok",
        donnees={"resume": "Le téléfilm La Controverse de Valladolid (1992) est adapté du roman de Jean-Claude Carrière."},
        sources=[SourceOutil(doc_id="d1", source="s", nom_fichier="cquae_doc_2920.txt", extrait="...")],
    )
    resultat = noter_summarize(cas, _brut(session, resultat_outil), _gold_index_reel())
    assert resultat.final_verdict == "PASS"
    assert resultat.details["structural_ok"] is True


def test_noter_summarize_routing_failure_search_au_lieu_de_summarize():
    """
    Cas SU-02 exact : si le routage échoue vers 'search' au lieu de
    'summarize', `brut['sortie']` est un ReponseRAG (pas un ResultatOutil) —
    le scoring ne doit pas planter dessus (régression réelle trouvée et
    corrigée pendant l'implémentation : voir `_resultat_routing_failure`).
    """
    cas = CasSmoke(
        test_id="SU-02", capability="SUMMARIZE",
        query="Donne-moi les points essentiels du document cquae_doc_219.txt.",
        expected_tool="summarize", native_gold=False, evaluation_type="structural_and_faithfulness",
        source_document="cquae_doc_219.txt", gold_qids=["cquae:test:5528"],
    )
    session = _FausseSession([_FausseEtape("intention", intention="search", desambiguisation_llm=False)], ["search"])
    reponse_recue_a_la_place = ReponseRAG(
        question=cas.query, reponse="réponse rédigée au lieu d'un résumé structuré",
        profil="generic", contexte_suffisant=True, citations_valides=True, citations_reparees=False,
        sources=[], recherche=None,
    )
    resultat = noter_summarize(cas, _brut(session, reponse_recue_a_la_place), _gold_index_reel())
    assert resultat.routing_status == "ROUTING_FAILURE"
    assert resultat.final_verdict == "WRONG"
    assert resultat.failure_category == "ROUTING_FAILURE"
    assert resultat.detected_tool == "search"


def test_noter_summarize_document_errone_signale():
    cas = CasSmoke(
        test_id="SU-01", capability="SUMMARIZE", query="Résume le document cquae_doc_2920.txt.",
        expected_tool="summarize", native_gold=False, evaluation_type="structural_and_faithfulness",
        source_document="cquae_doc_2920.txt", gold_qids=["cquae:test:10915"],
    )
    session = _FausseSession(
        [
            _FausseEtape("intention", intention="summarize", desambiguisation_llm=False),
            _FausseEtape("summarize", documents_demandes=["cquae_doc_9999.txt"], succes=True),
        ],
        ["summarize"],
    )
    resultat_outil = ResultatOutil(
        outil="summarize", succes=True, message="ok",
        donnees={"resume": "résumé quelconque"},
        sources=[SourceOutil(doc_id="d1", source="s", nom_fichier="cquae_doc_9999.txt", extrait="...")],
    )
    resultat = noter_summarize(cas, _brut(session, resultat_outil), _gold_index_reel())
    assert resultat.final_verdict == "WRONG"
    assert resultat.details["structural_ok"] is False


# ===========================================================================
# 7. Scoring — Anti-hallucination (AH-01)
# ===========================================================================


def test_noter_anti_hallucination_refus_est_abstain_correct():
    cas = CasSmoke(
        test_id="AH-01", capability="ANTI_HALLUCINATION",
        query="Quel est le score final de la finale de la Coupe du Monde de football 2022 ?",
        expected_tool="search", native_gold=False, evaluation_type="grounding", source_document=None,
    )
    session = _FausseSession([_FausseEtape("intention", intention="search", desambiguisation_llm=False)], ["search"])
    reponse = ReponseRAG(
        question=cas.query, reponse="Je ne dispose pas de preuve documentaire pour répondre.",
        profil="generic", contexte_suffisant=False, citations_valides=True, citations_reparees=False,
        sources=[], recherche=None,
    )
    resultat = noter_anti_hallucination(cas, _brut(session, reponse))
    assert resultat.final_verdict == "ABSTAIN_CORRECT"


def test_noter_anti_hallucination_reponse_sans_preuve_est_unsupported():
    cas = CasSmoke(
        test_id="AH-01", capability="ANTI_HALLUCINATION",
        query="Quel est le score final de la finale de la Coupe du Monde de football 2022 ?",
        expected_tool="search", native_gold=False, evaluation_type="grounding", source_document=None,
    )
    session = _FausseSession([_FausseEtape("intention", intention="search", desambiguisation_llm=False)], ["search"])
    reponse = ReponseRAG(
        question=cas.query, reponse="L'Argentine a battu la France 4-2 aux tirs au but.",
        profil="generic", contexte_suffisant=True, citations_valides=True, citations_reparees=False,
        sources=[
            SourceCitee(
                citation="S1", source="cquae_doc_1.txt", nom_fichier="cquae_doc_1.txt",
                page=None, categorie="autre", score=0.2, extrait="passage sans rapport",
            )
        ],
        recherche=None,
    )
    resultat = noter_anti_hallucination(cas, _brut(session, reponse))
    assert resultat.final_verdict == "UNSUPPORTED"
    assert resultat.final_verdict != "PASS"
