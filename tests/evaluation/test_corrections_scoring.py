"""
Tests ciblés des trois corrections de scoring du harnais (aucun moteur touché) :

  Défaut 1 — `evaluate_end_to_end.calculer_groundedness` reconstruit le
             contexte depuis le TEXTE COMPLET des passages cités (via
             `reponse.recherche.passages`), avec repli sur `source.extrait`.

  Défaut 2 — `cquae_multicapacite._cle_document_resolu` traduit le doc_id
             (UUID) enregistré dans la trace en nom de fichier avant
             comparaison au gold, avec repli sûr.

  Défaut 3 — `cquae_multicapacite.noter_extract` lit `donnees["extractions"]`
             (et non la racine) ; `_associer_champ` apparie exact / inclusion
             / Jaccard>=0.34 ; `_valeur_champ_extrait` gère les valeurs
             multiples.

Tous synthétiques : aucun appel agent, LLM, Qdrant ou réseau.
"""

from __future__ import annotations

from types import SimpleNamespace

from evaluation.common import Enregistrement
from evaluation.cquae_multicapacite import (
    CasSmoke,
    _associer_champ,
    _cle_document_resolu,
    _valeur_champ_extrait,
    noter_classify,
    noter_extract,
    noter_summarize,
)
from evaluation.common import cle_document
from evaluation.evaluate_end_to_end import calculer_groundedness
from src.rag.generation import ReponseRAG, SourceCitee
from src.tools.base import ResultatOutil, SourceOutil


# ===========================================================================
# Fakes minimaux (mêmes contrats que tests/evaluation/test_cquae_multicapacite.py)
# ===========================================================================


class _FausseEtape:
    def __init__(self, nom: str, **donnees):
        self.nom = nom
        self.donnees = donnees


class _FauxEtat:
    def __init__(self, trace):
        self.trace = trace
        self.max_tentatives = 6


class _FausseSession:
    def __init__(self, trace, outils):
        self.etat = _FauxEtat(trace)
        self._outils = outils

    def outils_utilises(self):
        return self._outils


def _brut(session, sortie):
    return {"session": session, "sortie": sortie, "duree_secondes": 1.0, "erreur": None}


def _reponse_rag(reponse: str, sources, recherche):
    return ReponseRAG(
        question="q",
        reponse=reponse,
        profil="generic",
        contexte_suffisant=True,
        citations_valides=True,
        citations_reparees=False,
        sources=list(sources),
        recherche=recherche,
    )


def _source_citee(citation: str, extrait: str) -> SourceCitee:
    return SourceCitee(
        citation=citation,
        source="d.txt",
        nom_fichier="d.txt",
        page=None,
        categorie="",
        score=0.9,
        extrait=extrait,
    )


# ===========================================================================
# Défaut 1 — groundedness sur le TEXTE COMPLET des passages récupérés
# ===========================================================================


_PHRASE = "La bataille de Valmy eut lieu en 1792."

# Un passage réel dépasse largement 320 caractères : l'information utile
# (« 20 septembre 1792 ») est ici placée APRÈS le 320e caractère pour
# reproduire exactement le défaut de l'audit — l'extrait tronqué
# (`extrait[:317] + "..."`) ne la contient pas, le passage complet si.
_BOURRAGE = "Cette huile sur toile de grand format est conservee au musee. " * 7
_PASSAGE_LONG = (
    _BOURRAGE
    + "La bataille de Valmy, premiere victoire revolutionnaire decisive, "
    "eut lieu le 20 septembre 1792 face aux troupes prussiennes."
)
assert len(_BOURRAGE) > 320  # garantit que l'info utile est hors de l'extrait tronqué


def _passage(citation: str, texte: str) -> SimpleNamespace:
    return SimpleNamespace(citation=citation, texte=texte)


def test_groundedness_reponse_reellement_ancree_est_acceptee():
    """Scénario 1 (Tâche A) : réponse réellement soutenue par les passages
    récupérés -> score élevé, au-dessus du seuil de décision (0.5)."""
    reponse = _reponse_rag(
        "La bataille de Valmy eut lieu le 20 septembre 1792.",
        sources=[_source_citee("S1", extrait="…tronqué…")],
        recherche=SimpleNamespace(passages=[_passage("S1", _PASSAGE_LONG)]),
    )
    assert calculer_groundedness(reponse) == 1.0


def test_groundedness_reponse_non_supportee_est_rejetee():
    """Scénario 2 (Tâche A) : réponse dont les faits n'apparaissent dans aucun
    passage récupéré -> score effondré, très en dessous du seuil. La
    correction ne rend pas la métrique permissive."""
    reponse = _reponse_rag(
        "Le traite fut signe a Vienne par le chancelier Metternich en 1815.",
        sources=[_source_citee("S1", extrait="…tronqué…")],
        recherche=SimpleNamespace(passages=[_passage("S1", _PASSAGE_LONG)]),
    )
    score = calculer_groundedness(reponse)
    assert score < 0.2


def test_groundedness_info_au_dela_du_320e_caractere_nest_plus_un_faux_negatif():
    """Scénario 3 (Tâche A) : l'information correcte est dans le passage cité
    mais APRÈS le 320e caractère. Mesurée sur l'extrait tronqué elle donnait
    un faux PROVENANCE_FAILURE ; mesurée sur le passage complet elle est
    correctement reconnue comme ancrée."""
    reponse_verbeuse = (
        "D'apres le document, la bataille de Valmy eut lieu le 20 septembre "
        "1792 et constitua la premiere victoire revolutionnaire decisive face "
        "aux troupes prussiennes."
    )
    extrait_tronque = _PASSAGE_LONG[:317] + "..."  # reproduit src/rag/generation.py

    grounded_tronque = calculer_groundedness(
        _reponse_rag(
            reponse_verbeuse,
            sources=[_source_citee("S1", extrait=extrait_tronque)],
            recherche=None,  # force la mesure sur l'extrait (repli historique)
        )
    )
    grounded_complet = calculer_groundedness(
        _reponse_rag(
            reponse_verbeuse,
            sources=[_source_citee("S1", extrait=extrait_tronque)],
            recherche=SimpleNamespace(passages=[_passage("S1", _PASSAGE_LONG)]),
        )
    )

    assert grounded_tronque < 0.5  # le faux négatif d'origine
    assert grounded_complet >= 0.5  # corrigé
    assert grounded_complet > grounded_tronque


def test_groundedness_utilise_tous_les_passages_recuperes_pas_seulement_les_cites():
    """Un fait exact repris d'un passage récupéré mais non cité explicitement
    reste ancré : le dénominateur est l'ensemble des passages récupérés
    (définition standard de la fidélité RAG), pas le seul extrait affiché."""
    reponse = _reponse_rag(
        "Napoleon fut sacre empereur en 1804 a l'age de trente-cinq ans.",
        sources=[_source_citee("S1", extrait="…")],
        recherche=SimpleNamespace(
            passages=[
                _passage("S1", "Napoleon Bonaparte est ne en 1769 en Corse."),
                _passage("S2", "Il fut sacre empereur des Francais en 1804."),
                _passage("S3", "Trente-cinq annees separent ces deux dates."),
            ]
        ),
    )
    assert calculer_groundedness(reponse) == 1.0


def test_groundedness_repli_sur_extrait_quand_recherche_absente():
    """`reponse.recherche is None` : comportement historique conservé
    (mesure sur `source.extrait`), sans exception."""
    reponse = _reponse_rag(
        _PHRASE,
        sources=[_source_citee("S1", extrait=_PHRASE)],
        recherche=None,
    )
    assert calculer_groundedness(reponse) == 1.0


def test_groundedness_repli_extrait_pauvre_reste_bas():
    """Le repli mesure bien sur l'extrait fourni : un extrait pauvre donne un
    score bas, il n'y a pas de contournement caché."""
    reponse = _reponse_rag(
        _PHRASE,
        sources=[_source_citee("S1", extrait="1792")],
        recherche=None,
    )
    score = calculer_groundedness(reponse)
    assert 0.0 < score < 0.5


def test_groundedness_repli_sur_extrait_si_rapport_sans_passage():
    """Rapport de recherche présent mais sans aucun passage : repli sur les
    extraits des sources, pas une division par zéro ni une exception."""
    reponse = _reponse_rag(
        _PHRASE,
        sources=[_source_citee("S1", extrait=_PHRASE)],
        recherche=SimpleNamespace(passages=[]),
    )
    assert calculer_groundedness(reponse) == 1.0


# ===========================================================================
# Défaut 2 — résolution UUID -> nom_fichier
# ===========================================================================


def _catalogue_factice(mapping: dict[str, str]):
    def par_identifiant(identifiant):
        nom = mapping.get(identifiant)
        return SimpleNamespace(nom_fichier=nom) if nom else None

    return SimpleNamespace(par_identifiant=par_identifiant)


def test_cle_document_resolu_uuid_vers_nom_fichier(monkeypatch):
    monkeypatch.setattr(
        "src.rag.retrieval.catalogue",
        lambda: _catalogue_factice({"uuid-de-la-version-219": "cquae_doc_219.txt"}),
    )
    assert _cle_document_resolu("uuid-de-la-version-219") == cle_document(
        "cquae_doc_219.txt"
    )


def test_cle_document_resolu_uuid_inconnu_repli_sur_identifiant(monkeypatch):
    monkeypatch.setattr("src.rag.retrieval.catalogue", lambda: _catalogue_factice({}))
    ident = "uuid-jamais-vu"
    assert _cle_document_resolu(ident) == cle_document(ident)


def test_cle_document_resolu_catalogue_indisponible_repli(monkeypatch):
    def boom():
        raise RuntimeError("Qdrant indisponible pendant le scoring")

    monkeypatch.setattr("src.rag.retrieval.catalogue", boom)
    assert _cle_document_resolu("cquae_doc_219.txt") == cle_document("cquae_doc_219.txt")


def test_cle_document_resolu_vide():
    assert _cle_document_resolu("") == ""
    assert _cle_document_resolu(None) == ""


def test_noter_classify_ne_souffre_plus_du_bug_uuid_vs_filename(monkeypatch):
    """CL-01/CL-02 : la trace porte l'UUID, le gold un nom de fichier ; après
    résolution, `document_demande_ok` est vrai et le verdict n'est plus
    WRONG/DOCUMENT_RETRIEVAL_FAILURE."""
    monkeypatch.setattr(
        "src.rag.retrieval.catalogue",
        lambda: _catalogue_factice({"UUID-219": "cquae_doc_219.txt"}),
    )
    cas = CasSmoke(
        test_id="CL-01",
        capability="CLASSIFY",
        query="Classe le document cquae_doc_219.txt.",
        expected_tool="classify",
        native_gold=False,
        evaluation_type="contract_validation",
        source_document="cquae_doc_219.txt",
    )
    session = _FausseSession(
        [
            _FausseEtape("intention", intention="classify", desambiguisation_llm=False),
            _FausseEtape("classify", mode="document_complet", document_demande="UUID-219"),
        ],
        ["classify"],
    )
    resultat_outil = ResultatOutil(
        outil="classify",
        succes=True,
        message="ok",
        donnees={"categorie": "autre", "citations": ["S1"]},
        sources=[
            SourceOutil(doc_id="UUID-219", source="s", nom_fichier="cquae_doc_219.txt", extrait="...")
        ],
    )
    r = noter_classify(
        cas,
        _brut(session, resultat_outil),
        ["contrat", "facture", "rapport", "correspondance", "identite", "technique", "autre"],
    )
    assert r.details["document_demande_ok"] is True
    assert r.final_verdict != "WRONG"
    assert r.failure_category != "DOCUMENT_RETRIEVAL_FAILURE"


def test_noter_summarize_ne_souffre_plus_du_bug_uuid_vs_filename(monkeypatch):
    monkeypatch.setattr(
        "src.rag.retrieval.catalogue",
        lambda: _catalogue_factice({"UUID-2920": "cquae_doc_2920.txt"}),
    )
    cas = CasSmoke(
        test_id="SU-01",
        capability="SUMMARIZE",
        query="Résume le document cquae_doc_2920.txt.",
        expected_tool="summarize",
        native_gold=False,
        evaluation_type="structural_and_faithfulness",
        source_document="cquae_doc_2920.txt",
        gold_qids=[],
    )
    session = _FausseSession(
        [
            _FausseEtape("intention", intention="summarize", desambiguisation_llm=False),
            _FausseEtape("summarize", documents_demandes=["UUID-2920"], succes=True),
        ],
        ["summarize"],
    )
    resultat_outil = ResultatOutil(
        outil="summarize",
        succes=True,
        message="ok",
        donnees={"resume": "Un résumé cohérent et sourcé [S1]."},
        sources=[
            SourceOutil(doc_id="UUID-2920", source="s", nom_fichier="cquae_doc_2920.txt", extrait="...")
        ],
    )
    r = noter_summarize(cas, _brut(session, resultat_outil), {})
    assert r.details["structural_ok"] is True
    assert r.final_verdict == "PASS"


# ===========================================================================
# Défaut 3 — EXTRACT : niveau d'imbrication, appariement, valeurs multiples
# ===========================================================================


def _session_extract():
    return _FausseSession(
        [_FausseEtape("intention", intention="extract", desambiguisation_llm=False)],
        ["extract"],
    )


def _cas_extract(champs, source_document="d.txt", test_id="EX-XX"):
    return CasSmoke(
        test_id=test_id,
        capability="EXTRACT",
        query="peu importe",
        expected_tool="extract",
        native_gold=False,
        evaluation_type="accuracy",
        source_document=source_document,
        champs=champs,
    )


def _resultat_extract(extractions, sources=None, succes=True):
    return ResultatOutil(
        outil="extract",
        succes=succes,
        message="ok",
        donnees={"document": "d.txt", "extractions": extractions, "champs_demandes": list(extractions)},
        sources=sources or [],
    )


def test_noter_extract_lit_donnees_extractions():
    cas = _cas_extract([{"label": "montant total", "gold_qid": None, "trouve_attendu": True}])
    res = _resultat_extract(
        {"montant total": {"trouve": True, "valeur": "42", "valeurs": [{"valeur": "42", "citations": ["S1"]}]}},
        sources=[SourceOutil(doc_id="d", source="s", nom_fichier="d.txt", extrait="...")],
    )
    r = noter_extract(cas, _brut(_session_extract(), res), {})
    detail = r.details["champs"]["montant total"]
    assert detail["champ_retourne"] == "montant total"
    assert detail["trouve"] is True
    assert r.final_verdict == "PASS"


def test_noter_extract_ignore_les_champs_a_la_racine_ancien_bug():
    """Champs placés à la racine de `donnees` (structure supposée par l'ancien
    scoring) : ne sont plus lus par erreur."""
    cas = _cas_extract([{"label": "montant total", "gold_qid": None, "trouve_attendu": True}])
    res = ResultatOutil(
        outil="extract",
        succes=True,
        message="ok",
        donnees={"montant total": {"trouve": True, "valeur": "42", "valeurs": []}},  # racine, pas "extractions"
        sources=[SourceOutil(doc_id="d", source="s", nom_fichier="d.txt", extrait="...")],
    )
    r = noter_extract(cas, _brut(_session_extract(), res), {})
    detail = r.details["champs"]["montant total"]
    assert detail["champ_retourne"] is None
    assert detail["trouve"] is False
    assert r.final_verdict == "WRONG"


def test_associer_champ_egalite_exacte_normalisee():
    assert _associer_champ("date du contrat", {"Date Du Contrat": {"x": 1}}) == "Date Du Contrat"
    assert _associer_champ("effectif de la Grande Armée", {"effectif de la grande armee": {"x": 1}}) == (
        "effectif de la grande armee"
    )


def test_associer_champ_inclusion():
    assert _associer_champ("montant total hors taxes", {"montant": {"x": 1}}) == "montant"
    assert _associer_champ(
        "date de signature", {"date de signature du contrat de bail": {"x": 1}}
    ) == "date de signature du contrat de bail"


def test_associer_champ_jaccard_au_dessus_du_seuil():
    # ni exact ni inclusion, mais 3 jetons communs sur 5 -> Jaccard 0.6 >= 0.34
    assert _associer_champ("effectif de la Grande Armée", {"effectif grande armee": {"x": 1}}) == (
        "effectif grande armee"
    )


def test_associer_champ_rejet_sous_le_seuil():
    # 2 jetons communs ("de", "du") sur 8 -> Jaccard 0.25 < 0.34, aucune inclusion
    assert _associer_champ("prix de vente du tableau", {"date de creation du corps": {"x": 1}}) is None
    # aucun jeton commun
    assert _associer_champ("nom du commissaire aux comptes", {"chiffre affaires consolide": {"x": 1}}) is None


def test_noter_extract_champ_absent_evalue_correctement():
    cas = _cas_extract([{"label": "prix de vente", "gold_qid": None, "trouve_attendu": False}])
    res = _resultat_extract({"prix de vente": {"trouve": False, "valeur": None, "valeurs": []}})
    r = noter_extract(cas, _brut(_session_extract(), res), {})
    detail = r.details["champs"]["prix de vente"]
    assert detail["champ_retourne"] == "prix de vente"
    assert detail["trouve"] is False
    assert r.final_verdict == "PASS"


def test_noter_extract_champ_present_alors_qu_attendu_absent_est_wrong():
    cas = _cas_extract([{"label": "prix de vente", "gold_qid": None, "trouve_attendu": False}])
    res = _resultat_extract(
        {"prix de vente": {"trouve": True, "valeur": "500000", "valeurs": [{"valeur": "500000", "citations": ["S1"]}]}},
        sources=[SourceOutil(doc_id="d", source="s", nom_fichier="d.txt", extrait="...")],
    )
    r = noter_extract(cas, _brut(_session_extract(), res), {})
    assert r.details["champs"]["prix de vente"]["trouve"] is True
    assert r.final_verdict == "WRONG"
    assert r.failure_category == "EXTRACTION_FAILURE"


def test_valeur_champ_extrait_prefere_la_valeur_unique():
    assert _valeur_champ_extrait({"valeur": "X", "valeurs": [{"valeur": "Y"}]}) == "X"


def test_valeur_champ_extrait_repli_sur_valeurs_multiples():
    entree = {"valeur": None, "valeurs": [{"valeur": "20 septembre 1792"}, {"valeur": "septembre 1792"}]}
    assert _valeur_champ_extrait(entree) == "20 septembre 1792 septembre 1792"


def test_valeur_champ_extrait_vide():
    assert _valeur_champ_extrait({"valeur": None, "valeurs": []}) == ""
    assert _valeur_champ_extrait({}) == ""


def test_noter_extract_champ_multi_valeurs_precision_mesuree_sur_toutes():
    """`valeur` vaut None quand il y a plusieurs valeurs : la précision est
    mesurée sur la concaténation des `valeurs[].valeur`, pas sur ''."""
    gold = Enregistrement(
        id="g1",
        question="q",
        expected_answer="20 septembre 1792",
        evidence_text="La bataille de Valmy, 20 septembre 1792.",
    )
    cas = _cas_extract([{"label": "date", "gold_qid": "g1", "trouve_attendu": True}])
    res = _resultat_extract(
        {
            "date": {
                "trouve": True,
                "valeur_unique": False,
                "valeur": None,
                "valeurs": [
                    {"valeur": "20 septembre 1792", "citations": ["S1"], "justification": ""},
                    {"valeur": "septembre 1792", "citations": ["S2"], "justification": ""},
                ],
            }
        },
        sources=[SourceOutil(doc_id="d", source="s", nom_fichier="d.txt", extrait="...")],
    )
    r = noter_extract(cas, _brut(_session_extract(), res), {"g1": gold})
    detail = r.details["champs"]["date"]
    assert detail["trouve"] is True
    assert detail["precision_valeur"] >= 0.3
    assert r.final_verdict == "PASS"


# ===========================================================================
# Défaut 2 (suite) — le harnais doit toujours ÉCHOUER sur un mauvais document
# ===========================================================================


def test_cle_document_resolu_nom_fichier_et_source_equivalents(monkeypatch):
    """Un chemin complet et un nom de fichier nu qui désignent le même
    document produisent la même clé (via `cle_document`), que la traduction
    par catalogue aboutisse ou non."""
    monkeypatch.setattr(
        "src.rag.retrieval.catalogue",
        lambda: _catalogue_factice({"UUID-9": "corpus/2024/rapport_9.txt"}),
    )
    assert _cle_document_resolu("UUID-9") == cle_document("rapport_9.txt")
    assert _cle_document_resolu("data/in/rapport_9.txt") == cle_document("rapport_9.txt")


def test_noter_classify_mauvais_document_reste_wrong(monkeypatch):
    """L'UUID de la trace se résout vers un AUTRE document que celui du gold :
    `document_demande_ok` doit rester faux et le verdict WRONG /
    DOCUMENT_RETRIEVAL_FAILURE — la correction ne masque pas une vraie
    erreur de périmètre."""
    monkeypatch.setattr(
        "src.rag.retrieval.catalogue",
        lambda: _catalogue_factice({"UUID-AUTRE": "cquae_doc_999.txt"}),
    )
    cas = CasSmoke(
        test_id="CL-XX",
        capability="CLASSIFY",
        query="Classe le document cquae_doc_219.txt.",
        expected_tool="classify",
        native_gold=False,
        evaluation_type="contract_validation",
        source_document="cquae_doc_219.txt",
    )
    session = _FausseSession(
        [
            _FausseEtape("intention", intention="classify", desambiguisation_llm=False),
            _FausseEtape("classify", mode="document_complet", document_demande="UUID-AUTRE"),
        ],
        ["classify"],
    )
    resultat_outil = ResultatOutil(
        outil="classify", succes=True, message="ok",
        donnees={"categorie": "autre", "citations": ["S1"]},
        sources=[SourceOutil(doc_id="UUID-AUTRE", source="s", nom_fichier="cquae_doc_999.txt", extrait="...")],
    )
    r = noter_classify(
        cas, _brut(session, resultat_outil),
        ["contrat", "facture", "rapport", "correspondance", "identite", "technique", "autre"],
    )
    assert r.details["document_demande_ok"] is False
    assert r.final_verdict == "WRONG"
    assert r.failure_category == "DOCUMENT_RETRIEVAL_FAILURE"


def test_noter_summarize_mauvais_document_reste_wrong(monkeypatch):
    monkeypatch.setattr(
        "src.rag.retrieval.catalogue",
        lambda: _catalogue_factice({"UUID-AUTRE": "cquae_doc_999.txt"}),
    )
    cas = CasSmoke(
        test_id="SU-XX",
        capability="SUMMARIZE",
        query="Résume le document cquae_doc_2920.txt.",
        expected_tool="summarize",
        native_gold=False,
        evaluation_type="structural_and_faithfulness",
        source_document="cquae_doc_2920.txt",
        gold_qids=[],
    )
    session = _FausseSession(
        [
            _FausseEtape("intention", intention="summarize", desambiguisation_llm=False),
            _FausseEtape("summarize", documents_demandes=["UUID-AUTRE"], succes=True),
        ],
        ["summarize"],
    )
    resultat_outil = ResultatOutil(
        outil="summarize", succes=True, message="ok",
        donnees={"resume": "Un résumé cohérent et sourcé [S1]."},
        sources=[SourceOutil(doc_id="UUID-AUTRE", source="s", nom_fichier="cquae_doc_999.txt", extrait="...")],
    )
    r = noter_summarize(cas, _brut(session, resultat_outil), {})
    assert r.details["structural_ok"] is False
    assert r.final_verdict == "WRONG"
    assert r.failure_category == "DOCUMENT_RETRIEVAL_FAILURE"


def test_noter_summarize_document_absent_conserve_l_abstention(monkeypatch):
    """Document visé mais non résolu de façon fiable : le harnais ne doit pas
    inventer un PASS. `structural_ok` faux (aucune source) -> WRONG, et jamais
    un crédit silencieux."""
    monkeypatch.setattr("src.rag.retrieval.catalogue", lambda: _catalogue_factice({}))
    cas = CasSmoke(
        test_id="SU-YY",
        capability="SUMMARIZE",
        query="Résume le document cquae_doc_2262.txt.",
        expected_tool="summarize",
        native_gold=False,
        evaluation_type="structural_and_faithfulness",
        source_document="cquae_doc_2262.txt",
        gold_qids=[],
    )
    session = _FausseSession(
        [
            _FausseEtape("intention", intention="summarize", desambiguisation_llm=False),
            _FausseEtape("summarize", documents_demandes=None, succes=False, mode="document_vise_non_resolu"),
        ],
        ["summarize"],
    )
    resultat_outil = ResultatOutil.echec("summarize", "Document à résumer non identifié de façon fiable.")
    r = noter_summarize(cas, _brut(session, resultat_outil), {})
    assert r.details["structural_ok"] is False
    assert r.final_verdict == "WRONG"


# ===========================================================================
# Défaut 3 (suite) — une valeur réellement fausse n'est jamais créditée
# ===========================================================================


def test_noter_extract_valeur_fausse_nest_pas_creditee():
    """Champ trouvé, attendu trouvé, mais la valeur extraite n'a rien à voir
    avec la référence gold (précision < 0.3) : le cas ne doit PAS passer
    (PASS interdit). UNSUPPORTED / PROVENANCE_FAILURE est acceptable."""
    gold = Enregistrement(
        id="g1", question="q",
        expected_answer="20 septembre 1792",
        evidence_text="La bataille de Valmy eut lieu le 20 septembre 1792.",
    )
    cas = _cas_extract([{"label": "date de la bataille", "gold_qid": "g1", "trouve_attendu": True}])
    res = _resultat_extract(
        {"date de la bataille": {"trouve": True, "valeur_unique": True,
                                 "valeur": "un millier de fantassins prussiens", "valeurs": []}},
        sources=[SourceOutil(doc_id="d", source="s", nom_fichier="d.txt", extrait="...")],
    )
    r = noter_extract(cas, _brut(_session_extract(), res), {"g1": gold})
    detail = r.details["champs"]["date de la bataille"]
    assert detail["precision_valeur"] < 0.3
    assert r.final_verdict != "PASS"


def test_noter_extract_multi_valeurs_dont_une_hors_sujet_reste_acceptable():
    """Pluralité légitime : la bonne valeur est présente parmi plusieurs. La
    précision agrégée reste au-dessus du seuil, le cas passe."""
    gold = Enregistrement(
        id="g1", question="q",
        expected_answer="20 septembre 1792",
        evidence_text="La bataille de Valmy, 20 septembre 1792, opposa Français et Prussiens.",
    )
    cas = _cas_extract([{"label": "date", "gold_qid": "g1", "trouve_attendu": True}])
    res = _resultat_extract(
        {"date": {"trouve": True, "valeur_unique": False, "valeur": None,
                  "valeurs": [{"valeur": "20 septembre 1792"}, {"valeur": "1792"}]}},
        sources=[SourceOutil(doc_id="d", source="s", nom_fichier="d.txt", extrait="...")],
    )
    r = noter_extract(cas, _brut(_session_extract(), res), {"g1": gold})
    assert r.final_verdict == "PASS"
