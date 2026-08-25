"""
Tests pytest du mode « document complet » de l'outil `classify` (Option E —
classification hiérarchique par lots + agrégation déterministe Python).

Aucun Ollama, aucun Qdrant réel : `catalogue`/`charger_document` sont
injectés exactement comme dans `tests/tools/test_summarize.py` (même
patron), et le LLM est une doublure scriptée.

Le mode historique (Cas B, `ContexteOutil.sources`) est couvert par
`tests/tools/test_classify.py`, non modifié par cette mission : ce fichier
ne teste QUE le nouveau chemin `documents=[...]`.
"""

from __future__ import annotations

import json
import re

import pytest
from langchain_core.messages import AIMessage

from src.rag.retrieval import CollectionIndisponible, DocumentInconnu, Passage, PerimetreDocumentaire
from src.tools import classify
from src.tools.base import ContexteOutil


# ===========================================================================
# Fabriques
# ===========================================================================


def _passage(doc_id: str, chunk_index: int, texte: str, *, page: int = 1, nom_fichier: str = "doc.pdf") -> Passage:
    rang = chunk_index + 1
    return Passage(
        citation=f"S{rang}",
        rang=rang,
        point_id=f"{doc_id}-{chunk_index}",
        doc_id=doc_id,
        chunk_index=chunk_index,
        texte=texte,
        source=nom_fichier,
        nom_fichier=nom_fichier,
        page=page,
        categorie="",
        score_recherche=0.0,
        score_reranking=None,
        payload={},
    )


def _perimetre_exact(doc_id: str) -> PerimetreDocumentaire:
    return PerimetreDocumentaire(statut="exact", valeurs_filtre=(doc_id,), libelles=(doc_id,))


def _perimetre_compatible(doc_ids: tuple[str, ...]) -> PerimetreDocumentaire:
    return PerimetreDocumentaire(statut="compatible", valeurs_filtre=doc_ids, libelles=doc_ids)


class _FauxCatalogue:
    def __init__(self, perimetre_ou_exception):
        self._p = perimetre_ou_exception

    def perimetre_explicite(self, documents):
        if isinstance(self._p, Exception):
            raise self._p
        return self._p


def _resoudre_vers(monkeypatch, doc_id: str, passages: list[Passage]) -> None:
    """Câble la résolution documentaire + le chargement pour un seul document connu."""
    monkeypatch.setattr(classify, "get_profil", lambda: None)
    monkeypatch.setattr(classify, "catalogue", lambda profil=None: _FauxCatalogue(_perimetre_exact(doc_id)))
    monkeypatch.setattr(
        classify,
        "charger_document",
        lambda cible: passages if cible == doc_id else (_ for _ in ()).throw(DocumentInconnu(cible)),
    )


def _citations_du_prompt(texte: str) -> list[str]:
    vues: list[str] = []
    for m in re.findall(r"\[(S\d+)\]", texte):
        if m not in vues:
            vues.append(m)
    return vues


class LLMVotes:
    """
    LLM factice : renvoie la catégorie n-ième de ``categories_par_appel`` au
    n-ième appel (un appel = un lot, dans l'ordre où `_executer_classify_
    document_complet` les traite). ``None`` simule une abstention du lot
    (categorie=null). Cite par défaut toutes les citations réellement
    présentes dans le prompt reçu (comportement honnête).
    """

    def __init__(self, categories_par_appel: list[str | None], *, citer: bool = True, citations_forcees=None):
        self._categories = categories_par_appel
        self._citer = citer
        self._citations_forcees = citations_forcees
        self.appels: list[str] = []

    def invoke(self, messages):
        _systeme, utilisateur = messages[0].content, messages[1].content
        self.appels.append(utilisateur)
        indice = len(self.appels) - 1
        categorie = self._categories[indice] if indice < len(self._categories) else None

        if self._citations_forcees is not None:
            citations = self._citations_forcees
        elif self._citer and categorie is not None:
            citations = _citations_du_prompt(utilisateur)
        else:
            citations = []

        payload = {
            "categorie": categorie,
            "confiance": 0.9 if categorie is not None else 0.0,
            "sources": citations,
            "justification": "ok" if categorie is not None else "insuffisant",
        }
        return AIMessage(content=json.dumps(payload))


class _LLMExplose:
    def invoke(self, messages):
        raise RuntimeError("Ollama injoignable.")


class _LLMExploseUneFois:
    """Explose au premier appel, répond normalement ensuite."""

    def __init__(self, categorie_normale: str):
        self._categorie = categorie_normale
        self.appels = 0

    def invoke(self, messages):
        self.appels += 1
        if self.appels == 1:
            raise RuntimeError("Timeout lot 1.")
        _systeme, utilisateur = messages[0].content, messages[1].content
        citations = _citations_du_prompt(utilisateur)
        payload = {"categorie": self._categorie, "confiance": 0.9, "sources": citations, "justification": "ok"}
        return AIMessage(content=json.dumps(payload))


class _LLMExplosePartiellement:
    """Explose aux appels dont le numéro (1-indexé) est dans ``indices_erreur``,
    vote normalement pour ``categorie_normale`` sinon."""

    def __init__(self, indices_erreur: set[int], categorie_normale: str):
        self._indices_erreur = indices_erreur
        self._categorie = categorie_normale
        self.appels = 0

    def invoke(self, messages):
        self.appels += 1
        if self.appels in self._indices_erreur:
            raise RuntimeError(f"Timeout lot {self.appels}.")
        _systeme, utilisateur = messages[0].content, messages[1].content
        citations = _citations_du_prompt(utilisateur)
        payload = {"categorie": self._categorie, "confiance": 0.9, "sources": citations, "justification": "ok"}
        return AIMessage(content=json.dumps(payload))


def _contexte(llm) -> ContexteOutil:
    return ContexteOutil(question="peu importe", llm=llm, sources=[])


CATEGORIES = ["rapport", "technique", "contrat"]


@pytest.fixture(autouse=True)
def _limite_lot_par_defaut(monkeypatch):
    """Restaure la vraie limite au besoin ; chaque test qui veut forcer
    plusieurs lots la réduit explicitement lui-même."""
    yield


def _executer(contexte, *, categories=CATEGORIES, documents=("A",), critere=None, instruction=None):
    return classify.definir_classify().executer(
        contexte=contexte,
        categories=categories,
        documents=list(documents),
        critere=critere,
        instruction=instruction,
    )


# ===========================================================================
# A — document mono-lot
# ===========================================================================


def test_document_mono_lot_classification_correcte(monkeypatch):
    passages = [_passage("A", i, f"contenu {i}") for i in range(3)]
    _resoudre_vers(monkeypatch, "A", passages)

    llm = LLMVotes(["rapport"])
    resultat = _executer(_contexte(llm))

    assert resultat.succes
    assert resultat.donnees["categorie"] == "rapport"
    assert resultat.donnees["nombre_lots"] == 1
    assert resultat.donnees["nombre_passages"] == 3
    assert len(llm.appels) == 1


# ===========================================================================
# B — document multi-lots : couverture totale, chaque passage vu une fois
# ===========================================================================


def test_document_multi_lots_couverture_totale(monkeypatch):
    passages = [_passage("A", i, f"contenu numero {i} " * 5) for i in range(12)]
    _resoudre_vers(monkeypatch, "A", passages)
    monkeypatch.setattr(classify, "LIMITE_CARACTERES_LOT_CLASSIFY", 300)

    llm = LLMVotes(["rapport"] * 20)  # assez pour couvrir tous les lots générés
    resultat = _executer(_contexte(llm))

    assert resultat.succes
    assert resultat.donnees["nombre_lots"] > 1

    citations_vues: set[str] = set()
    for utilisateur in llm.appels:
        vues_ce_lot = set(_citations_du_prompt(utilisateur))
        # aucune citation ne doit apparaître dans deux lots différents
        assert not (citations_vues & vues_ce_lot)
        citations_vues |= vues_ce_lot

    assert citations_vues == {p.citation for p in passages}


# ===========================================================================
# C — 200+ chunks simulés : aucune troncature silencieuse
# ===========================================================================


def test_document_volumineux_aucune_troncature_silencieuse(monkeypatch):
    passages = [_passage("A", i, f"contenu {i} " * 8) for i in range(220)]
    _resoudre_vers(monkeypatch, "A", passages)
    monkeypatch.setattr(classify, "LIMITE_CARACTERES_LOT_CLASSIFY", 400)

    llm = LLMVotes(["rapport"] * 200)
    resultat = _executer(_contexte(llm))

    assert resultat.succes
    assert resultat.donnees["nombre_passages"] == 220

    citations_vues: set[str] = set()
    for utilisateur in llm.appels:
        citations_vues |= set(_citations_du_prompt(utilisateur))
    assert citations_vues == {p.citation for p in passages}
    assert len(citations_vues) == 220


def test_partitionner_document_un_passage_depasse_le_budget_seul(monkeypatch):
    """Cas explicite (Phase 2) : un passage à lui seul plus gros que la limite
    forme son propre lot au lieu de disparaître."""
    petit = _passage("A", 0, "court")
    enorme = _passage("A", 1, "x" * 5_000)
    petit2 = _passage("A", 2, "court aussi")

    paires = [
        ("S1", classify._source_depuis_passage(petit)),
        ("S2", classify._source_depuis_passage(enorme)),
        ("S3", classify._source_depuis_passage(petit2)),
    ]

    lots = classify._partitionner_document(paires, limite_caracteres=1_000)

    toutes_citations = [c for lot in lots for c, _ in lot]
    assert sorted(toutes_citations) == ["S1", "S2", "S3"]
    # Le passage énorme doit apparaître dans un lot, jamais disparaître.
    assert any(("S2", paires[1][1]) in lot for lot in lots)


# ===========================================================================
# D — catégorie dominante claire : décision déterministe correcte
# ===========================================================================


def test_categorie_dominante_claire_decision_correcte(monkeypatch):
    passages = [_passage("A", i, f"contenu {i}") for i in range(5)]
    _resoudre_vers(monkeypatch, "A", passages)
    monkeypatch.setattr(classify, "LIMITE_CARACTERES_LOT_CLASSIFY", 60)  # 1 passage/lot environ

    llm = LLMVotes(["rapport", "rapport", "rapport", "rapport", "technique"])
    resultat = _executer(_contexte(llm))

    assert resultat.succes
    assert resultat.donnees["categorie"] == "rapport"
    assert resultat.donnees["votes_par_categorie"]["rapport"] == 4


# ===========================================================================
# E — égalité entre deux catégories -> abstention
# ===========================================================================


def test_egalite_entre_deux_categories_abstention(monkeypatch):
    passages = [_passage("A", i, f"contenu {i}") for i in range(2)]
    _resoudre_vers(monkeypatch, "A", passages)
    monkeypatch.setattr(classify, "LIMITE_CARACTERES_LOT_CLASSIFY", 60)

    llm = LLMVotes(["rapport", "technique"])
    resultat = _executer(_contexte(llm))

    assert resultat.succes
    assert resultat.donnees["categorie"] is None
    assert resultat.donnees["raison_abstention"] == "classification_ambigue"
    assert resultat.donnees["votes_par_categorie"] == {"rapport": 1, "technique": 1}


# ===========================================================================
# F — votes très partagés : comportement conservateur documenté
# ===========================================================================


def test_votes_tres_partages_abstention_conservatrice(monkeypatch):
    passages = [_passage("A", i, f"contenu {i}") for i in range(5)]
    _resoudre_vers(monkeypatch, "A", passages)
    monkeypatch.setattr(classify, "LIMITE_CARACTERES_LOT_CLASSIFY", 60)

    llm = LLMVotes(["rapport", "technique", "contrat", "rapport", "technique"])
    resultat = _executer(_contexte(llm))

    assert resultat.succes
    assert resultat.donnees["categorie"] is None
    assert resultat.donnees["raison_abstention"] == "classification_ambigue"
    # 2/5 ne constitue pas une majorité absolue (2*2=4, pas > 5).
    assert max(resultat.donnees["votes_par_categorie"].values()) * 2 <= 5


# ===========================================================================
# G — aucune classification valide -> abstention explicite
# ===========================================================================


def test_aucune_classification_valide_abstention_explicite(monkeypatch):
    passages = [_passage("A", i, f"contenu {i}") for i in range(3)]
    _resoudre_vers(monkeypatch, "A", passages)
    monkeypatch.setattr(classify, "LIMITE_CARACTERES_LOT_CLASSIFY", 60)

    llm = LLMVotes([None, None, None])
    resultat = _executer(_contexte(llm))

    assert resultat.succes  # comportement établi : "categorie=None" est un succès, pas un échec
    assert resultat.donnees["categorie"] is None
    assert resultat.donnees["raison_abstention"] == "aucune_classification_valide"
    assert resultat.donnees["lots_valides"] == 0


# ===========================================================================
# H — catégorie LLM hors taxonomie -> rejetée
# ===========================================================================


def test_categorie_hors_taxonomie_rejetee(monkeypatch):
    passages = [_passage("A", 0, "contenu")]
    _resoudre_vers(monkeypatch, "A", passages)

    llm = LLMVotes(["Categorie Inexistante"])
    resultat = _executer(_contexte(llm))

    assert resultat.succes
    assert resultat.donnees["categorie"] is None
    assert resultat.donnees["raison_abstention"] == "aucune_classification_valide"


# ===========================================================================
# I — classification positive sans citation valide -> vote rejeté
# ===========================================================================


def test_classification_sans_citation_vote_rejete(monkeypatch):
    passages = [_passage("A", 0, "contenu")]
    _resoudre_vers(monkeypatch, "A", passages)

    llm = LLMVotes(["rapport"], citations_forcees=[])
    resultat = _executer(_contexte(llm))

    assert resultat.succes
    assert resultat.donnees["categorie"] is None
    assert resultat.donnees["raison_abstention"] == "aucune_classification_valide"


# ===========================================================================
# J — citation inventée -> rejetée
# ===========================================================================


def test_citation_inventee_rejetee(monkeypatch):
    passages = [_passage("A", 0, "contenu")]
    _resoudre_vers(monkeypatch, "A", passages)

    llm = LLMVotes(["rapport"], citations_forcees=["S99"])
    resultat = _executer(_contexte(llm))

    assert resultat.succes
    assert resultat.donnees["categorie"] is None
    assert resultat.donnees["raison_abstention"] == "aucune_classification_valide"


# ===========================================================================
# K — citations identiques (S1, S2...) entre lots différents : aucune collision
# ===========================================================================


def test_citations_uniques_sur_tout_le_document_aucune_collision(monkeypatch):
    passages = [_passage("A", i, f"contenu {i}", page=i + 1) for i in range(6)]
    _resoudre_vers(monkeypatch, "A", passages)
    monkeypatch.setattr(classify, "LIMITE_CARACTERES_LOT_CLASSIFY", 60)  # ~1 passage/lot

    # Chaque lot vote "rapport" : la catégorie gagnante doit agréger les
    # citations de TOUS les lots gagnants, sans qu'aucune ne soit dupliquée
    # ou perdue malgré la numérotation continue S1..S6 à travers les lots.
    llm = LLMVotes(["rapport"] * 6)
    resultat = _executer(_contexte(llm))

    assert resultat.succes
    assert resultat.donnees["categorie"] == "rapport"
    assert resultat.donnees["nombre_lots"] == 6
    assert sorted(resultat.donnees["citations"]) == [f"S{i}" for i in range(1, 7)]
    assert len(resultat.sources) == 6
    # La provenance doit pointer vers les VRAIS passages d'origine (pages
    # distinctes 1..6), pas un mélange ou une perte.
    assert sorted(s.page for s in resultat.sources) == [1, 2, 3, 4, 5, 6]
    doc_ids_pages = {(s.page, s.extrait) for s in resultat.sources}
    attendu = {(p.page, p.texte) for p in passages}
    assert doc_ids_pages == attendu


# ===========================================================================
# L — document introuvable -> échec propre
# ===========================================================================


def test_document_introuvable_echec_propre(monkeypatch):
    monkeypatch.setattr(classify, "get_profil", lambda: None)
    monkeypatch.setattr(
        classify, "catalogue", lambda profil=None: _FauxCatalogue(DocumentInconnu("Z introuvable"))
    )

    resultat = _executer(_contexte(LLMVotes(["rapport"])), documents=("Z",))

    assert not resultat.succes
    assert "introuvable" in resultat.message


# ===========================================================================
# M — document ambigu -> échec propre
# ===========================================================================


def test_document_ambigu_echec_propre(monkeypatch):
    monkeypatch.setattr(classify, "get_profil", lambda: None)
    monkeypatch.setattr(
        classify,
        "catalogue",
        lambda profil=None: _FauxCatalogue(_perimetre_compatible(("A", "B"))),
    )

    resultat = _executer(_contexte(LLMVotes(["rapport"])), documents=("A et B",))

    assert not resultat.succes
    assert "mélange jamais" in resultat.message


# ===========================================================================
# N — erreur LLM sur un lot -> pipeline contrôlé, aucune exception non gérée
# ===========================================================================


def test_erreur_llm_sur_un_lot_minoritaire_dans_2_lots_abstention(monkeypatch):
    """
    Un seul lot valide sur 2 lots au total (1 erreur) : 1*2=2, pas > 2
    (total_lots, PAS lots_valides) -> abstention. Avant la correction de
    cette mission, la majorité se serait calculée sur lots_valides=1 seul et
    aurait accepté "rapport" à tort (1/1) ; c'est exactement le cas que la
    nouvelle règle (dénominateur = tous les lots) neutralise.
    """
    passages = [_passage("A", i, f"contenu {i}") for i in range(2)]
    _resoudre_vers(monkeypatch, "A", passages)
    monkeypatch.setattr(classify, "LIMITE_CARACTERES_LOT_CLASSIFY", 60)  # 2 lots

    llm = _LLMExploseUneFois("rapport")
    resultat = _executer(_contexte(llm))

    assert resultat.succes
    assert resultat.donnees["categorie"] is None
    assert resultat.donnees["raison_abstention"] == "classification_ambigue"
    assert resultat.donnees["nombre_total_lots"] == 2
    assert resultat.donnees["lots_valides"] == 1
    assert resultat.donnees["lots_invalides"] == 1
    assert any("erreur technique" in a for a in resultat.avertissements)


def test_erreur_llm_sur_un_lot_majorite_atteinte_malgre_erreur(monkeypatch):
    """Contre-exemple : la majorité absolue sur TOUS les lots reste atteignable
    malgré une erreur, tant que les votes valides suffisent (2/3 > 50%)."""
    passages = [_passage("A", i, f"contenu {i}") for i in range(3)]
    _resoudre_vers(monkeypatch, "A", passages)
    monkeypatch.setattr(classify, "LIMITE_CARACTERES_LOT_CLASSIFY", 60)  # 3 lots

    llm = _LLMExplosePartiellement(indices_erreur={2}, categorie_normale="rapport")
    resultat = _executer(_contexte(llm))

    assert resultat.succes
    assert resultat.donnees["categorie"] == "rapport"
    assert resultat.donnees["nombre_total_lots"] == 3
    assert resultat.donnees["lots_valides"] == 2
    assert resultat.donnees["lots_invalides"] == 1


def test_erreur_llm_sur_tous_les_lots_ne_leve_pas_exception(monkeypatch):
    passages = [_passage("A", 0, "contenu")]
    _resoudre_vers(monkeypatch, "A", passages)

    resultat = _executer(_contexte(_LLMExplose()))

    assert resultat.succes
    assert resultat.donnees["categorie"] is None
    assert resultat.donnees["raison_abstention"] == "aucune_classification_valide"


# ===========================================================================
# Majorité absolue sur TOUS les lots (correction ciblée) — cas obligatoires
# ===========================================================================


def _llm_votes_puis_abstentions(n_votes: int, categorie: str, n_total: int) -> LLMVotes:
    """``n_votes`` lots votent ``categorie``, les suivants s'abstiennent (categorie=null)."""
    return LLMVotes([categorie] * n_votes + [None] * (n_total - n_votes))


def test_1_vote_valide_sur_20_lots_abstention(monkeypatch):
    """1 vote valide / 20 lots -> abstention (1*2=2, pas > 20)."""
    passages = [_passage("A", i, f"contenu {i}") for i in range(20)]
    _resoudre_vers(monkeypatch, "A", passages)
    monkeypatch.setattr(classify, "LIMITE_CARACTERES_LOT_CLASSIFY", 60)  # ~1 passage/lot -> 20 lots

    llm = _llm_votes_puis_abstentions(1, "rapport", 20)
    resultat = _executer(_contexte(llm))

    assert resultat.succes
    assert resultat.donnees["nombre_total_lots"] == 20
    assert resultat.donnees["categorie"] is None
    assert resultat.donnees["raison_abstention"] == "classification_ambigue"
    assert resultat.donnees["votes_par_categorie"] == {"rapport": 1}


def test_10_sur_20_votes_gagnants_abstention(monkeypatch):
    """10/20 -> exactement la moitié, pas une majorité STRICTE -> abstention."""
    passages = [_passage("A", i, f"contenu {i}") for i in range(20)]
    _resoudre_vers(monkeypatch, "A", passages)
    monkeypatch.setattr(classify, "LIMITE_CARACTERES_LOT_CLASSIFY", 60)

    llm = _llm_votes_puis_abstentions(10, "rapport", 20)
    resultat = _executer(_contexte(llm))

    assert resultat.succes
    assert resultat.donnees["nombre_total_lots"] == 20
    assert resultat.donnees["categorie"] is None
    assert resultat.donnees["raison_abstention"] == "classification_ambigue"
    assert resultat.donnees["votes_par_categorie"]["rapport"] == 10


def test_11_sur_20_votes_gagnants_acceptation(monkeypatch):
    """11/20 -> majorité absolue stricte atteinte (11*2=22 > 20) -> acceptée."""
    passages = [_passage("A", i, f"contenu {i}") for i in range(20)]
    _resoudre_vers(monkeypatch, "A", passages)
    monkeypatch.setattr(classify, "LIMITE_CARACTERES_LOT_CLASSIFY", 60)

    llm = _llm_votes_puis_abstentions(11, "rapport", 20)
    resultat = _executer(_contexte(llm))

    assert resultat.succes
    assert resultat.donnees["nombre_total_lots"] == 20
    assert resultat.donnees["categorie"] == "rapport"
    assert resultat.donnees["raison_abstention"] is None


def test_erreurs_llm_nombreuses_restent_au_denominateur(monkeypatch):
    """
    1 lot vote "rapport", 19 lots échouent techniquement (erreur LLM, pas une
    abstention "propre" du LLM) -> abstention : les erreurs comptent dans le
    dénominateur exactement comme les abstentions propres.
    """
    passages = [_passage("A", i, f"contenu {i}") for i in range(20)]
    _resoudre_vers(monkeypatch, "A", passages)
    monkeypatch.setattr(classify, "LIMITE_CARACTERES_LOT_CLASSIFY", 60)

    llm = _LLMExplosePartiellement(indices_erreur=set(range(2, 21)), categorie_normale="rapport")
    resultat = _executer(_contexte(llm))

    assert resultat.succes
    assert resultat.donnees["nombre_total_lots"] == 20
    assert resultat.donnees["lots_valides"] == 1
    assert resultat.donnees["lots_invalides"] == 19
    assert resultat.donnees["categorie"] is None
    assert resultat.donnees["raison_abstention"] == "classification_ambigue"


def test_citations_invalides_restent_au_denominateur(monkeypatch):
    """
    10 lots : 5 votent "rapport" avec citation valide, 5 votent "rapport"
    SANS citation valide (rejetés). Avec l'ancienne règle (dénominateur =
    lots_valides = 5), 5/5 aurait été accepté à tort. Avec la nouvelle règle
    (dénominateur = 10 lots au total), 5*2=10 n'est pas strictement > 10 ->
    abstention : les lots sans citation valide comptent contre la majorité.
    """
    passages = [_passage("A", i, f"contenu {i}") for i in range(10)]
    _resoudre_vers(monkeypatch, "A", passages)
    monkeypatch.setattr(classify, "LIMITE_CARACTERES_LOT_CLASSIFY", 60)  # 10 lots

    llm = LLMVotes(
        ["rapport"] * 10,
        citations_forcees=None,  # honnête par défaut : cite ce qu'il voit
    )
    # On force artificiellement les 5 derniers appels à ne citer rien du tout
    # en remplaçant la doublure par une version qui alterne.
    class _LLMCitationsPartielles:
        def __init__(self):
            self.appels = 0

        def invoke(self, messages):
            self.appels += 1
            _systeme, utilisateur = messages[0].content, messages[1].content
            citations = _citations_du_prompt(utilisateur) if self.appels <= 5 else []
            payload = {
                "categorie": "rapport",
                "confiance": 0.9,
                "sources": citations,
                "justification": "ok",
            }
            return AIMessage(content=json.dumps(payload))

    resultat = _executer(_contexte(_LLMCitationsPartielles()))

    assert resultat.succes
    assert resultat.donnees["nombre_total_lots"] == 10
    assert resultat.donnees["lots_valides"] == 5
    assert resultat.donnees["lots_invalides"] == 5
    assert resultat.donnees["categorie"] is None
    assert resultat.donnees["raison_abstention"] == "classification_ambigue"


# ===========================================================================
# O / P — aucun search, aucun embedding, aucun reranker sur ce chemin
# ===========================================================================


def test_aucun_appel_rechercher_passages_embeddings_reranker(monkeypatch):
    passages = [_passage("A", i, f"contenu {i}") for i in range(3)]
    _resoudre_vers(monkeypatch, "A", passages)

    def _explose(*args, **kwargs):
        raise AssertionError("Ne doit jamais être appelé en mode document complet.")

    import src.rag.embeddings as embeddings_mod
    import src.rag.retrieval as retrieval_mod

    monkeypatch.setattr(retrieval_mod, "rechercher_passages", _explose)
    monkeypatch.setattr(embeddings_mod, "encoder_requete", _explose)
    monkeypatch.setattr(embeddings_mod, "reranker", _explose)

    llm = LLMVotes(["rapport"])
    resultat = _executer(_contexte(llm))

    assert resultat.succes
    assert resultat.donnees["categorie"] == "rapport"


# ===========================================================================
# Q — document vide -> échec propre
# ===========================================================================


def test_document_resolu_sans_contenu_indexe_echec_propre(monkeypatch):
    _resoudre_vers(monkeypatch, "A", [])

    resultat = _executer(_contexte(LLMVotes(["rapport"])))

    assert not resultat.succes
    assert "aucun contenu indexé" in resultat.message


def test_collection_indisponible_echec_propre(monkeypatch):
    monkeypatch.setattr(classify, "get_profil", lambda: None)
    monkeypatch.setattr(classify, "catalogue", lambda profil=None: _FauxCatalogue(_perimetre_exact("A")))
    monkeypatch.setattr(
        classify,
        "charger_document",
        lambda cible: (_ for _ in ()).throw(CollectionIndisponible("absente")),
    )

    resultat = _executer(_contexte(LLMVotes(["rapport"])))

    assert not resultat.succes
    assert "Corpus indisponible" in resultat.message


# ===========================================================================
# T — profil fictif complètement différent : généricité, aucune catégorie
# codée en dur (pas de mot-clé ESG/CNIL/juridique dans le code de classify)
# ===========================================================================


def test_generique_profil_fictif_sans_rapport_avec_taxonomie(monkeypatch):
    passages = [_passage("A", i, f"contenu {i}") for i in range(3)]
    _resoudre_vers(monkeypatch, "A", passages)

    categories_fictives = ["Poeme", "Recette de cuisine", "Facture d'energie"]
    llm = LLMVotes(["Recette de cuisine"])
    resultat = _executer(_contexte(llm), categories=categories_fictives)

    assert resultat.succes
    assert resultat.donnees["categorie"] == "Recette de cuisine"
    assert resultat.donnees["categories_autorisees"] == categories_fictives
