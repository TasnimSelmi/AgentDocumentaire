"""
Tests pytest de l'outil `summarize` (Action 03A).

Deux modes testés séparément :
    - Cas A, document complet nommé (``documents=[...]``) : la résolution
      documentaire (``CatalogueDocuments``) et ``charger_document`` sont
      injectées, sur le modèle de ``tests/rag/test_document_access.py``.
    - Cas B, résumé de ``ContexteOutil.sources`` (comportement historique).

Le LLM est une doublure scriptée (``LLMScripte``), sur le modèle de
``tests/agent/test_nodes.py`` : aucun Ollama, aucun Qdrant, déterministe.
"""

from __future__ import annotations

import re

import pytest
from langchain_core.messages import AIMessage

from src.rag.retrieval import (
    CollectionIndisponible,
    DocumentInconnu,
    Passage,
    PerimetreDocumentaire,
)
from src.tools import summarize
from src.tools.base import ContexteOutil, SourceOutil


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
        categorie="autre",
        score_recherche=0.0,
        score_reranking=None,
        payload={},
    )


class LLMScripte:
    """LLM factice : délègue la réponse à une fonction fournie par le test."""

    def __init__(self, repondre):
        self._repondre = repondre
        self.appels: list[tuple[str, str]] = []

    def invoke(self, messages):
        systeme, utilisateur = messages[0].content, messages[1].content
        self.appels.append((systeme, utilisateur))
        return AIMessage(content=self._repondre(systeme, utilisateur, len(self.appels)))


class _LLMExplose:
    def invoke(self, messages):
        raise RuntimeError("Ollama injoignable.")


def _cite_tout(systeme: str, utilisateur: str, numero_appel: int) -> str:
    """LLM 'honnête' : ne cite que ce qu'il voit réellement dans le prompt."""
    citations = list(dict.fromkeys(re.findall(r"\[S\d+\]", utilisateur)))
    return "Synthèse. " + " ".join(citations)


def _contexte(llm, sources: list[SourceOutil] | None = None) -> ContexteOutil:
    return ContexteOutil(question="peu importe", llm=llm, sources=list(sources or []))


def _perimetre_exact(doc_id: str) -> PerimetreDocumentaire:
    return PerimetreDocumentaire(statut="exact", valeurs_filtre=(doc_id,), libelles=(doc_id,))


class _FauxCatalogue:
    def __init__(self, perimetre_ou_exception):
        self._p = perimetre_ou_exception

    def perimetre_explicite(self, documents):
        if isinstance(self._p, Exception):
            raise self._p
        return self._p


def _resoudre_vers(monkeypatch, doc_id: str, passages: list[Passage]) -> None:
    """Câble la résolution documentaire + le chargement pour un seul document connu."""
    monkeypatch.setattr(summarize, "get_profil", lambda: None)
    monkeypatch.setattr(summarize, "catalogue", lambda profil=None: _FauxCatalogue(_perimetre_exact(doc_id)))
    monkeypatch.setattr(
        summarize,
        "charger_document",
        lambda cible: passages if cible == doc_id else (_ for _ in ()).throw(DocumentInconnu(cible)),
    )


def _citations_du_prompt(texte: str) -> set[str]:
    return set(re.findall(r"\[S\d+\]", texte))


def _perimetre_compatible(doc_ids: tuple[str, ...]) -> PerimetreDocumentaire:
    return PerimetreDocumentaire(statut="compatible", valeurs_filtre=doc_ids, libelles=doc_ids)


def _resoudre_vers_plusieurs(monkeypatch, documents_par_doc_id: dict[str, list[Passage]]) -> None:
    """
    Câble la résolution + le chargement pour PLUSIEURS documents.

    Chaque document est chargé avec des `Passage` dont la citation
    recommence à S1 (exactement le comportement réel de `charger_document`,
    Action 02) : c'est précisément la situation qui provoquait la collision.
    """
    monkeypatch.setattr(summarize, "get_profil", lambda: None)
    monkeypatch.setattr(
        summarize,
        "catalogue",
        lambda profil=None: _FauxCatalogue(_perimetre_compatible(tuple(documents_par_doc_id))),
    )
    monkeypatch.setattr(
        summarize,
        "charger_document",
        lambda cible: documents_par_doc_id.get(cible)
        or (_ for _ in ()).throw(DocumentInconnu(cible)),
    )


# ===========================================================================
# Test 1 — récupération complète : tous les chunks contribuent
# ===========================================================================


def test_document_complet_tous_les_chunks_contribuent(monkeypatch):
    passages = [_passage("A", i, f"contenu {i}") for i in range(4)]
    _resoudre_vers(monkeypatch, "A", passages)

    llm = LLMScripte(_cite_tout)
    outil = summarize.definir_summarize()
    resultat = outil.executer(contexte=_contexte(llm), documents=["A"])

    assert resultat.succes
    citations_vues: set[str] = set()
    for _, utilisateur in llm.appels:
        citations_vues |= _citations_du_prompt(utilisateur)
    assert citations_vues == {f"[{p.citation}]" for p in passages}
    assert resultat.donnees["nombre_sources_disponibles"] == 4


# ===========================================================================
# Test 2 — ne dépend pas du top-k de search
# ===========================================================================


def test_ignore_le_topk_du_contexte_existant(monkeypatch):
    passages_complets = [_passage("A", i, f"contenu {i}") for i in range(10)]
    _resoudre_vers(monkeypatch, "A", passages_complets)

    sources_topk = [
        SourceOutil(doc_id="A", source="doc.pdf", nom_fichier="doc.pdf", page=1, extrait="seulement 2 passages")
        for _ in range(2)
    ]

    llm = LLMScripte(_cite_tout)
    outil = summarize.definir_summarize()
    resultat = outil.executer(contexte=_contexte(llm, sources_topk), documents=["A"])

    assert resultat.succes
    citations_vues: set[str] = set()
    for _, utilisateur in llm.appels:
        citations_vues |= _citations_du_prompt(utilisateur)
    # Les 10 passages du document complet ont contribué, pas seulement les 2 du contexte.
    assert citations_vues == {f"[{p.citation}]" for p in passages_complets}


# ===========================================================================
# Test 3 — document volumineux : plusieurs lots, rien d'abandonné
# ===========================================================================


def test_document_volumineux_plusieurs_lots_puis_synthese(monkeypatch):
    passages = [_passage("A", i, f"contenu numero {i} " * 3) for i in range(8)]
    _resoudre_vers(monkeypatch, "A", passages)
    monkeypatch.setattr(summarize, "LIMITE_CARACTERES_LOT", 120)

    llm = LLMScripte(_cite_tout)
    outil = summarize.definir_summarize()
    resultat = outil.executer(contexte=_contexte(llm), documents=["A"])

    assert resultat.succes
    assert resultat.donnees["nombre_lots"] > 1

    # Chaque passage doit apparaître dans l'entrée d'au moins un appel LLM (map).
    citations_vues: set[str] = set()
    for _, utilisateur in llm.appels:
        citations_vues |= _citations_du_prompt(utilisateur)
    assert citations_vues == {f"[{p.citation}]" for p in passages}

    # La synthèse finale doit elle-même référencer l'ensemble des passages.
    assert set(resultat.donnees["citations_valides"]) == {p.citation for p in passages}


def test_synthese_hierarchique_a_plusieurs_niveaux(monkeypatch):
    """`_synthetiser` regroupe puis récurse quand un seul appel ne suffit pas."""
    monkeypatch.setattr(summarize, "LIMITE_CARACTERES_LOT", 50)

    textes = [f"résumé partiel numéro {i} bien assez long pour compter" for i in range(6)]
    llm = LLMScripte(lambda systeme, utilisateur, n: f"méta-résumé #{n}")
    contexte = _contexte(llm)

    resultat = summarize._synthetiser(
        textes, objectif=None, format_resume="court", contexte=contexte
    )

    assert resultat
    # Un seul appel n'aurait pas suffi : la réduction a dû se faire en plusieurs étapes.
    assert len(llm.appels) > 1


# ===========================================================================
# Test 4 — ordre documentaire respecté
# ===========================================================================


def test_ordre_des_lots_respecte_lordre_documentaire(monkeypatch):
    # charger_document renvoie déjà l'ordre correct (Action 02) ; ce test
    # vérifie que summarize ne le mélange pas en répartissant en lots.
    passages = [_passage("A", i, f"contenu {i}") for i in range(6)]
    _resoudre_vers(monkeypatch, "A", passages)
    monkeypatch.setattr(summarize, "LIMITE_CARACTERES_LOT", 90)

    llm = LLMScripte(_cite_tout)
    outil = summarize.definir_summarize()
    resultat = outil.executer(contexte=_contexte(llm), documents=["A"])

    assert resultat.succes

    ordre_observe: list[int] = []
    for _, utilisateur in llm.appels[: resultat.donnees["nombre_lots"]]:  # appels "map" seulement
        for citation in re.findall(r"\[S(\d+)\]", utilisateur):
            indice = int(citation) - 1
            if indice not in ordre_observe:
                ordre_observe.append(indice)

    assert ordre_observe == sorted(ordre_observe)


# ===========================================================================
# Test 5 — document inconnu
# ===========================================================================


def test_document_inconnu_echoue_proprement(monkeypatch):
    monkeypatch.setattr(summarize, "get_profil", lambda: None)
    monkeypatch.setattr(
        summarize, "catalogue", lambda profil=None: _FauxCatalogue(DocumentInconnu("Z introuvable"))
    )

    llm = LLMScripte(_cite_tout)
    outil = summarize.definir_summarize()
    resultat = outil.executer(contexte=_contexte(llm), documents=["Z"])

    assert not resultat.succes
    assert llm.appels == []


def test_document_ambigu_echoue_proprement(monkeypatch):
    monkeypatch.setattr(summarize, "get_profil", lambda: None)
    perimetre_ambigu = PerimetreDocumentaire(statut="ambigu", raison="marge_insuffisante")
    monkeypatch.setattr(summarize, "catalogue", lambda profil=None: _FauxCatalogue(perimetre_ambigu))

    llm = LLMScripte(_cite_tout)
    outil = summarize.definir_summarize()
    resultat = outil.executer(contexte=_contexte(llm), documents=["rapport"])

    assert not resultat.succes
    assert llm.appels == []


def test_collection_indisponible_echoue_proprement(monkeypatch):
    monkeypatch.setattr(summarize, "get_profil", lambda: None)
    monkeypatch.setattr(summarize, "catalogue", lambda profil=None: _FauxCatalogue(_perimetre_exact("A")))
    monkeypatch.setattr(
        summarize,
        "charger_document",
        lambda cible: (_ for _ in ()).throw(CollectionIndisponible("absente")),
    )

    llm = LLMScripte(_cite_tout)
    outil = summarize.definir_summarize()
    resultat = outil.executer(contexte=_contexte(llm), documents=["A"])

    assert not resultat.succes
    assert llm.appels == []


# ===========================================================================
# Test 6 — document vide : échec sans appel LLM inutile
# ===========================================================================


def test_document_vide_echoue_sans_appel_llm(monkeypatch):
    _resoudre_vers(monkeypatch, "A", [])

    llm = LLMScripte(_cite_tout)
    outil = summarize.definir_summarize()
    resultat = outil.executer(contexte=_contexte(llm), documents=["A"])

    assert not resultat.succes
    assert llm.appels == []


# ===========================================================================
# Test 7 — citations valides == sources originales
# ===========================================================================


def test_citations_valides_correspondent_aux_sources_originales(monkeypatch):
    passages = [_passage("A", i, f"contenu {i}") for i in range(3)]
    _resoudre_vers(monkeypatch, "A", passages)

    llm = LLMScripte(_cite_tout)
    outil = summarize.definir_summarize()
    resultat = outil.executer(contexte=_contexte(llm), documents=["A"])

    assert resultat.succes
    assert set(resultat.donnees["citations_valides"]) == {p.citation for p in passages}
    doc_ids_sources = {source.doc_id for source in resultat.sources}
    assert doc_ids_sources == {"A"}
    assert len(resultat.sources) == len(passages)


# ===========================================================================
# Test 8 — citation inventée rejetée
# ===========================================================================


def test_citation_inventee_rejetee(monkeypatch):
    passages = [_passage("A", 0, "contenu 0")]
    _resoudre_vers(monkeypatch, "A", passages)

    def _invente_une_citation(systeme, utilisateur, n):
        return "Résumé. [S1] et aussi [S999] qui n'existe pas."

    llm = LLMScripte(_invente_une_citation)
    outil = summarize.definir_summarize()
    resultat = outil.executer(contexte=_contexte(llm), documents=["A"])

    assert resultat.succes  # S1 reste valide, le résumé n'est pas rejeté en bloc
    assert resultat.donnees["citations_valides"] == ["S1"]
    assert any("S999" in a for a in resultat.avertissements)


# ===========================================================================
# Test 9 — zéro citation valide : jamais un succès silencieux (Cas 0)
# ===========================================================================


def test_zero_citation_valide_nest_pas_un_succes_document_complet(monkeypatch):
    passages = [_passage("A", 0, "contenu 0")]
    _resoudre_vers(monkeypatch, "A", passages)

    llm = LLMScripte(lambda systeme, utilisateur, n: "Résumé sans aucune citation.")
    outil = summarize.definir_summarize()
    resultat = outil.executer(contexte=_contexte(llm), documents=["A"])

    assert not resultat.succes
    assert "citation" in resultat.message.lower()


def test_zero_citation_valide_nest_pas_un_succes_contexte_existant():
    sources = [
        SourceOutil(doc_id="A", source="doc.pdf", nom_fichier="doc.pdf", page=1, extrait="contenu")
    ]
    llm = LLMScripte(lambda systeme, utilisateur, n: "Résumé sans aucune citation.")
    outil = summarize.definir_summarize()
    resultat = outil.executer(contexte=_contexte(llm, sources))

    assert not resultat.succes
    assert "citation" in resultat.message.lower()


# ===========================================================================
# Test 10 — compatibilité du mode contexte existant (Cas B, inchangé)
# ===========================================================================


def test_compatibilite_mode_contexte_existant():
    sources = [
        SourceOutil(doc_id="A", source="doc.pdf", nom_fichier="doc.pdf", page=1, extrait="contenu utile")
    ]
    llm = LLMScripte(_cite_tout)
    outil = summarize.definir_summarize()
    resultat = outil.executer(contexte=_contexte(llm, sources))

    assert resultat.succes
    assert resultat.donnees["citations_valides"] == ["S1"]
    assert len(resultat.sources) == 1


# ===========================================================================
# Échec d'un appel LLM intermédiaire : jamais une exception brute
# ===========================================================================


def test_echec_llm_pendant_un_lot_ne_leve_pas_dexception(monkeypatch):
    passages = [_passage("A", i, f"contenu {i}") for i in range(2)]
    _resoudre_vers(monkeypatch, "A", passages)

    outil = summarize.definir_summarize()
    resultat = outil.executer(contexte=_contexte(_LLMExplose()), documents=["A"])

    assert not resultat.succes
    assert "Ollama injoignable" in resultat.message or resultat.message


# ===========================================================================
# Correctif REVIEW NEEDED : unicité des citations sur plusieurs documents
#
# `charger_document` (Action 02) renumérote S1..Sn par document — donc deux
# documents chargés ensemble produisent chacun un S1, un S2, etc. Ces tests
# vérifient que `summarize` renumérote localement plutôt que de réutiliser
# `passage.citation` tel quel (ce qui écrasait silencieusement les sources
# du premier document dans `sources_par_citation`).
# ===========================================================================


def test_deux_documents_citations_uniques(monkeypatch):
    passages_a = [_passage("A", i, f"A contenu {i}") for i in range(3)]
    passages_b = [_passage("B", i, f"B contenu {i}") for i in range(3)]
    _resoudre_vers_plusieurs(monkeypatch, {"A": passages_a, "B": passages_b})

    llm = LLMScripte(_cite_tout)
    outil = summarize.definir_summarize()
    resultat = outil.executer(contexte=_contexte(llm), documents=["A", "B"])

    assert resultat.succes
    citations_vues: set[str] = set()
    for _, utilisateur in llm.appels:
        citations_vues |= _citations_du_prompt(utilisateur)
    assert citations_vues == {"[S1]", "[S2]", "[S3]", "[S4]", "[S5]", "[S6]"}
    assert set(resultat.donnees["citations_valides"]) == {"S1", "S2", "S3", "S4", "S5", "S6"}


def test_deux_documents_aucune_source_ecrasee(monkeypatch):
    passages_a = [_passage("A", i, f"A contenu {i}") for i in range(3)]
    passages_b = [_passage("B", i, f"B contenu {i}") for i in range(4)]
    _resoudre_vers_plusieurs(monkeypatch, {"A": passages_a, "B": passages_b})

    # Reconstruit directement sources_par_citation comme le fait le tool,
    # pour vérifier son invariant central (aucune collision) indépendamment
    # du texte produit par le LLM.
    doc_ids = summarize._resoudre_documents(["A", "B"])
    passages = summarize._charger_passages_documents(doc_ids)
    sources_par_citation = {
        f"S{index}": summarize._source_depuis_passage(p)
        for index, p in enumerate(passages, start=1)
    }

    assert len(passages) == 7
    assert len(sources_par_citation) == len(passages)


def test_deux_documents_provenance_intacte(monkeypatch):
    passages_a = [_passage("A", i, f"A contenu {i}", page=1) for i in range(3)]
    passages_b = [_passage("B", i, f"B contenu {i}", page=9) for i in range(3)]
    _resoudre_vers_plusieurs(monkeypatch, {"A": passages_a, "B": passages_b})

    doc_ids = summarize._resoudre_documents(["A", "B"])
    passages = summarize._charger_passages_documents(doc_ids)
    sources_par_citation = {
        f"S{index}": summarize._source_depuis_passage(p)
        for index, p in enumerate(passages, start=1)
    }

    # S1, S2, S3 -> document A, chunk_index 0, 1, 2, page 1
    for i, citation in enumerate(["S1", "S2", "S3"]):
        source = sources_par_citation[citation]
        assert source.doc_id == "A"
        assert source.page == 1
        assert source.extrait == f"A contenu {i}"

    # S4, S5, S6 -> document B, chunk_index 0, 1, 2, page 9
    for i, citation in enumerate(["S4", "S5", "S6"]):
        source = sources_par_citation[citation]
        assert source.doc_id == "B"
        assert source.page == 9
        assert source.extrait == f"B contenu {i}"


def test_deux_documents_ordre_deterministe(monkeypatch):
    passages_a = [_passage("A", i, f"A contenu {i}") for i in range(3)]
    passages_b = [_passage("B", i, f"B contenu {i}") for i in range(3)]
    _resoudre_vers_plusieurs(monkeypatch, {"A": passages_a, "B": passages_b})

    doc_ids = summarize._resoudre_documents(["A", "B"])
    passages = summarize._charger_passages_documents(doc_ids)
    citations = [f"S{index}" for index in range(1, len(passages) + 1)]

    assert citations == ["S1", "S2", "S3", "S4", "S5", "S6"]
    assert [p.doc_id for p in passages] == ["A", "A", "A", "B", "B", "B"]
    assert [p.chunk_index for p in passages] == [0, 1, 2, 0, 1, 2]


def test_resume_multi_document_pipeline_complet(monkeypatch):
    passages_a = [_passage("A", i, f"A contenu {i}") for i in range(3)]
    passages_b = [_passage("B", i, f"B contenu {i}") for i in range(3)]
    _resoudre_vers_plusieurs(monkeypatch, {"A": passages_a, "B": passages_b})

    llm = LLMScripte(_cite_tout)
    outil = summarize.definir_summarize()
    resultat = outil.executer(contexte=_contexte(llm), documents=["A", "B"])

    assert resultat.succes
    assert resultat.donnees["nombre_sources_disponibles"] == 6

    # Tous les passages ont bien participé au map (aucune collision de citation).
    citations_vues: set[str] = set()
    for _, utilisateur in llm.appels:
        citations_vues |= _citations_du_prompt(utilisateur)
    assert citations_vues == {f"[S{i}]" for i in range(1, 7)}

    assert set(resultat.donnees["citations_valides"]) == {f"S{i}" for i in range(1, 7)}

    doc_ids_sources = {source.doc_id for source in resultat.sources}
    assert doc_ids_sources == {"A", "B"}
    assert len(resultat.sources) == 6
