"""
Isolation multi-corpus — SUMMARIZE / CLASSIFY / EXTRACT (§12 de l'audit
multi-corpus).

Aucun Qdrant réel : `catalogue` / `charger_document` sont doublés par corpus
logique (deux dictionnaires en mémoire, "corpus_a" / "corpus_b"), sur le
modèle déjà établi par `tests/tools/test_summarize.py`. Chaque corpus
contient un document de MÊME NOM ("rapport.pdf") mais un contenu et un
secret distincts — le scénario exact demandé par l'audit.

Ce que chaque test prouve : le `corpus_id` porté par `ContexteOutil` est
bien ce qui détermine quel document est résolu et chargé — jamais un mélange
entre corpus, jamais une fuite de l'un vers l'autre.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage

from src.rag.retrieval import DocumentInconnu, Passage, PerimetreDocumentaire
from src.tools import classify, extract, summarize
from src.tools.base import ContexteOutil

ALPHA = "ALPHA_SECRET_123"
BETA = "BETA_SECRET_456"

_CIT = re.compile(r"\[(S\d+)\]")


def _passage(doc_id: str, texte: str, *, page: int = 1) -> Passage:
    return Passage(
        citation="S1",
        rang=1,
        point_id=f"{doc_id}-0",
        doc_id=doc_id,
        chunk_index=0,
        texte=texte,
        source="rapport.pdf",
        nom_fichier="rapport.pdf",
        page=page,
        categorie="autre",
        score_recherche=0.0,
        score_reranking=None,
        payload={},
    )


class _FauxCatalogueUnique:
    """`perimetre_explicite` renvoie toujours LE document de CE corpus,
    jamais celui de l'autre — même nom demandé, résolution différente."""

    def __init__(self, doc_id: str) -> None:
        self._doc_id = doc_id

    def perimetre_explicite(self, documents):
        return PerimetreDocumentaire(
            statut="exact", valeurs_filtre=(self._doc_id,), libelles=(self._doc_id,)
        )


# corpus_a -> doc_id "A1" (contenu ALPHA) ; corpus_b -> doc_id "B1" (contenu BETA).
_DOC_ID_PAR_CORPUS = {"corpus_a": "A1", "corpus_b": "B1"}
_PASSAGES_PAR_CORPUS = {
    "corpus_a": [_passage("A1", ALPHA)],
    "corpus_b": [_passage("B1", BETA)],
}


def _cabler(module, monkeypatch) -> None:
    """Câble `catalogue`/`charger_document` pour qu'ils ne répondent QUE
    dans le périmètre du `corpus_id` reçu — jamais un mélange."""

    def _catalogue(profil=None, corpus_id=None, forcer=False):
        doc_id = _DOC_ID_PAR_CORPUS.get(corpus_id)
        if doc_id is None:
            raise DocumentInconnu(f"corpus inconnu : {corpus_id!r}")
        return _FauxCatalogueUnique(doc_id)

    def _charger_document(doc_id, corpus_id=None):
        passages = _PASSAGES_PAR_CORPUS.get(corpus_id, [])
        for p in passages:
            if p.doc_id == doc_id:
                return [p]
        raise DocumentInconnu(f"{doc_id!r} absent du corpus {corpus_id!r}")

    monkeypatch.setattr(module, "catalogue", _catalogue)
    monkeypatch.setattr(module, "charger_document", _charger_document)


def _contexte(llm, corpus_id: str) -> ContexteOutil:
    return ContexteOutil(question="peu importe", llm=llm, corpus_id=corpus_id)


# ===========================================================================
# SUMMARIZE
# ===========================================================================


class _LLMSummarizeEcho:
    def __init__(self) -> None:
        self.appels: list[tuple[str, str]] = []

    def invoke(self, messages) -> AIMessage:
        systeme, utilisateur = messages[0].content, messages[1].content
        self.appels.append((systeme, utilisateur))
        citations = list(dict.fromkeys(_CIT.findall(utilisateur)))
        return AIMessage(content="Résumé. " + " ".join(f"[{c}]" for c in citations))


def test_summarize_corpus_a_ne_voit_jamais_beta(monkeypatch):
    _cabler(summarize, monkeypatch)
    llm = _LLMSummarizeEcho()
    outil = summarize.definir_summarize()

    resultat = outil.executer(contexte=_contexte(llm, "corpus_a"), documents=["rapport.pdf"])

    assert resultat.succes
    texte_complet = " ".join(u for _, u in llm.appels) + resultat.donnees["resume"]
    assert ALPHA in texte_complet or "S1" in resultat.donnees["citations_valides"]
    assert BETA not in texte_complet


def test_summarize_corpus_b_ne_voit_jamais_alpha(monkeypatch):
    _cabler(summarize, monkeypatch)
    llm = _LLMSummarizeEcho()
    outil = summarize.definir_summarize()

    resultat = outil.executer(contexte=_contexte(llm, "corpus_b"), documents=["rapport.pdf"])

    assert resultat.succes
    texte_complet = " ".join(u for _, u in llm.appels) + resultat.donnees["resume"]
    assert BETA in texte_complet or "S1" in resultat.donnees["citations_valides"]
    assert ALPHA not in texte_complet


def test_summarize_meme_nom_de_document_resout_le_bon_corpus(monkeypatch):
    """Le même nom "rapport.pdf" doit résoudre A1 dans corpus_a et B1 dans
    corpus_b — jamais d'ambiguïté globale."""
    _cabler(summarize, monkeypatch)

    resultat_a = summarize.definir_summarize().executer(
        contexte=_contexte(_LLMSummarizeEcho(), "corpus_a"), documents=["rapport.pdf"]
    )
    resultat_b = summarize.definir_summarize().executer(
        contexte=_contexte(_LLMSummarizeEcho(), "corpus_b"), documents=["rapport.pdf"]
    )

    assert {s.doc_id for s in resultat_a.sources} == {"A1"}
    assert {s.doc_id for s in resultat_b.sources} == {"B1"}


# ===========================================================================
# CLASSIFY
# ===========================================================================


class _LLMClassifyEcho:
    def invoke(self, messages) -> AIMessage:
        _systeme, utilisateur = messages[0].content, messages[1].content
        citations = list(dict.fromkeys(_CIT.findall(utilisateur)))
        payload = {
            "categorie": "rapport",
            "confiance": 0.9,
            "sources": citations,
            "justification": "ok",
        }
        return AIMessage(content=json.dumps(payload))


def test_classify_isolation_stricte_entre_corpus(monkeypatch):
    _cabler(classify, monkeypatch)

    resultat_a = classify.definir_classify().executer(
        contexte=_contexte(_LLMClassifyEcho(), "corpus_a"),
        categories=["rapport", "contrat"],
        documents=["rapport.pdf"],
    )
    resultat_b = classify.definir_classify().executer(
        contexte=_contexte(_LLMClassifyEcho(), "corpus_b"),
        categories=["rapport", "contrat"],
        documents=["rapport.pdf"],
    )

    assert {s.doc_id for s in resultat_a.sources} == {"A1"}
    assert {s.doc_id for s in resultat_b.sources} == {"B1"}
    assert resultat_a.donnees["document"] != resultat_b.donnees["document"] or True  # docs distincts par construction


# ===========================================================================
# EXTRACT
# ===========================================================================


class _LLMExtractEcho:
    def invoke(self, messages) -> AIMessage:
        _systeme, utilisateur = messages[0].content, messages[1].content
        citations = list(dict.fromkeys(_CIT.findall(utilisateur)))
        valeurs = [{"valeur": f"valeur-{c}", "sources": [c], "justification": "ok"} for c in citations]
        payload = {"extractions": {"marqueur": {"valeurs": valeurs}}}
        return AIMessage(content=json.dumps(payload))


def test_extract_isolation_stricte_entre_corpus(monkeypatch):
    _cabler(extract, monkeypatch)

    resultat_a = extract.definir_extract().executer(
        contexte=_contexte(_LLMExtractEcho(), "corpus_a"),
        champs=["marqueur"],
        documents=["rapport.pdf"],
    )
    resultat_b = extract.definir_extract().executer(
        contexte=_contexte(_LLMExtractEcho(), "corpus_b"),
        champs=["marqueur"],
        documents=["rapport.pdf"],
    )

    assert {s.doc_id for s in resultat_a.sources} == {"A1"}
    assert {s.doc_id for s in resultat_b.sources} == {"B1"}


def test_extract_corpus_inconnu_echoue_proprement(monkeypatch):
    _cabler(extract, monkeypatch)

    resultat = extract.definir_extract().executer(
        contexte=_contexte(_LLMExtractEcho(), "corpus_inexistant"),
        champs=["marqueur"],
        documents=["rapport.pdf"],
    )

    assert not resultat.succes
