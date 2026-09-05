"""
Tests de stress STRUCTURELS pour documents très longs — audit long-documents
(sections A/B/C). Aucun Ollama, aucun Qdrant réel, aucun corpus existant
touché : documents synthétiques en mémoire, LLM déterministes.

Quatre gabarits, un "page" synthétique = un `Passage` de ~1800 caractères
(taille de chunk par défaut du projet, `config/default.yaml`) :
    court   ~5 pages
    moyen   ~50 pages
    long    ~300 pages
    extrême ~1000 pages

Pour chaque capacité (SUMMARIZE, CLASSIFY, EXTRACT), on mesure : nombre de
passages, nombre de lots, taille maximale de prompt réellement envoyée
(comparée au budget réel), nombre d'appels LLM, convergence, et l'absence
de perte du DERNIER passage (marqueur unique inséré à la toute fin du
document).

Pour COMPARE/SYNTHESIZE : `tests/agent/test_multidoc_budget_preuve.py`
couvre déjà en détail NB_LOTS_MAX et le budget REDUCE ; ce fichier ajoute
uniquement une vérification à l'échelle "moyen"/"long" que ces protections
restent intactes, sans dupliquer cette suite.
"""

from __future__ import annotations

import json
import re

import pytest
from langchain_core.messages import AIMessage

from src.agent import multidoc_pipeline
from src.agent.multidoc_pipeline import budget_caracteres_entree_llm
from src.rag.retrieval import DocumentInconnu, Passage, PerimetreDocumentaire
from src.tools import classify, extract, summarize
from src.tools.base import ContexteOutil
from src.tools.compare import comparer
from tests.agent._multidoc_fakes import LLMScripte as LLMMultidoc
from tests.agent._multidoc_fakes import cabler_corpus
from tests.agent._multidoc_fakes import passage as passage_multidoc

TAILLES = pytest.mark.parametrize(
    "nb_pages",
    [
        pytest.param(5, id="court"),
        pytest.param(50, id="moyen"),
        pytest.param(300, id="long"),
        pytest.param(1000, id="extreme"),
    ],
)

MARQUEUR = "MARQUEUR-DERNIERE-PAGE-4242"
_CIT = re.compile(r"\[(S\d+)\]")


def _pages_synthetiques(doc_id: str, nb_pages: int) -> list[Passage]:
    """`nb_pages` passages de ~1800 caractères ; la DERNIÈRE page porte un
    marqueur unique, pour vérifier qu'elle n'est jamais perdue."""
    pages = [
        Passage(
            citation=f"S{i}",
            rang=i,
            point_id=f"{doc_id}-{i}",
            doc_id=doc_id,
            chunk_index=i - 1,
            texte=(f"Paragraphe de remplissage {i}. " * 55).strip(),
            source=f"{doc_id}.pdf",
            nom_fichier=f"{doc_id}.pdf",
            page=i,
            categorie="",
            score_recherche=0.0,
            score_reranking=None,
        )
        for i in range(1, nb_pages)
    ]
    pages.append(
        Passage(
            citation=f"S{nb_pages}",
            rang=nb_pages,
            point_id=f"{doc_id}-{nb_pages}",
            doc_id=doc_id,
            chunk_index=nb_pages - 1,
            texte=f"{MARQUEUR}.",
            source=f"{doc_id}.pdf",
            nom_fichier=f"{doc_id}.pdf",
            page=nb_pages,
            categorie="",
            score_recherche=0.0,
            score_reranking=None,
        )
    )
    return pages


class _FauxCatalogueUnique:
    def __init__(self, doc_id: str) -> None:
        self._doc_id = doc_id

    def perimetre_explicite(self, documents):
        return PerimetreDocumentaire(
            statut="exact", valeurs_filtre=(self._doc_id,), libelles=(self._doc_id,)
        )


def _cabler(module, monkeypatch, doc_id: str, passages: list[Passage]) -> None:
    monkeypatch.setattr(module, "get_profil", lambda: None)
    monkeypatch.setattr(module, "catalogue", lambda profil=None: _FauxCatalogueUnique(doc_id))
    monkeypatch.setattr(
        module,
        "charger_document",
        lambda cible: passages if cible == doc_id else (_ for _ in ()).throw(DocumentInconnu(cible)),
    )


def _contexte(llm) -> ContexteOutil:
    return ContexteOutil(question="peu importe", llm=llm, sources=[])


# ===========================================================================
# SUMMARIZE — convergence hiérarchique, jamais de prompt hors budget
# ===========================================================================


class _LLMSummarizeEcho:
    def __init__(self) -> None:
        self.appels: list[tuple[str, str]] = []

    def invoke(self, messages) -> AIMessage:
        systeme, utilisateur = messages[0].content, messages[1].content
        self.appels.append((systeme, utilisateur))
        citations = list(dict.fromkeys(_CIT.findall(utilisateur)))
        marqueur = " MARQUEUR_VU" if MARQUEUR in utilisateur else ""
        return AIMessage(content="Synthèse. " + " ".join(f"[{c}]" for c in citations) + marqueur)


@TAILLES
def test_summarize_stress(monkeypatch, nb_pages: int) -> None:
    passages = _pages_synthetiques("DOC", nb_pages)
    _cabler(summarize, monkeypatch, "DOC", passages)

    llm = _LLMSummarizeEcho()
    outil = summarize.definir_summarize()
    resultat = outil.executer(contexte=_contexte(llm), documents=["DOC"])

    budget = budget_caracteres_entree_llm()

    assert resultat.succes, resultat.message
    assert resultat.donnees["nombre_sources_disponibles"] == nb_pages
    assert resultat.donnees["nombre_lots"] >= 1

    tailles_prompts = [len(s) + len(u) for s, u in llm.appels]
    assert tailles_prompts, "au moins un appel LLM"
    assert max(tailles_prompts) <= budget, (
        f"prompt hors budget : {max(tailles_prompts)} > {budget}"
    )

    # Aucune perte du dernier passage : sa citation a atteint la synthèse finale.
    assert f"S{nb_pages}" in resultat.donnees["citations_valides"]
    assert any(MARQUEUR in u for _, u in llm.appels)


# ===========================================================================
# CLASSIFY — tous les lots traités, agrégation déterministe (aucun LLM)
# ===========================================================================


class _LLMClassifyEcho:
    def __init__(self, categorie: str = "rapport") -> None:
        self._categorie = categorie
        self.appels: list[tuple[str, str]] = []

    def invoke(self, messages) -> AIMessage:
        systeme, utilisateur = messages[0].content, messages[1].content
        self.appels.append((systeme, utilisateur))
        citations = list(dict.fromkeys(_CIT.findall(utilisateur)))
        payload = {
            "categorie": self._categorie,
            "confiance": 0.9,
            "sources": citations,
            "justification": "ok",
        }
        return AIMessage(content=json.dumps(payload))


@TAILLES
def test_classify_stress(monkeypatch, nb_pages: int) -> None:
    passages = _pages_synthetiques("DOC", nb_pages)
    _cabler(classify, monkeypatch, "DOC", passages)

    llm = _LLMClassifyEcho()
    outil = classify.definir_classify()
    resultat = outil.executer(
        contexte=_contexte(llm),
        categories=["rapport", "contrat"],
        documents=["DOC"],
    )

    budget = budget_caracteres_entree_llm()

    assert resultat.succes, resultat.message
    assert resultat.donnees["categorie"] == "rapport"
    # TOUS les lots ont voté (unanimité) : aucun lot perdu, agrégation sur
    # l'ensemble du document.
    assert resultat.donnees["lots_valides"] == resultat.donnees["nombre_total_lots"]
    assert resultat.donnees["nombre_passages"] == nb_pages

    tailles_prompts = [len(s) + len(u) for s, u in llm.appels]
    assert max(tailles_prompts) <= budget

    # Le dernier passage a bien été vu par un lot (donc voté).
    assert any(f"S{nb_pages}" in u for _, u in llm.appels)
    assert f"S{nb_pages}" in resultat.donnees["citations"]


# ===========================================================================
# EXTRACT — tous les lots traités, agrégation déterministe, aucune perte
# ===========================================================================


class _LLMExtractEcho:
    def __init__(self) -> None:
        self.appels: list[tuple[str, str]] = []

    def invoke(self, messages) -> AIMessage:
        systeme, utilisateur = messages[0].content, messages[1].content
        self.appels.append((systeme, utilisateur))
        citations = list(dict.fromkeys(_CIT.findall(utilisateur)))
        valeurs = [
            {"valeur": f"valeur-{c}", "sources": [c], "justification": "ok"}
            for c in citations
        ]
        payload = {"extractions": {"marqueur": {"valeurs": valeurs}}}
        return AIMessage(content=json.dumps(payload))


@TAILLES
def test_extract_stress(monkeypatch, nb_pages: int) -> None:
    passages = _pages_synthetiques("DOC", nb_pages)
    _cabler(extract, monkeypatch, "DOC", passages)

    llm = _LLMExtractEcho()
    outil = extract.definir_extract()
    resultat = outil.executer(
        contexte=_contexte(llm),
        champs=["marqueur"],
        documents=["DOC"],
    )

    budget = budget_caracteres_entree_llm()

    assert resultat.succes, resultat.message
    assert resultat.donnees["lots_invalides"] == 0
    assert resultat.donnees["nombre_passages"] == nb_pages

    tailles_prompts = [len(s) + len(u) for s, u in llm.appels]
    assert max(tailles_prompts) <= budget

    # Aucune perte due au découpage : la valeur du dernier passage est présente.
    valeurs = {v["valeur"] for v in resultat.donnees["extractions"]["marqueur"]["valeurs"]}
    assert f"valeur-S{nb_pages}" in valeurs


# ===========================================================================
# COMPARE / SYNTHESIZE — vérifie que NB_LOTS_MAX / budget REDUCE tiennent
# toujours à l'échelle (déjà testés en détail par
# test_multidoc_budget_preuve.py ; ne pas dupliquer cette suite ici).
# ===========================================================================


@pytest.mark.parametrize("nb_pages", [50, 300])
def test_compare_stress_echelle_moyenne_et_longue(monkeypatch, nb_pages: int) -> None:
    a = [passage_multidoc("A", i, f"contenu A {i} " * 20, page=i) for i in range(1, nb_pages + 1)]
    b = [passage_multidoc("B", i, f"contenu B {i} " * 20, page=i) for i in range(1, min(nb_pages, 20) + 1)]
    cabler_corpus(
        monkeypatch,
        multidoc_pipeline,
        fiches={"a.pdf": "A", "b.pdf": "B"},
        passages_par_doc={"A": a, "B": b},
    )

    llm = LLMMultidoc()
    r = comparer("Compare a.pdf et b.pdf.", ["a.pdf", "b.pdf"], llm=llm)

    budget = budget_caracteres_entree_llm()
    # Soit un succès avec des prompts tous dans le budget (protection
    # intacte), soit un refus explicite au-delà de NB_LOTS_MAX — jamais un
    # envoi hors budget dans les deux cas.
    for systeme, utilisateur in llm.appels:
        assert len(systeme) + len(utilisateur) <= budget

    if r.succes:
        assert r.donnees["statut"] in {"complet", "partiel"}
    else:
        assert r.outil == "compare"
