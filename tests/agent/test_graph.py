"""
Tests d'intégration du graphe agentique (rechercher/évaluer/reformuler/répondre).

`src.rag.generation.generer_reponse` est systématiquement remplacé par une
doublure : ces tests vérifient le câblage et le routage du graphe, pas la
génération RAG elle-même (déjà testée ailleurs). Aucun serveur Ollama ni
collection Qdrant n'est nécessaire.

Les scores factices encadrent volontairement `nodes.SEUIL_PERTINENCE_MINIMALE`
(0.08) — voir le commentaire de calibrage dans `src/agent/nodes.py`.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from src.agent import nodes
from src.agent.graph import invoquer_agent
from src.tools.base import DefinitionOutil, ResultatOutil, SourceOutil

_SCORE_EVIDENCE_FORTE = 0.90
_SCORE_EVIDENCE_FAIBLE = 0.03


class _ArgsSearchFactice(BaseModel):
    requete: str = Field(default="", description="Requête factice.")


class _ReponseFactice:
    def __init__(self, question: str) -> None:
        self.question = question
        self.contexte_suffisant = True
        self.sources: list[Any] = []


def _fabrique_reponse_factice(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    appels: list[dict[str, Any]] = []

    def _fausse_generer_reponse(**kwargs: Any):
        appels.append(kwargs)
        return _ReponseFactice(kwargs["question"])

    monkeypatch.setattr(nodes, "generer_reponse", _fausse_generer_reponse)
    return appels


def _outil_search(score: float | None) -> DefinitionOutil:
    """
    `score=None` simule un résultat vide ; toute autre valeur simule un
    unique passage portant ce score de pertinence (reranking factice).
    """

    def _fonction(*, contexte=None, **kw):
        if score is None:
            return ResultatOutil(outil="search", succes=True, message="Rien trouvé.")
        return ResultatOutil(
            outil="search",
            succes=True,
            message="1 passage trouvé.",
            sources=[
                SourceOutil(
                    doc_id="d1",
                    source="doc.pdf",
                    nom_fichier="doc.pdf",
                    page=1,
                    score=score,
                    extrait="Extrait.",
                )
            ],
        )

    return DefinitionOutil(
        nom="search",
        description="Search factice à score contrôlé.",
        schema_arguments=_ArgsSearchFactice,
        fonction=_fonction,
    )


class _LLMNonSollicite:
    def invoke(self, messages: Any):  # pragma: no cover - non sollicité ici
        raise AssertionError("Aucun LLM ne devrait être invoqué dans ce test.")


class _LLMReformulationEnBoucle:
    """Reformule à chaque appel, mais ne trouvera jamais rien de pertinent :
    le budget de tentatives doit rester la seule garantie de terminaison."""

    def __init__(self) -> None:
        self.appels = 0

    def invoke(self, messages: Any) -> AIMessage:
        self.appels += 1
        return AIMessage(
            content=f'{{"requete_reformulee": "reformulation numéro {self.appels}"}}'
        )


def test_evidence_forte_du_premier_coup_ne_reformule_pas(monkeypatch):
    appels = _fabrique_reponse_factice(monkeypatch)

    reponse = invoquer_agent(
        "Quelles sont les conditions de résiliation ?",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[lambda: _outil_search(_SCORE_EVIDENCE_FORTE)],
    )

    assert isinstance(reponse, _ReponseFactice)
    assert len(appels) == 1


def test_evidence_faible_route_vers_reformuler_puis_stoppe_au_budget(monkeypatch):
    """
    Un passage est bien retrouvé à chaque tentative (`nombre_preuves == 1`),
    mais son score de pertinence reste sous le seuil : le graphe doit quand
    même router vers `reformuler`, pas conclure à tort que l'evidence est
    suffisante — exactement le défaut corrigé dans `noeud_evaluer_preuves`.
    """
    appels = _fabrique_reponse_factice(monkeypatch)
    llm = _LLMReformulationEnBoucle()

    reponse = invoquer_agent(
        "Question dont l'évidence est toujours faible",
        llm=llm,
        charger_profil_domaine=False,
        fabriques=[lambda: _outil_search(_SCORE_EVIDENCE_FAIBLE)],
        max_tentatives=3,
    )

    assert isinstance(reponse, _ReponseFactice)
    # 3 tentatives de recherche, donc 2 reformulations entre elles, puis
    # arrêt propre sur generer_reponse une fois le budget épuisé.
    assert llm.appels == 2
    assert len(appels) == 1


def test_resultat_vide_route_aussi_vers_reformuler_puis_stoppe_au_budget(monkeypatch):
    appels = _fabrique_reponse_factice(monkeypatch)
    llm = _LLMReformulationEnBoucle()

    reponse = invoquer_agent(
        "Question introuvable dans le corpus",
        llm=llm,
        charger_profil_domaine=False,
        fabriques=[lambda: _outil_search(None)],
        max_tentatives=3,
    )

    assert isinstance(reponse, _ReponseFactice)
    assert llm.appels == 2
    assert len(appels) == 1
