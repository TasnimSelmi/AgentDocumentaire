"""
Tests d'intégration du graphe agentique (rechercher/évaluer/reformuler/répondre).

`src.rag.generation.generer_depuis_recherche` est systématiquement remplacé
par une doublure : ces tests vérifient le câblage et le routage du graphe,
pas la génération RAG elle-même (déjà testée ailleurs). Aucun serveur Ollama
ni collection Qdrant n'est nécessaire.

Les scores factices encadrent volontairement `nodes.SEUIL_PERTINENCE_MINIMALE`
(0.15) — voir le commentaire de calibrage dans `src/agent/nodes.py`.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from src.agent import nodes
from src.agent.graph import construire_graphe
from src.agent.graph_state import EtatGraphe
from src.agent.session import construire_session
from src.rag.generation import ReponseRAG
from src.rag.retrieval import Passage, RapportRecherche
from src.tools.base import DefinitionOutil, ResultatOutil, SourceOutil

_SCORE_EVIDENCE_FORTE = 0.90
_SCORE_EVIDENCE_FAIBLE = 0.03


class _ArgsSearchFactice(BaseModel):
    requete: str = Field(default="", description="Requête factice.")


def _reponse_factice(question: str) -> ReponseRAG:
    return ReponseRAG(
        question=question,
        reponse="Réponse factice.",
        profil="generic",
        contexte_suffisant=True,
        citations_valides=True,
        citations_reparees=False,
        sources=[],
    )


def _fabrique_generation_factice(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    appels: list[dict[str, Any]] = []

    def _fausse_generation(**kwargs: Any) -> ReponseRAG:
        appels.append(kwargs)
        return _reponse_factice(kwargs["question"])

    monkeypatch.setattr(nodes, "generer_depuis_recherche", _fausse_generation)
    return appels


def _un_passage(score: float) -> Passage:
    return Passage(
        citation="S1",
        rang=1,
        point_id="p1",
        doc_id="d1",
        chunk_index=0,
        texte="Extrait.",
        source="doc.pdf",
        nom_fichier="doc.pdf",
        page=1,
        categorie="",
        score_recherche=score,
        score_reranking=score,
    )


def _un_rapport(score: float | None) -> RapportRecherche:
    return RapportRecherche(
        requete="peu importe",
        profil="generic",
        filtres={},
        passages=[] if score is None else [_un_passage(score)],
        candidats_recuperes=0 if score is None else 1,
        reranking_utilise=True,
        seuil_applique=None,
        duree_secondes=0.01,
    )


def _outil_search_sequence(
    scores: Sequence[float | None],
) -> tuple[Callable[[], DefinitionOutil], list[int]]:
    """
    Fabrique de recherche factice à scores contrôlés (un par appel, le
    dernier répété au-delà). Renvoie aussi un compteur d'appels partagé,
    pour vérifier explicitement l'absence de double retrieval.
    """
    compteur: list[int] = [0]

    def _fonction(*, contexte=None, **kw) -> ResultatOutil:
        index = min(compteur[0], len(scores) - 1)
        compteur[0] += 1
        score = scores[index]

        rapport = _un_rapport(score)
        if contexte is not None:
            contexte.dernier_rapport_recherche = rapport

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

    def _fabrique() -> DefinitionOutil:
        return DefinitionOutil(
            nom="search",
            description="Search factice à scores contrôlés.",
            schema_arguments=_ArgsSearchFactice,
            fonction=_fonction,
        )

    return _fabrique, compteur


def _outil_search(score: float | None):
    fabrique, _ = _outil_search_sequence([score])
    return fabrique()


class _LLMNonSollicite:
    def invoke(self, messages: Any):  # pragma: no cover - non sollicité ici
        raise AssertionError("Aucun LLM ne devrait être invoqué dans ce test.")


class _LLMScripte:
    """Distingue reformulation et jugement de suffisance par le message système."""

    def __init__(
        self,
        *,
        reformulations: Sequence[str] = (),
        verdicts_suffisance: Sequence[str] = (),
    ) -> None:
        self._reformulations = list(reformulations)
        self._verdicts = list(verdicts_suffisance)
        self.appels_reformulation = 0
        self.appels_suffisance = 0

    def invoke(self, messages: Any) -> AIMessage:
        systeme = messages[0].content

        if "reformules une requête" in systeme:
            self.appels_reformulation += 1
            if self._reformulations:
                return AIMessage(content=self._reformulations.pop(0))
            return AIMessage(
                content=f'{{"requete_reformulee": "reformulation {self.appels_reformulation}"}}'
            )

        if "juges si des passages" in systeme:
            self.appels_suffisance += 1
            if self._verdicts:
                return AIMessage(content=self._verdicts.pop(0))
            return AIMessage(content='{"suffisant": false, "raison": "défaut de test"}')

        raise AssertionError(f"Message système inattendu : {systeme[:80]!r}")


def _invoquer(session) -> ReponseRAG:
    graphe = construire_graphe()
    limite = max(25, session.etat.max_tentatives * 3 + 5)
    resultat = graphe.invoke(EtatGraphe(session=session), config={"recursion_limit": limite})
    return resultat["reponse"]


# ---------------------------------------------------------------------------


def test_evidence_forte_et_suffisante_du_premier_coup_ne_reformule_pas(monkeypatch):
    appels_generation = _fabrique_generation_factice(monkeypatch)
    fabrique, compteur_recherche = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    llm = _LLMScripte(verdicts_suffisance=['{"suffisant": true, "raison": "ok"}'])

    session = construire_session(
        "Quelles sont les conditions de résiliation ?",
        llm=llm,
        charger_profil_domaine=False,
        fabriques=[fabrique],
    )
    reponse = _invoquer(session)

    assert isinstance(reponse, ReponseRAG)
    assert len(appels_generation) == 1
    assert compteur_recherche[0] == 1
    assert llm.appels_reformulation == 0
    assert llm.appels_suffisance == 1
    assert session.etat.tentatives == 1


def test_reformulation_ameliore_la_suffisance_et_permet_de_repondre(monkeypatch):
    """
    Cas positif explicite : preuves pertinentes mais jugées insuffisantes au
    premier essai, suffisantes après une reformulation. Le budget de
    recherche n'est pas épuisé, la génération est bien appelée avec le
    second rapport.
    """
    appels_generation = _fabrique_generation_factice(monkeypatch)
    fabrique, compteur_recherche = _outil_search_sequence(
        [_SCORE_EVIDENCE_FORTE, _SCORE_EVIDENCE_FORTE]
    )
    llm = _LLMScripte(
        reformulations=['{"requete_reformulee": "reformulation plus précise"}'],
        verdicts_suffisance=[
            '{"suffisant": false, "raison": "sujet correct mais valeur absente"}',
            '{"suffisant": true, "raison": "valeur trouvée", "elements_support": ["S1"]}',
        ],
    )

    session = construire_session(
        "Quel est le niveau exact de la métrique X pour l'année Y ?",
        llm=llm,
        charger_profil_domaine=False,
        fabriques=[fabrique],
        max_tentatives=6,
    )
    reponse = _invoquer(session)

    assert isinstance(reponse, ReponseRAG)
    assert compteur_recherche[0] == 2
    assert llm.appels_reformulation == 1
    assert llm.appels_suffisance == 2
    assert len(appels_generation) == 1
    assert session.etat.a_ete_reformulee


def test_evidence_non_pertinente_stagnante_stoppe_avant_le_budget(monkeypatch):
    """
    Score de pertinence constant d'une tentative à l'autre (le cas mesuré en
    smoke test réel) : le garde-fou anti-stagnation doit interrompre la
    boucle avant `max_tentatives`, sans jamais appeler le LLM de génération.
    """
    appels_generation = _fabrique_generation_factice(monkeypatch)
    fabrique, compteur_recherche = _outil_search_sequence([_SCORE_EVIDENCE_FAIBLE])
    llm = _LLMScripte()

    session = construire_session(
        "Question dont l'évidence est toujours faible",
        llm=llm,
        charger_profil_domaine=False,
        fabriques=[fabrique],
        max_tentatives=6,
    )
    reponse = _invoquer(session)

    assert isinstance(reponse, ReponseRAG)
    assert reponse.contexte_suffisant is False
    assert len(appels_generation) == 0
    assert llm.appels_suffisance == 0  # jamais pertinent, niveau 2 jamais atteint
    # Stagnation détectée dès la 2e évaluation : une seule reformulation.
    assert llm.appels_reformulation == 1
    assert compteur_recherche[0] == 2
    assert session.etat.tentatives < session.etat.max_tentatives


def test_evidence_non_pertinente_variable_epuise_le_budget(monkeypatch):
    """
    Sans stagnation (le score varie mais reste sous le seuil), la boucle va
    bien jusqu'au bout du budget avant de refuser — la garde stagnation ne
    doit pas interrompre un cas qui progresse simplement trop lentement.
    """
    appels_generation = _fabrique_generation_factice(monkeypatch)
    fabrique, compteur_recherche = _outil_search_sequence([0.02, 0.05, 0.03])
    llm = _LLMScripte()

    session = construire_session(
        "Question dont l'évidence reste faible mais varie",
        llm=llm,
        charger_profil_domaine=False,
        fabriques=[fabrique],
        max_tentatives=3,
    )
    reponse = _invoquer(session)

    assert isinstance(reponse, ReponseRAG)
    assert reponse.contexte_suffisant is False
    assert len(appels_generation) == 0
    assert compteur_recherche[0] == 3
    assert llm.appels_reformulation == 2
    assert session.etat.tentatives == 3
    assert not session.etat.peut_reessayer


def test_resultat_vide_route_vers_reformuler_puis_refuse(monkeypatch):
    appels_generation = _fabrique_generation_factice(monkeypatch)
    fabrique, compteur_recherche = _outil_search_sequence([None])
    llm = _LLMScripte()

    session = construire_session(
        "Question introuvable dans le corpus",
        llm=llm,
        charger_profil_domaine=False,
        fabriques=[fabrique],
        max_tentatives=6,
    )
    reponse = _invoquer(session)

    assert isinstance(reponse, ReponseRAG)
    assert reponse.contexte_suffisant is False
    assert len(appels_generation) == 0
    # Un RapportRecherche (vide) a bien été construit à chaque tentative :
    # le refus final passe par `refuser_sans_generation`, pas par le repli
    # « aucun rapport du tout » (réservé au cas où `search` échoue
    # techniquement, jamais exercé ici).
    assert "Corpus indisponible" not in reponse.reponse
    # Score constant à 0.0 (aucun passage) : stagnation dès la 2e évaluation.
    assert llm.appels_reformulation == 1
    assert compteur_recherche[0] == 2


def test_double_retrieval_ne_se_reproduit_plus(monkeypatch):
    """
    Test dédié au correctif principal de cette tâche : le nombre d'appels à
    l'outil `search` doit être strictement égal au nombre de tentatives
    consommées, jamais un de plus pour la génération finale.
    """
    appels_generation = _fabrique_generation_factice(monkeypatch)
    fabrique, compteur_recherche = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    llm = _LLMScripte(verdicts_suffisance=['{"suffisant": true, "raison": "ok"}'])

    session = construire_session(
        "Question answerable dès le premier essai",
        llm=llm,
        charger_profil_domaine=False,
        fabriques=[fabrique],
    )
    _invoquer(session)

    assert compteur_recherche[0] == session.etat.tentatives == 1
    assert len(appels_generation) == 1
