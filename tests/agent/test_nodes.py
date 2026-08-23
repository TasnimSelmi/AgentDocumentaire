"""
Tests des nœuds du graphe agentique.

Comme pour `tests/agent/test_session.py`, aucun test ici n'exige de serveur
Ollama ni de collection Qdrant : le LLM est une doublure et l'outil
`search` est une fabrique factice enregistrée à la place de la vraie
(`src.tools.search`). `noeud_generer_reponse` est testé en isolant
`src.rag.generation.generer_reponse` (monkeypatch), puisque ce nœud délègue
entièrement à ce module et ne doit donc jamais l'appeler réellement ici.

Les scores factices ci-dessous encadrent volontairement
`nodes.SEUIL_PERTINENCE_MINIMALE` (0.08) : voir le commentaire de calibrage
dans `src/agent/nodes.py` pour les scores réels observés (reranker
BAAI/bge-reranker-v2-m3) qui justifient ce seuil.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from src.agent import nodes
from src.agent.graph_state import EtatGraphe
from src.agent.session import construire_session
from src.tools.base import DefinitionOutil, ResultatOutil, SourceOutil

# Au-dessus du seuil (0.08) : cas net observé en smoke test réel (0.12 à 0.90).
_SCORE_EVIDENCE_FORTE = 0.90
# Sous le seuil mais non nul : c'est exactement le cas qui échappait à
# l'ancienne heuristique (nombre_preuves >= 1) — un passage existe, mais son
# score de reranking le désigne comme non pertinent (0.03 à 0.06 observés en
# smoke test réel sur des requêtes hors-corpus).
_SCORE_EVIDENCE_FAIBLE = 0.03


class _ArgsSearchFactice(BaseModel):
    requete: str = Field(default="", description="Requête factice.")


def _une_source(score: float) -> SourceOutil:
    return SourceOutil(
        doc_id="d1",
        source="doc.pdf",
        nom_fichier="doc.pdf",
        page=1,
        score=score,
        extrait="Un extrait.",
    )


def _outil_search(*, score: float | None) -> DefinitionOutil:
    """
    Fabrique un outil `search` factice.

    `score=None` simule un résultat vide (aucun passage) ; toute autre
    valeur simule un passage unique portant ce score de pertinence.
    """

    def _fonction(*, contexte=None, **kw) -> ResultatOutil:
        if score is None:
            return ResultatOutil(
                outil="search",
                succes=True,
                message="Aucun passage pertinent trouvé.",
            )
        return ResultatOutil(
            outil="search",
            succes=True,
            message="1 passage trouvé.",
            sources=[_une_source(score)],
        )

    return DefinitionOutil(
        nom="search",
        description="Search factice à score contrôlé.",
        schema_arguments=_ArgsSearchFactice,
        fonction=_fonction,
    )


def _outil_search_evidence_forte() -> DefinitionOutil:
    return _outil_search(score=_SCORE_EVIDENCE_FORTE)


def _outil_search_evidence_faible() -> DefinitionOutil:
    return _outil_search(score=_SCORE_EVIDENCE_FAIBLE)


def _outil_search_sans_resultats() -> DefinitionOutil:
    return _outil_search(score=None)


class _LLMNonSollicite:
    """Doublure : ces tests ne doivent jamais invoquer de LLM."""

    def invoke(self, messages: Any):  # pragma: no cover - non sollicité ici
        raise AssertionError("Aucun LLM ne devrait être invoqué dans ce test.")


def _session_factice(*, fabriques, max_tentatives: int = 6):
    return construire_session(
        "Quelles sont les conditions de résiliation ?",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=fabriques,
        max_tentatives=max_tentatives,
    )


# ---------------------------------------------------------------------------
# noeud_rechercher
# ---------------------------------------------------------------------------


def test_rechercher_consomme_une_tentative_et_accumule_les_sources():
    session = _session_factice(fabriques=[_outil_search_evidence_forte])
    etat = EtatGraphe(session=session)

    mise_a_jour = nodes.noeud_rechercher(etat)

    assert mise_a_jour["session"] is session
    assert session.etat.tentatives == 1
    assert session.nombre_preuves == 1
    assert "outil" in session.etat.noms_etapes()


def test_rechercher_leve_si_le_budget_est_deja_epuise():
    from src.agent.state import BudgetTentativesEpuise

    session = _session_factice(fabriques=[_outil_search_evidence_forte], max_tentatives=1)
    session.etat.incrementer_tentative()
    etat = EtatGraphe(session=session)

    with pytest.raises(BudgetTentativesEpuise):
        nodes.noeud_rechercher(etat)


# ---------------------------------------------------------------------------
# noeud_evaluer_preuves : evidence forte / faible / vide
# ---------------------------------------------------------------------------


def test_evaluer_preuves_suffisant_si_evidence_forte():
    session = _session_factice(fabriques=[_outil_search_evidence_forte])
    session.executer_outil("search", requete="peu importe")
    etat = EtatGraphe(session=session)

    mise_a_jour = nodes.noeud_evaluer_preuves(etat)

    assert mise_a_jour["preuves_suffisantes"] is True
    derniere_trace = session.etat.trace[-1]
    assert derniere_trace.nom == "evaluation_preuves"
    assert derniere_trace.donnees["suffisant"] is True
    assert derniere_trace.donnees["score_pertinence_maximal"] == _SCORE_EVIDENCE_FORTE
    assert derniere_trace.donnees["seuil_pertinence"] == nodes.SEUIL_PERTINENCE_MINIMALE


def test_evaluer_preuves_insuffisant_si_evidence_faible():
    """
    Un passage existe (`nombre_preuves == 1`) mais son score de pertinence
    est sous le seuil : c'est exactement le cas que l'ancienne heuristique
    (nombre_preuves >= 1) ne détectait pas.
    """
    session = _session_factice(fabriques=[_outil_search_evidence_faible])
    session.executer_outil("search", requete="peu importe")
    etat = EtatGraphe(session=session)

    mise_a_jour = nodes.noeud_evaluer_preuves(etat)

    assert mise_a_jour["preuves_suffisantes"] is False
    derniere_trace = session.etat.trace[-1]
    assert derniere_trace.donnees["nombre_preuves"] == 1
    assert derniere_trace.donnees["suffisant"] is False
    assert derniere_trace.donnees["score_pertinence_maximal"] == _SCORE_EVIDENCE_FAIBLE


def test_evaluer_preuves_insuffisant_si_resultat_vide():
    session = _session_factice(fabriques=[_outil_search_sans_resultats])
    session.executer_outil("search", requete="peu importe")
    etat = EtatGraphe(session=session)

    mise_a_jour = nodes.noeud_evaluer_preuves(etat)

    assert mise_a_jour["preuves_suffisantes"] is False
    derniere_trace = session.etat.trace[-1]
    assert derniere_trace.donnees["nombre_preuves"] == 0
    assert derniere_trace.donnees["score_pertinence_maximal"] == 0.0


# ---------------------------------------------------------------------------
# router_apres_evaluation
# ---------------------------------------------------------------------------


def test_router_vers_generer_reponse_si_preuves_suffisantes():
    session = _session_factice(fabriques=[_outil_search_evidence_forte])
    etat = EtatGraphe(session=session, preuves_suffisantes=True)

    assert nodes.router_apres_evaluation(etat) == "generer_reponse"


def test_router_vers_reformuler_si_preuves_insuffisantes_et_budget_restant():
    session = _session_factice(fabriques=[_outil_search_evidence_faible], max_tentatives=3)
    session.etat.incrementer_tentative()
    etat = EtatGraphe(session=session, preuves_suffisantes=False)

    assert nodes.router_apres_evaluation(etat) == "reformuler"


def test_router_vers_generer_reponse_si_budget_epuise_meme_sans_preuves():
    """
    L'abstention n'est pas gérée ici : elle appartient à
    `generer_reponse` -> `src.rag.generation`, qui refuse déjà sourcé
    quand le contexte est insuffisant.
    """
    session = _session_factice(fabriques=[_outil_search_sans_resultats], max_tentatives=1)
    session.etat.incrementer_tentative()
    etat = EtatGraphe(session=session, preuves_suffisantes=False)

    assert not session.etat.peut_reessayer
    assert nodes.router_apres_evaluation(etat) == "generer_reponse"


def test_router_lit_la_decision_deja_calculee_sans_la_recalculer():
    """
    Le router ne doit jamais recalculer un jugement de pertinence à partir
    de `session` : il lit uniquement `etat.preuves_suffisantes`, pour ne
    jamais diverger de ce que `evaluer_preuves` a journalisé.
    """
    session = _session_factice(fabriques=[_outil_search_evidence_forte], max_tentatives=3)
    session.executer_outil("search", requete="peu importe")  # score fort réel

    etat_incoherent = EtatGraphe(session=session, preuves_suffisantes=False)

    assert nodes.router_apres_evaluation(etat_incoherent) == "reformuler"


# ---------------------------------------------------------------------------
# noeud_reformuler
# ---------------------------------------------------------------------------


class _LLMReformulationValide:
    def invoke(self, messages: Any) -> AIMessage:
        return AIMessage(content='{"requete_reformulee": "conditions résiliation contrat"}')


class _LLMReformulationJSONInvalide:
    def invoke(self, messages: Any) -> AIMessage:
        return AIMessage(content="ceci n'est pas du JSON")


class _LLMIndisponible:
    def invoke(self, messages: Any):
        raise RuntimeError("Ollama injoignable.")


def test_reformuler_met_a_jour_la_requete_courante():
    session = construire_session(
        "Question initiale introuvable",
        llm=_LLMReformulationValide(),
        charger_profil_domaine=False,
        fabriques=[_outil_search_sans_resultats],
    )
    etat = EtatGraphe(session=session)

    nodes.noeud_reformuler(etat)

    assert session.etat.requete_courante == "conditions résiliation contrat"
    assert session.etat.a_ete_reformulee


@pytest.mark.parametrize("faux_llm", [_LLMReformulationJSONInvalide(), _LLMIndisponible()])
def test_reformuler_echec_ne_casse_pas_et_journalise(faux_llm):
    session = construire_session(
        "Question initiale introuvable",
        llm=faux_llm,
        charger_profil_domaine=False,
        fabriques=[_outil_search_sans_resultats],
    )
    requete_avant = session.etat.requete_courante
    etat = EtatGraphe(session=session)

    nodes.noeud_reformuler(etat)

    assert session.etat.requete_courante == requete_avant
    assert session.etat.noms_etapes()[-1] == "reformulation_echec"


# ---------------------------------------------------------------------------
# noeud_generer_reponse
# ---------------------------------------------------------------------------


def test_generer_reponse_delegue_a_la_generation_rag(monkeypatch):
    appels: list[dict[str, Any]] = []

    class _ReponseFactice:
        contexte_suffisant = True
        sources: list[Any] = []

    def _fausse_generer_reponse(**kwargs: Any):
        appels.append(kwargs)
        return _ReponseFactice()

    monkeypatch.setattr(nodes, "generer_reponse", _fausse_generer_reponse)

    session = _session_factice(fabriques=[_outil_search_evidence_forte])
    session.executer_outil("search", requete="peu importe")
    etat = EtatGraphe(session=session)

    mise_a_jour = nodes.noeud_generer_reponse(etat)

    assert len(appels) == 1
    assert appels[0]["question"] == session.etat.requete_courante
    assert mise_a_jour["reponse"] is not None
    assert session.etat.noms_etapes()[-1] == "reponse"


def test_generer_reponse_ne_journalise_pas_contexte_suffisant_comme_tel(monkeypatch):
    """
    `resultat.contexte_suffisant` ne mesure qu'une validité structurelle
    (voir src/rag/validation.py::valider_contexte) : la trace agent ne doit
    pas exposer ce champ sous un nom qui laisserait croire à un jugement de
    pertinence, ce jugement étant déjà celui, distinct, de
    `evaluation_preuves`.
    """

    class _ReponseFactice:
        contexte_suffisant = True
        sources: list[Any] = []

    monkeypatch.setattr(nodes, "generer_reponse", lambda **kw: _ReponseFactice())

    session = _session_factice(fabriques=[_outil_search_evidence_forte])
    etat = EtatGraphe(session=session)

    nodes.noeud_generer_reponse(etat)

    donnees = session.etat.trace[-1].donnees
    assert "contexte_suffisant" not in donnees
    assert donnees["rag_contexte_structurellement_valide"] is True
