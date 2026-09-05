"""
Façade applicative `AgentService` (P2.1).

Tous les tests sont **hors ligne** : le point d'entrée P1 (`executer_agent`)
est remplacé par un fake. Aucun Ollama, aucun Qdrant, aucun graphe LangGraph
exécuté ici — la façade ne fait que déléguer.

Un unique test câble le vrai cœur P1 avec un LLM/corpus fictif (mêmes
doublures que `test_response.py`) pour prouver que la façade atteint bien P1
et renvoie un `AgentResponse` réel sans le déformer.
"""

from __future__ import annotations

import inspect
import json

import pytest

from src.agent import service as service_mod
from src.agent.graph import executer_agent
from src.agent.response import (
    STATUT_ERREUR,
    STATUT_SUCCES,
    AgentResponse,
)
from src.agent.service import CODE_REQUETE_INVALIDE, AgentService


# ---------------------------------------------------------------------------
# Fake du point d'entrée P1
# ---------------------------------------------------------------------------


class FauxPointEntree:
    """Compte les appels et enregistre les arguments reçus."""

    def __init__(self, reponse: AgentResponse | None = None, *, exception: BaseException | None = None) -> None:
        self.reponse = reponse or AgentResponse(status=STATUT_SUCCES, capability="search")
        self.exception = exception
        self.appels: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs) -> AgentResponse:
        self.appels.append((args, kwargs))
        if self.exception is not None:
            raise self.exception
        return self.reponse


def _reponse_riche() -> AgentResponse:
    return AgentResponse(
        status=STATUT_SUCCES,
        capability="compare",
        answer="Conclusion.",
        sources=[{"citation": "D1S1", "document": "a.pdf", "page": 1,
                  "categorie": "", "extrait": "x", "hors_perimetre": False}],
        citations=["D1S1", "D2S1"],
        warnings=["un avertissement"],
        data={"comparaison": {"documents": ["a.pdf", "b.pdf"]}},
        metadata={"documents_resolus": ["a.pdf", "b.pdf"], "nombre_sources": 1},
        error=None,
    )


# ---------------------------------------------------------------------------
# 1. Délégation
# ---------------------------------------------------------------------------


def test_query_valide_delegue_une_seule_fois() -> None:
    fake = FauxPointEntree()
    service = AgentService(point_entree=fake)

    service.query("Quelle est la date de la bataille de Valmy ?")

    assert len(fake.appels) == 1
    (args, kwargs) = fake.appels[0]
    assert args == ("Quelle est la date de la bataille de Valmy ?",)
    assert kwargs == {}


def test_query_propage_agentresponse_sans_alteration() -> None:
    attendue = _reponse_riche()
    service = AgentService(point_entree=FauxPointEntree(attendue))

    obtenue = service.query("Compare a.pdf et b.pdf.")

    # Objet identique, aucun champ retouché.
    assert obtenue is attendue
    assert obtenue.vers_dict() == attendue.vers_dict()
    # Toujours sérialisable en types natifs.
    assert json.loads(json.dumps(obtenue.vers_dict(), ensure_ascii=False)) == obtenue.vers_dict()


def test_options_session_transmises_telles_quelles() -> None:
    fake = FauxPointEntree()
    options = {"charger_profil_domaine": False, "max_tentatives": 2, "llm": object()}
    service = AgentService(point_entree=fake, options_session=options)

    service.query("une requête")

    (_, kwargs) = fake.appels[0]
    assert kwargs == options


def test_options_session_copiee_defensivement() -> None:
    fake = FauxPointEntree()
    options = {"max_tentatives": 3}
    service = AgentService(point_entree=fake, options_session=options)

    options["max_tentatives"] = 99  # mutation après construction
    service.query("q")

    (_, kwargs) = fake.appels[0]
    assert kwargs == {"max_tentatives": 3}


def test_defaut_utilise_executer_agent() -> None:
    # Câblage par défaut = point d'entrée public P1, sans l'appeler.
    assert AgentService()._point_entree is executer_agent


# ---------------------------------------------------------------------------
# 2. Entrée invalide — gérée sans toucher le cœur
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entree", ["", "   ", "\n\t", None, 42, [], {"q": "x"}])
def test_query_entree_invalide_ne_touche_pas_le_coeur(entree) -> None:
    fake = FauxPointEntree()
    service = AgentService(point_entree=fake)

    reponse = service.query(entree)  # type: ignore[arg-type]

    assert isinstance(reponse, AgentResponse)
    assert reponse.status == STATUT_ERREUR
    assert reponse.capability == ""
    assert reponse.error == {
        "code": CODE_REQUETE_INVALIDE,
        "message": "La requête doit être une chaîne de caractères non vide.",
    }
    assert fake.appels == []  # le cœur n'a jamais été appelé


# ---------------------------------------------------------------------------
# 3. Exceptions — la façade ne propage jamais
# ---------------------------------------------------------------------------


def test_query_exception_construction_donne_status_error() -> None:
    from src.agent.session import ErreurSession

    fake = FauxPointEntree(exception=ErreurSession("La requête de la session est vide."))
    service = AgentService(point_entree=fake)

    reponse = service.query("requête non vide côté façade")

    assert isinstance(reponse, AgentResponse)
    assert reponse.status == STATUT_ERREUR
    assert reponse.capability == ""
    assert reponse.error == {
        "code": "ErreurSession",
        "message": "La requête de la session est vide.",
    }


def test_query_exception_inattendue_donne_status_error() -> None:
    fake = FauxPointEntree(exception=RuntimeError("panne imprévue"))
    service = AgentService(point_entree=fake)

    reponse = service.query("une requête")

    assert reponse.status == STATUT_ERREUR
    assert reponse.error == {"code": "RuntimeError", "message": "panne imprévue"}


def test_query_ne_leve_jamais() -> None:
    for exc in (ValueError("x"), KeyError("y"), RuntimeError("z")):
        service = AgentService(point_entree=FauxPointEntree(exception=exc))
        assert service.query("q").status == STATUT_ERREUR


# ---------------------------------------------------------------------------
# 4. Aucune logique de capacité / de cœur dupliquée dans la façade
# ---------------------------------------------------------------------------


def test_service_ne_duplique_aucune_logique_de_coeur() -> None:
    source = inspect.getsource(service_mod)

    interdits = [
        "langgraph", "StateGraph", "EtatGraphe",
        "src.rag", "rechercher_passages", "charger_document", "generer_",
        "src.tools", "multidoc", "detecter_intention", "resoudre_document",
        "normaliser_reponse_agent", "_juger_suffisance", "reformul",
        "compare", "synthesize", "summarize", "classify", "extract",
    ]
    presents = [mot for mot in interdits if mot in source]
    assert not presents, f"la façade référence de la logique de cœur : {presents}"


def test_service_importe_seulement_le_point_dentree_et_la_reponse() -> None:
    # Frontière stricte : la façade ne dépend que du point d'entrée public P1
    # et du contrat de sortie. Rien d'autre du paquet `src.agent`.
    arbre = inspect.getsource(service_mod)
    assert "from src.agent.graph import executer_agent" in arbre
    assert "from src.agent.response import" in arbre
    for module_interne in ("src.agent.nodes", "src.agent.session",
                           "src.agent.graph_state", "src.agent.multidoc"):
        assert module_interne not in arbre


def test_surface_publique_minimale() -> None:
    publics = [n for n in dir(AgentService) if not n.startswith("_")]
    assert publics == ["query"]


# ---------------------------------------------------------------------------
# 5. Bout en bout : la façade atteint réellement le cœur P1 (offline)
# ---------------------------------------------------------------------------


def test_query_bout_en_bout_atteint_le_coeur_p1(monkeypatch) -> None:
    """Vrai `executer_agent`, corpus + LLM fictifs — aucun réseau."""
    from langchain_core.messages import AIMessage

    from src.agent import multidoc_pipeline
    from tests.agent._multidoc_fakes import LLMScripte, cabler_corpus, passage

    cabler_corpus(
        monkeypatch,
        multidoc_pipeline,
        fiches={"rapport_alpha.pdf": "A", "rapport_beta.pdf": "B"},
        passages_par_doc={
            "A": [passage("A", 1, "Le taux est de 3 %.", page=1)],
            "B": [passage("B", 1, "Le taux est de 5 %.", page=1)],
        },
    )

    class _LLM:
        def invoke(self, messages, think: bool | None = None):
            if "deux intentions possibles" in messages[0].content:
                return AIMessage(content='{"intention": "SEARCH"}')
            return LLMScripte().invoke(messages, think=think)

    service = AgentService(options_session={"llm": _LLM(), "charger_profil_domaine": False})
    reponse = service.query("Compare rapport_alpha.pdf et rapport_beta.pdf.")

    assert isinstance(reponse, AgentResponse)
    assert reponse.status == STATUT_SUCCES
    assert reponse.capability == "compare"
    assert "comparaison" in reponse.data
    assert reponse.metadata["documents_resolus"]
