"""
Façade applicative au-dessus du cœur agentique P1.

`AgentService` est la **seule** porte d'entrée que les couches applicatives à
venir (FastAPI, UI, connecteurs documentaires de l'entreprise) doivent
utiliser. Elles ne doivent jamais importer LangGraph, le module du graphe
(`graph.py`), ses nœuds (`nodes.py`) ni aucune structure interne du graphe.

Couche **MINCE**, sans logique métier. Elle ne réimplémente aucune des six
capacités, ni le routage, ni les outils, ni le socle documentaire, ni la
résolution du périmètre, ni la génération, ni la mise en forme du contrat de
sortie. Tout cela reste dans le cœur P1, appelé via son point d'entrée public
unique `executer_agent(...)`.

Ce que la façade fait, et rien de plus :
    1. une validation légère de l'entrée (chaîne non vide) ;
    2. un unique appel au point d'entrée public P1 ;
    3. la propagation **intacte** de l'`AgentResponse` produit ;
    4. la garantie qu'elle ne lève jamais : une entrée invalide ou une erreur
       de construction de session ressort en `AgentResponse(status="error")`,
       cohérent avec ce que `executer_agent` renvoie déjà sur exception du
       graphe. Les couches API/UI n'ont donc aucun `try/except` à écrire.

Frontière P1 / P2 : voir `docs/architecture.md` §7.7.
"""

from __future__ import annotations

from typing import Any, Callable

from src.agent.graph import executer_agent
from src.agent.response import STATUT_ERREUR, AgentResponse

#: Signature du point d'entrée P1 délégué. Injectable pour les tests (un fake
#: suffit alors, sans Ollama ni Qdrant).
PointEntreeAgent = Callable[..., AgentResponse]

#: Code d'erreur propre à la façade (jamais produit par le cœur P1) : l'entrée
#: n'est pas une chaîne exploitable. Les erreurs techniques réutilisent le nom
#: de l'exception, comme `executer_agent`.
CODE_REQUETE_INVALIDE = "requete_invalide"


class AgentService:
    """
    Façade synchrone et sans état du cœur agentique P1.

    `query()` renvoie **toujours** un `AgentResponse` — jamais d'exception,
    ni pour une entrée invalide, ni pour une erreur de construction de
    session (`ErreurSession`, profil de domaine introuvable…). Ces cas
    ressortent en `AgentResponse(status="error")`.

    Aucune option n'est interprétée par la façade : `options_session` est
    transmise telle quelle à `executer_agent` (donc à `construire_session` :
    `llm`, `profil_domaine`, `charger_profil_domaine`, `max_tentatives`, …).
    """

    def __init__(
        self,
        *,
        point_entree: PointEntreeAgent = executer_agent,
        options_session: dict[str, Any] | None = None,
    ) -> None:
        self._point_entree = point_entree
        # Copie défensive : la façade ne doit pas voir sa configuration mutée
        # sous elle après construction.
        self._options_session = dict(options_session or {})

    def query(self, requete: str) -> AgentResponse:
        """
        Traite une requête utilisateur via le cœur agentique P1.

        - `requete` doit être une chaîne non vide (après `strip`). Sinon ->
          `AgentResponse(status="error", error.code="requete_invalide")`,
          **sans** appeler le cœur.
        - Sinon : un **unique** appel à
          `executer_agent(requete, **options_session)`. L'`AgentResponse`
          produit est renvoyé **tel quel** (aucune altération fonctionnelle).
        - Toute exception levée avant/pendant cet appel (typiquement une
          `ErreurSession` de `construire_session`, hors périmètre du
          `try/except` interne de `executer_agent`) est convertie en
          `AgentResponse(status="error")`.
        """
        if not isinstance(requete, str) or not requete.strip():
            return AgentResponse(
                status=STATUT_ERREUR,
                capability="",
                error={
                    "code": CODE_REQUETE_INVALIDE,
                    "message": "La requête doit être une chaîne de caractères non vide.",
                },
            )

        try:
            return self._point_entree(requete, **self._options_session)
        except Exception as exc:  # noqa: BLE001 — la façade ne propage jamais
            return AgentResponse(
                status=STATUT_ERREUR,
                capability="",
                error={"code": type(exc).__name__, "message": str(exc)},
            )


__all__ = [
    "AgentService",
    "PointEntreeAgent",
    "CODE_REQUETE_INVALIDE",
]
