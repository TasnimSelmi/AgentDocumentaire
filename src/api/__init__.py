"""
Couche API HTTP (P2.3) — transport, validation et adaptation HTTP au-dessus
des façades applicatives déjà gelées.

Cette couche ne contient **aucune** intelligence documentaire : ni routage, ni
capacité (SEARCH / SUMMARIZE / CLASSIFY / EXTRACT / COMPARE / SYNTHESIZE), ni
ingestion, ni résolution documentaire, ni logique Qdrant, ni re-normalisation
d'`AgentResponse`. Chaque route fait exactement :

    validation HTTP  →  un appel de service  →  adaptation de la réponse HTTP

Points d'entrée consommés (inchangés) :
    - `src.agent.service.AgentService.query`      (cœur P1, P2.1)
    - `src.sources.service.IngestionService.sync` (ingestion multi-source, P2.2)

Contrat de sortie de `/query` : `AgentResponse.vers_dict()` tel quel
(`src/agent/response.py`, §7.5 de `docs/architecture.md`). Aucun modèle
Pydantic miroir n'est défini.

Voir `docs/P2.3_API.md`.
"""

from src.api.app import create_app

__all__ = ["create_app"]
