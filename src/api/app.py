"""
Fabrique de l'application FastAPI.

`create_app` câble les trois routes et les gestionnaires d'erreur, et place
sur `app.state` les collaborateurs — injectables pour les tests, construits
paresseusement sinon. Aucune logique documentaire, aucun accès réseau ici.

Lancement standard (aucun module `__main__` fourni, cf. design P2.3) :

    uvicorn "src.api:create_app" --factory --host 127.0.0.1 --port 8000

⚠️ MVP : pas d'authentification (hors périmètre P2.3). `POST /ingestion`
**mute l'index** ; cette API n'est pas destinée à être exposée publiquement
sans couche d'authentification en amont. Voir `docs/P2.3_API.md`.
"""

from __future__ import annotations

from fastapi import FastAPI

from src.agent.service import AgentService
from src.api.dependencies import (
    RegistreSources,
    agent_service_par_defaut,
    ingestion_service_par_defaut,
    registre_sources_par_defaut,
)
from src.api.errors import enregistrer_gestionnaires_erreurs
from src.api.routes import router
from src.observability import TraceSink, install_observability
from src.sources import IngestionService

_DESCRIPTION = (
    "API HTTP mince du MVP AgentDocumentaire (P2.3) : transport et validation "
    "au-dessus de `AgentService` (P1) et `IngestionService` (P2.2). "
    "`/query` renvoie le contrat public `AgentResponse`. MVP sans "
    "authentification — ne pas exposer publiquement en l'état."
)


def create_app(
    *,
    agent_service: AgentService | None = None,
    ingestion_service: IngestionService | None = None,
    sources: RegistreSources | None = None,
    sink: TraceSink | None = None,
) -> FastAPI:
    """Construit l'app. Tout collaborateur omis prend sa valeur par défaut.

    `sink` (P2.4) : destination de traces d'observabilité injectée
    explicitement (défaut : `LoggingTraceSink` — JSON structuré sur stdout).
    Deux `create_app()` avec des sinks distincts n'interfèrent pas.
    """
    app = FastAPI(
        title="AgentDocumentaire API",
        version="0.2.3",
        description=_DESCRIPTION,
    )

    app.state.agent_service = agent_service or agent_service_par_defaut()
    app.state.ingestion_service = ingestion_service or ingestion_service_par_defaut()
    app.state.sources = (
        sources if sources is not None else registre_sources_par_defaut()
    )

    enregistrer_gestionnaires_erreurs(app)
    app.include_router(router)

    # P2.4 : couche d'observabilité transverse. Installe le middleware de
    # corrélation ASGI et enveloppe les services dans leurs wrappers observants.
    # `src/api/routes.py` reste sans aucune logique d'observabilité.
    install_observability(app, sink=sink)
    return app


__all__ = ["create_app"]
