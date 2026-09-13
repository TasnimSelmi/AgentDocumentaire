"""
Fabrique de l'application FastAPI.

`create_app` câble les routes API, les gestionnaires d'erreur et le frontend
statique (`src/ui/`), et place sur `app.state` les collaborateurs —
injectables pour les tests, construits paresseusement sinon. Aucune logique
documentaire, aucun accès réseau ici.

Lancement recommandé (un seul process, un seul port) : `python scripts/run.py`
— ouvre ensuite `http://127.0.0.1:8000/`. Équivalent manuel :

    uvicorn "src.api:create_app" --factory --host 127.0.0.1 --port 8000

⚠️ MVP : pas d'authentification (hors périmètre P2.3). `POST /ingestion`
**mute l'index** ; cette API n'est pas destinée à être exposée publiquement
sans couche d'authentification en amont. Voir `docs/P2.3_API.md`.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

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

#: Frontend statique (P2.5+) : pages HTML autonomes (client HTTP pur vers
#: cette même API, cf. docs/DO_NOT_TOUCH.md §4ter). Servies ici uniquement
#: pour offrir UN process/UN port/UNE origine au lancement — aucune logique
#: documentaire, aucune route API ajoutée à l'OpenAPI (`include_in_schema`
#: reste par défaut sur `/`, exclu explicitement ci-dessous).
_DOSSIER_UI = Path(__file__).resolve().parent.parent / "ui"
_PAGE_ACCUEIL = "Gestion des corpus.html"

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

    # MVP sans authentification (cf. module docstring) : les deux pages
    # frontend sont des fichiers HTML statiques ouverts en local (origine
    # `file://` ou un simple serveur statique de dev), jamais un domaine
    # de confiance à restreindre finement ici. Pas de cookies/identifiants
    # transportés (`allow_credentials` reste `False`).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    enregistrer_gestionnaires_erreurs(app)
    app.include_router(router)

    # P2.4 : couche d'observabilité transverse. Installe le middleware de
    # corrélation ASGI et enveloppe les services dans leurs wrappers observants.
    # `src/api/routes.py` reste sans aucune logique d'observabilité.
    install_observability(app, sink=sink)

    # Frontend statique : une seule origine, un seul port. `/ui/*` sert les
    # fichiers de `src/ui/` tels quels (les liens relatifs entre pages —
    # `Agent Documentaire.html?corpus_id=...` — continuent de fonctionner
    # sans modification). `/` redirige vers la page d'accueil pour éviter à
    # l'utilisateur de connaître un nom de fichier encodé.
    if _DOSSIER_UI.is_dir():
        app.mount("/ui", StaticFiles(directory=_DOSSIER_UI, html=True), name="ui")

        @app.get("/", include_in_schema=False)
        def _accueil() -> RedirectResponse:
            return RedirectResponse(url="/ui/" + quote(_PAGE_ACCUEIL))

    return app


__all__ = ["create_app"]
