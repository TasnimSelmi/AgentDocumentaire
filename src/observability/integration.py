"""
Point d'intégration FastAPI — `install_observability(app, *, sink=None)`.

Il :

- résout le sink (paramètre explicite, sinon `LoggingTraceSink` par défaut) ;
- installe `CorrelationMiddleware` (ASGI pur) ;
- **enveloppe** `app.state.agent_service` et `app.state.ingestion_service` dans
  leurs wrappers observants ;
- expose un `ObservabilityRuntime` sur `app.state.observability` (lu par le
  gestionnaire d'erreurs HTTP pour émettre `http_unhandled_error`).

Aucun global mutable : tout vit sur `app.state`. Deux `create_app()` avec des
sinks distincts n'interfèrent pas. `src/api/routes.py` reste sans aucune
logique d'observabilité.

Quand `Settings.observability_enabled` est faux, l'installation est un **no-op**
métier : ni middleware, ni wrappers ; `app.state.observability` porte alors un
runtime désactivé (sink `Null`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import Settings, get_settings
from src.observability.instrumentation import (
    InstrumentedAgentService,
    InstrumentedIngestionService,
)
from src.observability.middleware import CorrelationMiddleware
from src.observability.sinks import LoggingTraceSink, NullTraceSink, TraceSink


@dataclass(frozen=True)
class ObservabilityRuntime:
    """État d'observabilité attaché à une application. Immuable."""

    sink: TraceSink
    enabled: bool
    emit_start: bool


def install_observability(
    app: Any,
    *,
    sink: TraceSink | None = None,
    settings: Settings | None = None,
) -> ObservabilityRuntime:
    """
    Câble l'observabilité sur `app` et renvoie le `ObservabilityRuntime` posé
    sur `app.state.observability`.

    `app` est un `FastAPI` (typé `Any` pour ne pas importer FastAPI ici).
    `sink` explicite l'emporte toujours sur le défaut.
    """
    reglages = settings if settings is not None else get_settings()
    enabled = bool(getattr(reglages, "observability_enabled", True))
    emit_start = bool(getattr(reglages, "observability_emit_start", True))

    if not enabled:
        runtime = ObservabilityRuntime(
            sink=NullTraceSink(), enabled=False, emit_start=False
        )
        app.state.observability = runtime
        return runtime

    sink_actif: TraceSink = sink if sink is not None else LoggingTraceSink()

    app.add_middleware(CorrelationMiddleware)
    app.state.agent_service = InstrumentedAgentService(
        app.state.agent_service, sink_actif, emit_start=emit_start
    )
    app.state.ingestion_service = InstrumentedIngestionService(
        app.state.ingestion_service, sink_actif, emit_start=emit_start
    )

    runtime = ObservabilityRuntime(
        sink=sink_actif, enabled=True, emit_start=emit_start
    )
    app.state.observability = runtime
    return runtime


__all__ = ["ObservabilityRuntime", "install_observability"]
