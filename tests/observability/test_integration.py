"""
`install_observability` — câblage sur une app factice, injection explicite du
sink, absence de global mutable, respect de `Settings.observability_enabled`.
"""

from __future__ import annotations

import pytest

from src.config import Settings
from src.observability import (
    InstrumentedAgentService,
    InstrumentedIngestionService,
    NullTraceSink,
)
from src.observability.integration import ObservabilityRuntime, install_observability
from src.observability.middleware import CorrelationMiddleware
from tests.observability.conftest import (
    CapturingSink,
    EspionAgentInner,
    EspionIngestionInner,
)


class _FauxApp:
    """Substitut minimal de `FastAPI` : `state` + `add_middleware`."""

    class _State:
        pass

    def __init__(self):
        self.state = self._State()
        self.middlewares: list = []

    def add_middleware(self, cls, **kw):
        self.middlewares.append((cls, kw))


def _app_avec_services():
    app = _FauxApp()
    app.state.agent_service = EspionAgentInner()
    app.state.ingestion_service = EspionIngestionInner()
    return app


def test_install_enveloppe_les_services_et_installe_le_middleware():
    app = _app_avec_services()
    sink = CapturingSink()

    runtime = install_observability(app, sink=sink, settings=Settings())

    assert isinstance(runtime, ObservabilityRuntime)
    assert runtime.enabled is True
    assert runtime.sink is sink
    assert app.state.observability is runtime
    assert isinstance(app.state.agent_service, InstrumentedAgentService)
    assert isinstance(app.state.ingestion_service, InstrumentedIngestionService)
    assert (CorrelationMiddleware, {}) in app.middlewares


def test_sink_injecte_est_utilise_par_les_wrappers():
    app = _app_avec_services()
    sink = CapturingSink()
    install_observability(app, sink=sink, settings=Settings())

    app.state.agent_service.query("q")

    assert sink.noms() == ["agent_execution_started", "agent_execution_completed"]


def test_deux_apps_sinks_distincts_sans_contamination():
    app_a, app_b = _app_avec_services(), _app_avec_services()
    sink_a, sink_b = CapturingSink(), CapturingSink()

    install_observability(app_a, sink=sink_a, settings=Settings())
    install_observability(app_b, sink=sink_b, settings=Settings())

    app_a.state.agent_service.query("qa")

    assert len(sink_a.events) == 2
    assert sink_b.events == []


def test_emit_start_pilote_par_settings():
    app = _app_avec_services()
    sink = CapturingSink()
    install_observability(
        app, sink=sink, settings=Settings(observability_emit_start=False)
    )

    app.state.agent_service.query("q")

    assert sink.noms() == ["agent_execution_completed"]


def test_observability_desactivee_est_un_noop_metier():
    app = _app_avec_services()
    agent_avant = app.state.agent_service
    ingestion_avant = app.state.ingestion_service

    runtime = install_observability(
        app, sink=CapturingSink(), settings=Settings(observability_enabled=False)
    )

    assert runtime.enabled is False
    assert isinstance(runtime.sink, NullTraceSink)
    # services non enveloppés, middleware non installé
    assert app.state.agent_service is agent_avant
    assert app.state.ingestion_service is ingestion_avant
    assert app.middlewares == []
    assert app.state.observability is runtime


def test_default_sink_est_logging_quand_non_fourni():
    app = _app_avec_services()
    runtime = install_observability(app, settings=Settings())
    # pas un CapturingSink de test ; c'est le sink de production
    assert type(runtime.sink).__name__ == "LoggingTraceSink"
