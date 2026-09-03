"""
Sinks — `NullTraceSink` inerte, `LoggingTraceSink` JSON valide, configuration
locale idempotente, root logger préservé, robustesse d'`emit`.
"""

from __future__ import annotations

import io
import json
import logging

from src.observability.events import (
    AGENT_EXECUTION_COMPLETED,
    AgentExecutionAttributes,
    ObservabilityEvent,
)
from src.observability.sinks import LoggingTraceSink, NullTraceSink, TraceSink


def _event(**kw) -> ObservabilityEvent:
    base = dict(
        event=AGENT_EXECUTION_COMPLETED,
        operation_id="op-1",
        request_id="req-1",
        started_at="2026-09-03T10:00:00+00:00",
        finished_at="2026-09-03T10:00:01+00:00",
        duration_ms=1000.0,
        outcome="success",
        attributes=AgentExecutionAttributes(capability="search", source_count=1),
    )
    base.update(kw)
    return ObservabilityEvent(**base)


def test_null_sink_est_inerte_et_respecte_le_protocole():
    sink = NullTraceSink()
    assert isinstance(sink, TraceSink)
    assert sink.emit(_event()) is None


def test_logging_sink_emet_une_ligne_json_valide():
    buf = io.StringIO()
    LoggingTraceSink(stream=buf).emit(_event())

    lignes = [l for l in buf.getvalue().splitlines() if l.strip()]
    assert len(lignes) == 1
    charge = json.loads(lignes[0])
    assert charge["event"] == "agent_execution_completed"
    assert charge["schema_version"] == "1.0"
    assert charge["request_id"] == "req-1"
    assert charge["operation_id"] == "op-1"
    assert charge["outcome"] == "success"
    assert charge["attributes"]["capability"] == "search"
    assert charge["level"] == "info"


def test_logging_sink_configuration_idempotente():
    """Instancier plusieurs `LoggingTraceSink` (dans le même process) n'accumule
    aucun état : chacun écrit exactement une ligne par événement sur son flux."""
    for _ in range(5):
        buf = io.StringIO()
        sink = LoggingTraceSink(stream=buf)
        sink.emit(_event())
        sink.emit(_event())
        lignes = [l for l in buf.getvalue().splitlines() if l.strip()]
        assert len(lignes) == 2
        for ligne in lignes:
            json.loads(ligne)


def test_logging_sink_ne_touche_pas_le_root_logger():
    root = logging.getLogger()
    handlers_avant = list(root.handlers)
    niveau_avant = root.level

    buf = io.StringIO()
    sink = LoggingTraceSink(stream=buf)
    sink.emit(_event())

    assert list(root.handlers) == handlers_avant
    assert root.level == niveau_avant


def test_logging_sink_deux_flux_independants():
    a, b = io.StringIO(), io.StringIO()
    LoggingTraceSink(stream=a).emit(_event(request_id="A"))
    LoggingTraceSink(stream=b).emit(_event(request_id="B"))
    assert json.loads(a.getvalue())["request_id"] == "A"
    assert json.loads(b.getvalue())["request_id"] == "B"
    assert "B" not in a.getvalue()


class _FluxCassé(io.StringIO):
    def write(self, _s):  # noqa: D401
        raise OSError("flux indisponible")


def test_logging_sink_emit_qui_echoue_n_explose_pas():
    sink = LoggingTraceSink(stream=_FluxCassé())
    # Ne lève pas : une trace ne casse jamais la requête métier.
    assert sink.emit(_event()) is None
