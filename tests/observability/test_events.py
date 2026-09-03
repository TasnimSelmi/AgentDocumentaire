"""
Modèle d'événements — enveloppe versionnée fermée, attributs typés,
immuabilité, et couverture de l'allow-list de redaction.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.observability.events import (
    EVENEMENTS,
    OUTCOMES,
    SCHEMA_VERSION,
    AgentExecutionAttributes,
    HttpErrorAttributes,
    IngestionAttributes,
    ObservabilityEvent,
    noms_de_champs,
)
from src.observability.redaction import (
    _agent_attrs_vers_log,
    _http_attrs_vers_log,
    _ingestion_attrs_vers_log,
)


def test_schema_version_est_1_0():
    assert SCHEMA_VERSION == "1.0"
    assert ObservabilityEvent(event="x", operation_id="op").schema_version == "1.0"


def test_noms_d_evenements_stables():
    assert EVENEMENTS == (
        "agent_execution_started",
        "agent_execution_completed",
        "agent_execution_failed",
        "ingestion_started",
        "ingestion_completed",
        "ingestion_failed",
        "http_unhandled_error",
    )


def test_outcomes_fermes():
    assert OUTCOMES == ("success", "refusal", "partial", "error")


def test_enveloppe_est_immuable():
    event = ObservabilityEvent(event="agent_execution_completed", operation_id="op")
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.event = "autre"  # type: ignore[misc]


def test_attributs_sont_immuables():
    a = AgentExecutionAttributes(capability="search")
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.capability = "x"  # type: ignore[misc]


def test_enveloppe_champs_attendus():
    event = ObservabilityEvent(
        event="agent_execution_completed",
        operation_id="op-1",
        request_id="req-1",
        started_at="2026-09-03T10:00:00+00:00",
        finished_at="2026-09-03T10:00:01+00:00",
        duration_ms=1000.0,
        outcome="success",
        attributes=AgentExecutionAttributes(capability="search"),
    )
    assert set(noms_de_champs(ObservabilityEvent)) == {
        "event",
        "operation_id",
        "schema_version",
        "request_id",
        "started_at",
        "finished_at",
        "duration_ms",
        "outcome",
        "attributes",
    }
    assert event.outcome == "success"


def test_agent_attributes_type_ferme():
    assert noms_de_champs(AgentExecutionAttributes) == (
        "capability",
        "documents",
        "citations",
        "source_count",
        "refusal_code",
        "error_category",
        "error_code",
        "error_message",
        "error_stack",
    )


def test_ingestion_attributes_type_ferme():
    assert noms_de_champs(IngestionAttributes) == (
        "source",
        "profile",
        "files_found",
        "files_processed",
        "files_failed",
        "files_skipped",
        "files_empty",
        "files_ocr",
        "files_deleted",
        "chunks_indexed",
        "error_category",
        "error_code",
        "error_message",
        "error_stack",
    )


@pytest.mark.parametrize(
    ("dc", "serializer"),
    [
        (AgentExecutionAttributes, _agent_attrs_vers_log),
        (IngestionAttributes, _ingestion_attrs_vers_log),
        (HttpErrorAttributes, _http_attrs_vers_log),
    ],
)
def test_allow_list_couvre_toute_la_surface_des_attributs(dc, serializer):
    """Chaque champ d'une dataclass d'attributs est explicitement traité par
    l'allow-list — sinon un champ ajouté fuiterait ou disparaîtrait en silence."""
    sortie = serializer(dc())
    assert set(sortie) == set(noms_de_champs(dc))
