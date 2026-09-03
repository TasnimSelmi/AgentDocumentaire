"""
`InstrumentedAgentService` — délégation unique, réponse intacte,
started/completed/failed, refus, erreur métier, exception inattendue, durée
déterministe, cohérence hors HTTP.
"""

from __future__ import annotations

import pytest

from src.observability.correlation import (
    current_request_id,
    delier_correlation,
    lier_correlation,
)
from src.observability.events import (
    AGENT_EXECUTION_COMPLETED,
    AGENT_EXECUTION_FAILED,
    AGENT_EXECUTION_STARTED,
)
from src.observability.instrumentation import InstrumentedAgentService
from tests.observability.conftest import (
    EspionAgentInner,
    FauxAgentResponse,
    SinkQuiExplose,
)


def test_inner_appele_exactement_une_fois_et_reponse_identique(sink):
    reponse = FauxAgentResponse(status="success", capability="compare", answer="C.")
    inner = EspionAgentInner(reponse)
    service = InstrumentedAgentService(inner, sink)

    resultat = service.query("compare a et b")

    assert inner.appels == ["compare a et b"]
    assert resultat is reponse  # objet métier renvoyé tel quel


def test_started_puis_completed_sur_succes(sink):
    service = InstrumentedAgentService(EspionAgentInner(FauxAgentResponse()), sink)
    service.query("q")

    assert sink.noms() == [AGENT_EXECUTION_STARTED, AGENT_EXECUTION_COMPLETED]
    started, completed = sink.events
    assert started.outcome is None and started.finished_at is None
    assert started.duration_ms is None
    assert completed.outcome == "success"
    assert completed.request_id == started.request_id
    assert completed.operation_id == started.operation_id
    assert completed.duration_ms is not None and completed.finished_at is not None


def test_emit_start_desactive_ne_produit_que_completed(sink):
    service = InstrumentedAgentService(
        EspionAgentInner(FauxAgentResponse()), sink, emit_start=False
    )
    service.query("q")
    assert sink.noms() == [AGENT_EXECUTION_COMPLETED]


def test_documents_et_citations_derives_du_contrat(sink):
    reponse = FauxAgentResponse(
        status="success",
        capability="compare",
        citations=["D1S1", "D2S1"],
        sources=[
            {"document": "a.pdf", "citation": "D1S1"},
            {"document": "b.pdf", "citation": "D2S1"},
            {"document": "a.pdf", "citation": "D1S2"},
        ],
        metadata={"documents_resolus": ["a.pdf", "b.pdf"]},
    )
    InstrumentedAgentService(EspionAgentInner(reponse), sink).query("q")
    attrs = sink.unique(AGENT_EXECUTION_COMPLETED).attributes
    assert attrs.capability == "compare"
    assert attrs.documents == ("a.pdf", "b.pdf")
    assert attrs.citations == ("D1S1", "D2S1")
    assert attrs.source_count == 3


def test_documents_derives_des_sources_si_pas_de_metadata(sink):
    reponse = FauxAgentResponse(
        status="success",
        sources=[{"document": "x.pdf"}, {"document": "x.pdf"}, {"document": "y.pdf"}],
    )
    InstrumentedAgentService(EspionAgentInner(reponse), sink).query("q")
    assert sink.unique(AGENT_EXECUTION_COMPLETED).attributes.documents == (
        "x.pdf",
        "y.pdf",
    )


def test_refus_metier_est_completed_outcome_refusal(sink):
    reponse = FauxAgentResponse(
        status="refusal",
        capability="extract",
        answer="Précisez le document.",
        error={"code": "perimetre_non_resolu", "message": "Précisez le document."},
    )
    InstrumentedAgentService(EspionAgentInner(reponse), sink).query("q")
    event = sink.unique(AGENT_EXECUTION_COMPLETED)
    assert event.outcome == "refusal"
    assert event.attributes.refusal_code == "perimetre_non_resolu"
    assert event.attributes.error_code is None  # un refus n'est pas une erreur


def test_erreur_metier_est_failed_outcome_error(sink):
    reponse = FauxAgentResponse(
        status="error",
        capability="",
        error={"code": "ConnectionError", "message": "qdrant refused /home/x token=z"},
    )
    InstrumentedAgentService(EspionAgentInner(reponse), sink).query("q")
    event = sink.unique(AGENT_EXECUTION_FAILED)
    assert event.outcome == "error"
    assert event.attributes.error_category == "agent_error"
    assert event.attributes.error_code == "ConnectionError"
    # message brut conservé dans l'événement (scrub à la sérialisation du sink)
    assert event.attributes.error_stack is None


def test_exception_inattendue_emet_failed_puis_releve(sink):
    inner = EspionAgentInner(exception=RuntimeError("boom /opt/app/key"))
    service = InstrumentedAgentService(inner, sink)

    with pytest.raises(RuntimeError):
        service.query("q")

    assert sink.noms() == [AGENT_EXECUTION_STARTED, AGENT_EXECUTION_FAILED]
    failed = sink.unique(AGENT_EXECUTION_FAILED)
    assert failed.outcome == "error"
    assert failed.attributes.error_category == "unexpected"
    assert failed.attributes.error_code == "RuntimeError"
    assert failed.attributes.error_stack is not None


def test_duree_deterministe_avec_monkeypatch(monkeypatch, sink):
    valeurs = iter([100.0, 100.25])  # secondes perf_counter
    monkeypatch.setattr("src.observability.instrumentation.time.perf_counter", lambda: next(valeurs))

    InstrumentedAgentService(EspionAgentInner(FauxAgentResponse()), sink, emit_start=False).query("q")

    assert sink.unique(AGENT_EXECUTION_COMPLETED).duration_ms == pytest.approx(250.0)


def test_sink_defaillant_n_impacte_pas_l_appel(monkeypatch):
    inner = EspionAgentInner(FauxAgentResponse(answer="ok"))
    service = InstrumentedAgentService(inner, SinkQuiExplose())
    # Ne lève pas, renvoie la réponse métier.
    assert service.query("q").answer == "ok"


def test_hors_http_contexte_correlation_complet_puis_reset(sink):
    assert current_request_id() is None
    InstrumentedAgentService(EspionAgentInner(FauxAgentResponse()), sink).query("q")

    started, completed = sink.events
    assert started.request_id and started.operation_id
    assert started.request_id == completed.request_id
    assert started.operation_id == completed.operation_id
    # le contexte temporaire est réinitialisé après l'appel
    assert current_request_id() is None


def test_sous_contexte_http_les_ids_sont_reutilises(sink):
    tokens = lier_correlation("req-http", "exe-http")
    try:
        InstrumentedAgentService(EspionAgentInner(FauxAgentResponse()), sink).query("q")
    finally:
        delier_correlation(tokens)

    for event in sink.events:
        assert event.request_id == "req-http"
        assert event.operation_id == "exe-http"


def test_inner_appele_une_fois_meme_avec_emit_start(sink):
    inner = EspionAgentInner(FauxAgentResponse())
    InstrumentedAgentService(inner, sink, emit_start=True).query("q")
    assert len(inner.appels) == 1
