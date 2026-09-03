"""
Observabilité vue de l'API (P2.4) — corrélation HTTP de bout en bout, émission
d'événements via `create_app`, isolation entre deux apps, `http_unhandled_error`,
non-régression du contrat P2.3 (`/query`, `/health`, OpenAPI).
"""

from __future__ import annotations

import json

import pytest

from src.agent.response import (
    STATUT_ERREUR,
    STATUT_REFUS,
    STATUT_SUCCES,
    AgentResponse,
)
from src.rag.ingestion import RapportIngestion
from src.sources.base import ErreurSource
from tests.api.conftest import CapturingSink, EspionAgentService, EspionIngestionService


# --------------------------------------------------------------------------
# Corrélation HTTP
# --------------------------------------------------------------------------
def test_reponse_porte_toujours_les_deux_entetes(client):
    r = client.post("/query", json={"query": "q"})
    assert r.headers["x-request-id"]
    assert len(r.headers["x-execution-id"]) == 32


def test_x_request_id_valide_conserve_x_execution_id_genere(client):
    r = client.post(
        "/query", json={"query": "q"}, headers={"X-Request-Id": "corr-2026.09_A"}
    )
    assert r.headers["x-request-id"] == "corr-2026.09_A"
    assert r.headers["x-execution-id"] != "corr-2026.09_A"
    assert len(r.headers["x-execution-id"]) == 32


def test_x_request_id_invalide_est_remplace(client):
    r = client.post(
        "/query", json={"query": "q"}, headers={"X-Request-Id": "pas valide / ici"}
    )
    assert r.headers["x-request-id"] != "pas valide / ici"
    assert len(r.headers["x-request-id"]) == 32


def test_ids_uniques_entre_requetes(client):
    vus = {
        (
            client.post("/query", json={"query": "q"}).headers["x-request-id"],
            client.get("/health").headers.get("x-execution-id"),
        )
        for _ in range(10)
    }
    assert len(vus) == 10


def test_health_inchange_mais_correle(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    assert r.headers["x-request-id"]
    assert r.headers["x-execution-id"]


# --------------------------------------------------------------------------
# Émission d'événements de bout en bout (threadpool sync → contextvars)
# --------------------------------------------------------------------------
def test_query_succes_emet_started_puis_completed_correles(build_client):
    sink = CapturingSink()
    espion = EspionAgentService(
        AgentResponse(
            status=STATUT_SUCCES,
            capability="search",
            answer="r",
            citations=["S1"],
            sources=[{"document": "a.pdf", "citation": "S1"}],
        )
    )
    client = build_client(agent_service=espion, sink=sink)

    r = client.post("/query", json={"query": "q"}, headers={"X-Request-Id": "abc"})

    noms = [e.event for e in sink.events]
    assert noms == ["agent_execution_started", "agent_execution_completed"]
    completed = sink.par_nom("agent_execution_completed")[0]
    # l'id posé par le middleware (event loop) est vu dans le threadpool sync
    assert completed.request_id == "abc" == r.headers["x-request-id"]
    assert completed.operation_id == r.headers["x-execution-id"]
    assert completed.outcome == "success"
    assert completed.attributes.capability == "search"
    assert completed.attributes.citations == ("S1",)
    assert completed.duration_ms is not None


def test_query_refus_emet_outcome_refusal(build_client):
    sink = CapturingSink()
    refus = AgentResponse(
        status=STATUT_REFUS,
        capability="extract",
        answer="Précisez.",
        error={"code": "perimetre_non_resolu", "message": "Précisez."},
    )
    client = build_client(agent_service=EspionAgentService(refus), sink=sink)

    r = client.post("/query", json={"query": "q"})

    assert r.status_code == 200  # contrat P2.3 : un refus n'est pas une panne
    completed = sink.par_nom("agent_execution_completed")[0]
    assert completed.outcome == "refusal"
    assert completed.attributes.refusal_code == "perimetre_non_resolu"


def test_query_erreur_technique_emet_failed_et_masque_le_500(build_client):
    sink = CapturingSink()
    fuite = AgentResponse(
        status=STATUT_ERREUR,
        capability="",
        error={"code": "ConnectionError", "message": "qdrant /home/j/db token=s3cr3t"},
    )
    client = build_client(agent_service=EspionAgentService(fuite), sink=sink)

    r = client.post("/query", json={"query": "q"})

    assert r.status_code == 500
    corps = r.json()
    assert corps["status"] == "error"
    assert corps["error"] == {
        "code": "internal_error",
        "message": "Une erreur interne est survenue.",
    }
    assert "s3cr3t" not in json.dumps(corps)
    failed = sink.par_nom("agent_execution_failed")[0]
    assert failed.outcome == "error"
    assert failed.attributes.error_code == "ConnectionError"


def test_query_exception_inattendue_emet_failed_et_http_unhandled(build_client):
    sink = CapturingSink()
    espion = EspionAgentService(exception=RuntimeError("secret /opt/app/key token=z"))
    client = build_client(agent_service=espion, sink=sink)

    r = client.post("/query", json={"query": "q"}, headers={"X-Request-Id": "trace-1"})

    assert r.status_code == 500
    assert r.json() == {"detail": "Erreur interne."}
    noms = [e.event for e in sink.events]
    assert "agent_execution_failed" in noms
    assert "http_unhandled_error" in noms
    http_evt = sink.par_nom("http_unhandled_error")[0]
    assert http_evt.request_id == "trace-1"
    assert http_evt.operation_id == r.headers["x-execution-id"]
    # aucune donnée sensible dans la sérialisation
    from src.observability import evenement_vers_log

    charge = json.dumps(evenement_vers_log(http_evt))
    for interdit in ("/opt/app/key", "token=z", "s3cr3t"):
        assert interdit not in charge


def test_ingestion_partielle_emet_outcome_partial(build_client):
    sink = CapturingSink()
    rapport = RapportIngestion(
        profil="generic",
        fichiers_trouves=10,
        fichiers_traites=7,
        fichiers_en_echec=3,
        chunks_indexes=40,
    )
    client = build_client(ingestion_service=EspionIngestionService(rapport), sink=sink)

    r = client.post("/ingestion", json={})

    assert r.status_code == 200
    completed = sink.par_nom("ingestion_completed")[0]
    assert completed.outcome == "partial"
    assert completed.attributes.files_failed == 3
    assert completed.attributes.chunks_indexed == 40


def test_ingestion_erreur_source_emet_failed_et_reste_503(build_client):
    sink = CapturingSink()
    espion = EspionIngestionService(
        exception=ErreurSource("montage /home/j/corpus indisponible")
    )
    client = build_client(ingestion_service=espion, sink=sink)

    r = client.post("/ingestion", json={})

    assert r.status_code == 503
    assert r.json() == {"detail": "Source documentaire temporairement indisponible."}
    failed = sink.par_nom("ingestion_failed")[0]
    assert failed.outcome == "error"
    assert failed.attributes.error_category == "source_unavailable"


# --------------------------------------------------------------------------
# Contrat P2.3 strictement préservé
# --------------------------------------------------------------------------
def test_body_query_strictement_identique_a_p2_3(build_client):
    riche = AgentResponse(
        status=STATUT_SUCCES,
        capability="compare",
        answer="Conclusion transverse.",
        sources=[
            {
                "citation": "D1S1",
                "document": "a.pdf",
                "page": 2,
                "categorie": "rapport",
                "extrait": "…",
                "hors_perimetre": False,
            }
        ],
        citations=["D1S1", "D2S1"],
        warnings=["divergence"],
        data={"comparaison": {"documents": ["a.pdf", "b.pdf"], "conclusion": "…"}},
        metadata={"documents_resolus": ["a.pdf", "b.pdf"], "nombre_sources": 1},
        error=None,
    )
    client = build_client(agent_service=EspionAgentService(riche), sink=CapturingSink())

    r = client.post("/query", json={"query": "compare a et b"})

    assert r.status_code == 200
    assert r.json() == riche.vers_dict()


def test_openapi_toujours_trois_routes_seulement(build_client):
    app = build_client().app
    assert set(app.openapi()["paths"]) == {"/health", "/query", "/ingestion"}
    schemas = app.openapi().get("components", {}).get("schemas", {})
    assert "AgentResponse" not in schemas
    assert "ObservabilityEvent" not in schemas


def test_deux_create_app_sinks_distincts_sans_contamination(build_client):
    sink_a, sink_b = CapturingSink(), CapturingSink()
    client_a = build_client(
        agent_service=EspionAgentService(
            AgentResponse(status=STATUT_SUCCES, capability="search", answer="a")
        ),
        sink=sink_a,
    )
    client_b = build_client(
        agent_service=EspionAgentService(
            AgentResponse(status=STATUT_SUCCES, capability="summarize", answer="b")
        ),
        sink=sink_b,
    )

    client_a.post("/query", json={"query": "qa"})

    assert [e.event for e in sink_a.events] == [
        "agent_execution_started",
        "agent_execution_completed",
    ]
    assert sink_b.events == []

    client_b.post("/query", json={"query": "qb"})
    assert len(sink_b.events) == 2
    assert sink_b.par_nom("agent_execution_completed")[0].attributes.capability == "summarize"


def test_concurrence_de_requetes_ne_melange_pas_les_correlations(build_client):
    """Plusieurs requêtes concurrentes via httpx/ASGI : chaque événement
    complété porte l'id de sa propre requête, sans fuite entre requêtes."""
    import anyio
    import httpx

    sink = CapturingSink()
    client = build_client(
        agent_service=EspionAgentService(
            AgentResponse(status=STATUT_SUCCES, capability="search", answer="ok")
        ),
        sink=sink,
    )
    app = client.app
    resultats: dict[int, httpx.Response] = {}

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
            async def une(i: int) -> None:
                resultats[i] = await ac.post(
                    "/query",
                    json={"query": f"q{i}"},
                    headers={"X-Request-Id": f"r-{i}"},
                )

            async with anyio.create_task_group() as tg:
                for i in range(20):
                    tg.start_soon(une, i)

    anyio.run(scenario)

    assert len(resultats) == 20
    for i, resp in resultats.items():
        assert resp.headers["x-request-id"] == f"r-{i}"
        assert len(resp.headers["x-execution-id"]) == 32
    completes = {
        e.request_id: e
        for e in sink.events
        if e.event == "agent_execution_completed"
    }
    for i in range(20):
        assert f"r-{i}" in completes
        assert completes[f"r-{i}"].outcome == "success"
