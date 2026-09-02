"""`GET /health` — liveness, aucun collaborateur sollicité."""

from __future__ import annotations


def test_health_repond_ok(client):
    reponse = client.get("/health")
    assert reponse.status_code == 200
    assert reponse.json() == {"status": "ok"}


def test_health_ne_touche_aucun_service(client, agent_service, ingestion_service):
    client.get("/health")
    assert agent_service.appels == []
    assert ingestion_service.appels == []
