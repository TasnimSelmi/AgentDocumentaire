"""
`POST /query` — transport pur au-dessus de `AgentService.query`.

Vérifie : délégation exacte (1 appel, argument intact), fidélité du contrat
public `AgentResponse`, et le mapping HTTP :
succès/refus → 200, `requete_invalide` → 422, autre `error` → 500 masqué.
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
from src.agent.service import AgentService
from tests.api.conftest import EspionAgentService


def test_query_valide_delegue_une_seule_fois(build_client):
    espion = EspionAgentService(
        AgentResponse(status=STATUT_SUCCES, capability="compare", answer="Conclusion.")
    )
    client = build_client(agent_service=espion)

    reponse = client.post("/query", json={"query": "Compare A et B"})

    assert reponse.status_code == 200
    assert espion.appels == ["Compare A et B"]
    assert reponse.json() == espion.reponse.vers_dict()


def test_query_ne_touche_pas_l_ingestion(client, ingestion_service):
    client.post("/query", json={"query": "coucou"})
    assert ingestion_service.appels == []


def test_query_succes_est_200(build_client):
    espion = EspionAgentService(
        AgentResponse(status=STATUT_SUCCES, capability="search", answer="Réponse.")
    )
    reponse = build_client(agent_service=espion).post("/query", json={"query": "q"})
    assert reponse.status_code == 200
    assert reponse.json()["status"] == "success"


def test_query_refus_metier_est_200_et_intact(build_client):
    refus = AgentResponse(
        status=STATUT_REFUS,
        capability="extract",
        answer="Précisez le document.",
        error={"code": "perimetre_non_resolu", "message": "Précisez le document."},
    )
    reponse = build_client(agent_service=EspionAgentService(refus)).post(
        "/query", json={"query": "extrais les montants"}
    )
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["status"] == "refusal"
    assert corps["error"] == {
        "code": "perimetre_non_resolu",
        "message": "Précisez le document.",
    }


def test_query_erreur_technique_est_500_sans_fuite(build_client):
    fuite = AgentResponse(
        status=STATUT_ERREUR,
        capability="",
        error={
            "code": "ConnectionError",
            "message": "qdrant refused at /home/jawher/data/vectordb (token=s3cr3t)",
        },
    )
    reponse = build_client(agent_service=EspionAgentService(fuite)).post(
        "/query", json={"query": "q"}
    )
    assert reponse.status_code == 500
    corps = reponse.json()
    assert corps["status"] == "error"
    assert corps["error"] == {
        "code": "internal_error",
        "message": "Une erreur interne est survenue.",
    }
    brut = json.dumps(corps)
    for sensible in ("ConnectionError", "/home/jawher", "vectordb", "s3cr3t"):
        assert sensible not in brut


def test_query_exception_du_service_est_500_sans_fuite(build_client):
    # `AgentService` réel ne lève jamais, mais on blinde la route : si une
    # doublure lève, le gestionnaire générique masque le détail.
    espion = EspionAgentService(exception=RuntimeError("secret path /opt/app/key"))
    reponse = build_client(agent_service=espion).post("/query", json={"query": "q"})
    assert reponse.status_code == 500
    assert reponse.json() == {"detail": "Erreur interne."}


@pytest.mark.parametrize("vide", ["", "   ", "\n\t"])
def test_query_vide_ou_blanc_est_422_via_agent_service(build_client, vide):
    """`AgentService` reste l'autorité unique : entrée vide → `error /
    requete_invalide` → 422, **sans** appeler le cœur P1."""

    def coeur_interdit(*_a, **_k):
        raise AssertionError("le cœur P1 ne doit pas être appelé pour une entrée vide")

    client = build_client(agent_service=AgentService(point_entree=coeur_interdit))
    reponse = client.post("/query", json={"query": vide})

    assert reponse.status_code == 422
    corps = reponse.json()
    assert corps["status"] == "error"
    assert corps["error"]["code"] == "requete_invalide"


def test_query_corps_sans_champ_query_est_422(client, agent_service):
    reponse = client.post("/query", json={})
    assert reponse.status_code == 422
    assert agent_service.appels == []


def test_query_query_mauvais_type_est_422(client):
    assert client.post("/query", json={"query": 123}).status_code == 422


def test_query_cle_inconnue_est_422(client, agent_service):
    reponse = client.post("/query", json={"query": "q", "options": {"k": 1}})
    assert reponse.status_code == 422
    assert agent_service.appels == []


def test_query_serialisation_fidele_du_contrat(build_client):
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
        warnings=["divergence sur la date"],
        data={"comparaison": {"documents": ["a.pdf", "b.pdf"], "conclusion": "…"}},
        metadata={"documents_resolus": ["a.pdf", "b.pdf"], "nombre_sources": 1},
        error=None,
    )
    reponse = build_client(agent_service=EspionAgentService(riche)).post(
        "/query", json={"query": "compare a et b"}
    )
    assert reponse.status_code == 200
    assert reponse.json() == riche.vers_dict()
