"""
Propagation de `corpus_id` à travers la couche API (§9/§10 de l'audit
multi-corpus).

Aucun cœur agentique réel, aucun Qdrant : les espions `EspionAgentService` /
`EspionIngestionService` (déjà utilisés par `test_query.py`/`test_ingestion.py`)
suffisent à prouver que `corpus_id` est bien lu du corps de requête et transmis
tel quel — jamais un nom de collection ni un chemin — et que l'absence de
`corpus_id` reste rétrocompatible (repli sur "default").
"""

from __future__ import annotations

from src.agent.service import AgentService
from src.rag.corpus import CorpusInconnu, ErreurCorpusInvalide


def test_query_sans_corpus_id_utilise_default_par_defaut(build_client, agent_service):
    build_client(agent_service=agent_service).post("/query", json={"query": "bonjour"})

    assert agent_service.appels == ["bonjour"]
    assert agent_service.corpus_ids == ["default"]


def test_query_transmet_le_corpus_id_fourni(build_client, agent_service):
    build_client(agent_service=agent_service).post(
        "/query", json={"query": "bonjour", "corpus_id": "finance"}
    )

    assert agent_service.corpus_ids == ["finance"]


def test_query_corpus_id_invalide_est_422(build_client):
    def coeur_leve_corpus_invalide(*_a, **_k):
        raise ErreurCorpusInvalide("corpus_id invalide : '../x'.")

    service = AgentService(point_entree=coeur_leve_corpus_invalide)
    reponse = build_client(agent_service=service).post(
        "/query", json={"query": "bonjour", "corpus_id": "../x"}
    )

    assert reponse.status_code == 422
    assert reponse.json()["error"]["code"] == "corpus_invalide"


def test_query_corpus_id_inconnu_est_422(build_client):
    def coeur_leve_corpus_inconnu(*_a, **_k):
        raise CorpusInconnu("Corpus inconnu : 'rh'.")

    service = AgentService(point_entree=coeur_leve_corpus_inconnu)
    reponse = build_client(agent_service=service).post(
        "/query", json={"query": "bonjour", "corpus_id": "rh"}
    )

    assert reponse.status_code == 422
    assert reponse.json()["error"]["code"] == "corpus_invalide"


def test_query_aucun_chemin_ni_collection_arbitraire_accepte(build_client):
    """`extra="forbid"` sur `QueryRequest` : un client ne peut jamais glisser
    un nom de collection, un `qdrant_path` ou un chemin dans le corps."""
    reponse = build_client().post(
        "/query",
        json={"query": "bonjour", "collection": "autre_collection", "qdrant_path": "/tmp/x"},
    )
    assert reponse.status_code == 422


def test_ingestion_sans_corpus_id_utilise_default_par_defaut(build_client, ingestion_service):
    build_client(ingestion_service=ingestion_service).post("/ingestion", json={"source": "local"})

    assert ingestion_service.appels[0]["corpus_id"] == "default"


def test_ingestion_transmet_le_corpus_id_fourni(build_client, ingestion_service, monkeypatch):
    from src.config import ConfigCorpus
    from tests.rag.conftest import cabler_registre_corpus

    cabler_registre_corpus(monkeypatch, {"default": ConfigCorpus(), "finance": ConfigCorpus()})

    build_client(ingestion_service=ingestion_service).post(
        "/ingestion", json={"source": "local", "corpus_id": "finance"}
    )

    assert ingestion_service.appels[0]["corpus_id"] == "finance"


def test_ingestion_corpus_id_invalide_est_422_sans_appeler_le_service(
    build_client, ingestion_service
):
    reponse = build_client(ingestion_service=ingestion_service).post(
        "/ingestion", json={"source": "local", "corpus_id": "Invalide Avec Espaces"}
    )

    assert reponse.status_code == 422
    assert ingestion_service.appels == []


def test_ingestion_aucune_collection_ni_chemin_arbitraire_accepte(build_client):
    reponse = build_client().post(
        "/ingestion",
        json={"source": "local", "collection": "autre", "path": "/tmp/x"},
    )
    assert reponse.status_code == 422
