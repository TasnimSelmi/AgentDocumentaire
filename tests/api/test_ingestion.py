"""
`POST /ingestion` — transport pur au-dessus de `IngestionService.sync`.

Vérifie : résolution de la source par **nom logique** (jamais un chemin),
transmission des options, fidélité du `RapportIngestion`, et le mapping HTTP :
succès → 200 (même avec des échecs par fichier), `ErreurSource` → 503,
exception inattendue → 500, corps/source invalide → 422.
"""

from __future__ import annotations

import json

import pytest

from src.rag.ingestion import RapportIngestion
from src.sources.base import ErreurSource
from tests.api.conftest import EspionIngestionService


def test_ingestion_reussie_renvoie_le_rapport_intact(build_client, source_marqueur):
    rapport = RapportIngestion(
        profil="generic", fichiers_trouves=4, fichiers_traites=4, chunks_indexes=87
    )
    espion = EspionIngestionService(rapport)
    client = build_client(ingestion_service=espion)

    reponse = client.post("/ingestion", json={})

    assert reponse.status_code == 200
    from dataclasses import asdict

    assert reponse.json() == asdict(rapport)
    assert len(espion.appels) == 1
    assert espion.appels[0]["source"] is source_marqueur


def test_ingestion_source_par_defaut_est_local(build_client, source_marqueur):
    espion = EspionIngestionService()
    build_client(ingestion_service=espion).post("/ingestion", json={})
    assert espion.appels[0]["source"] is source_marqueur


def test_ingestion_transmet_les_options_exposees(build_client):
    espion = EspionIngestionService()
    build_client(ingestion_service=espion).post(
        "/ingestion", json={"reinitialiser": True, "limite": 10}
    )
    appel = espion.appels[0]
    assert appel["reinitialiser"] is True
    assert appel["limite"] == 10


def test_ingestion_n_expose_ni_inferer_ni_nom_profil(build_client):
    espion = EspionIngestionService()
    build_client(ingestion_service=espion).post(
        "/ingestion", json={"inferer": False, "nom_profil": "x"}
    )
    # 422 : clés interdites (extra="forbid"). Le service n'est pas appelé.
    assert espion.appels == []


@pytest.mark.parametrize(
    "corps",
    [
        {"path": "/etc/passwd"},
        {"dossier": "/var/data"},
        {"racine": "../../secret"},
        {"source": "local", "path": "/tmp"},
    ],
)
def test_ingestion_refuse_tout_chemin_du_client(build_client, corps):
    espion = EspionIngestionService()
    reponse = build_client(ingestion_service=espion).post("/ingestion", json=corps)
    assert reponse.status_code == 422
    assert espion.appels == []


def test_ingestion_source_inconnue_est_422(build_client):
    espion = EspionIngestionService()
    reponse = build_client(ingestion_service=espion).post(
        "/ingestion", json={"source": "s3-entreprise"}
    )
    assert reponse.status_code == 422
    assert espion.appels == []


def test_ingestion_erreur_source_est_503_sans_fuite(build_client):
    espion = EspionIngestionService(
        exception=ErreurSource("Répertoire source introuvable : /home/jawher/corpus")
    )
    reponse = build_client(ingestion_service=espion).post("/ingestion", json={})
    assert reponse.status_code == 503
    corps = reponse.json()
    assert corps == {"detail": "Source documentaire temporairement indisponible."}
    assert "/home/jawher" not in json.dumps(corps)


def test_ingestion_erreur_source_depuis_la_fabrique_est_503(build_client):
    def fabrique_qui_leve():
        raise ErreurSource("montage distant indisponible : //nas/share")

    reponse = build_client(sources={"local": fabrique_qui_leve}).post(
        "/ingestion", json={}
    )
    assert reponse.status_code == 503


def test_ingestion_exception_inattendue_est_500_sans_fuite(build_client):
    espion = EspionIngestionService(
        exception=RuntimeError("qdrant timeout 10.0.0.5:6333 apikey=zzz")
    )
    reponse = build_client(ingestion_service=espion).post("/ingestion", json={})
    assert reponse.status_code == 500
    corps = reponse.json()
    assert corps == {"detail": "Erreur interne."}
    assert "10.0.0.5" not in json.dumps(corps)


def test_ingestion_echecs_par_fichier_restent_200(build_client):
    rapport = RapportIngestion(
        profil="generic",
        fichiers_trouves=10,
        fichiers_traites=7,
        fichiers_en_echec=3,
        chunks_indexes=40,
    )
    reponse = build_client(ingestion_service=EspionIngestionService(rapport)).post(
        "/ingestion", json={}
    )
    assert reponse.status_code == 200
    assert reponse.json()["fichiers_en_echec"] == 3


def test_ingestion_ne_touche_pas_l_agent(client, agent_service):
    client.post("/ingestion", json={})
    assert agent_service.appels == []
