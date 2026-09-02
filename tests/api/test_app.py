"""
`create_app` — câblage, injection, et garanties de conception P2.3 :
offline à la construction, aucun modèle miroir d'`AgentResponse`, registre de
sources par défaut sans chemin client.
"""

from __future__ import annotations

from pathlib import Path

from src.agent.service import AgentService
from src.api import create_app
from src.api.dependencies import registre_sources_par_defaut
from src.config import get_settings
from src.sources import IngestionService
from src.sources.local import LocalDocumentSource


def test_create_app_sans_arguments_est_offline():
    app = create_app()
    assert isinstance(app.state.agent_service, AgentService)
    assert isinstance(app.state.ingestion_service, IngestionService)
    assert "local" in app.state.sources


def test_openapi_expose_les_trois_routes():
    schema = create_app().openapi()
    assert set(schema["paths"]) == {"/health", "/query", "/ingestion"}


def test_openapi_ne_definit_aucun_modele_miroir_agentresponse():
    schemas = create_app().openapi().get("components", {}).get("schemas", {})
    assert "AgentResponse" not in schemas


def test_injection_des_collaborateurs():
    agent = AgentService()
    ing = IngestionService()
    reg = {"x": lambda: object()}
    app = create_app(agent_service=agent, ingestion_service=ing, sources=reg)
    assert app.state.agent_service is agent
    assert app.state.ingestion_service is ing
    assert app.state.sources is reg


def test_registre_par_defaut_pointe_sur_le_dossier_configure():
    registre = registre_sources_par_defaut()
    assert set(registre) == {"local"}
    source = registre["local"]()
    assert isinstance(source, LocalDocumentSource)
    assert Path(source.racine) == Path(get_settings().documents_dir)
