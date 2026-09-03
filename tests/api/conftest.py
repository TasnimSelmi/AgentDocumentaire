"""
Doublures et fabrique de client pour les tests de la couche API.

Tout est **hors ligne** : `AgentService` et `IngestionService` sont remplacés
par des espions qui enregistrent leurs appels. Aucun Ollama, aucun Qdrant,
aucun graphe LangGraph exécuté, aucun accès au système de fichiers.
`TestClient(raise_server_exceptions=False)` : on observe le statut HTTP réel,
comme un vrai client.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.agent.response import STATUT_SUCCES, AgentResponse
from src.api import create_app
from src.observability import NullTraceSink, ObservabilityEvent
from src.rag.ingestion import RapportIngestion


class CapturingSink:
    """`TraceSink` de test : conserve les événements émis, en mémoire."""

    def __init__(self) -> None:
        self.events: list[ObservabilityEvent] = []

    def emit(self, event: ObservabilityEvent) -> None:
        self.events.append(event)

    def par_nom(self, nom: str) -> list[ObservabilityEvent]:
        return [e for e in self.events if e.event == nom]


class EspionAgentService:
    """Faux `AgentService` : compte les appels, renvoie une réponse scriptée
    ou lève une exception scriptée."""

    def __init__(
        self,
        reponse: AgentResponse | None = None,
        *,
        exception: BaseException | None = None,
    ) -> None:
        self.reponse = reponse or AgentResponse(
            status=STATUT_SUCCES, capability="search", answer="ok"
        )
        self.exception = exception
        self.appels: list[str] = []

    def query(self, requete: str) -> AgentResponse:
        self.appels.append(requete)
        if self.exception is not None:
            raise self.exception
        return self.reponse


class EspionIngestionService:
    """Faux `IngestionService` : mémorise `source` + options, renvoie un
    rapport scripté ou lève une exception scriptée."""

    def __init__(
        self,
        rapport: RapportIngestion | None = None,
        *,
        exception: BaseException | None = None,
    ) -> None:
        self.rapport = rapport or RapportIngestion(
            profil="espion", fichiers_trouves=3, fichiers_traites=3, chunks_indexes=12
        )
        self.exception = exception
        self.appels: list[dict[str, Any]] = []

    def sync(
        self,
        source: Any,
        *,
        reinitialiser: bool = False,
        limite: int | None = None,
        inferer: bool = True,
        nom_profil: str | None = None,
    ) -> RapportIngestion:
        self.appels.append(
            {
                "source": source,
                "reinitialiser": reinitialiser,
                "limite": limite,
                "inferer": inferer,
                "nom_profil": nom_profil,
            }
        )
        if self.exception is not None:
            raise self.exception
        return self.rapport


class SourceMarqueur:
    """`DocumentSource` factice : jamais matérialisée par les espions ci-dessus
    (l'espion n'entre pas dans `materialiser()`)."""

    def __init__(self, nom: str = "marqueur") -> None:
        self.nom = nom


@pytest.fixture(autouse=True)
def _qdrant_path_local(tmp_path, monkeypatch):
    """Sur cette machine, `.env` pointe `QDRANT_PATH` vers un chemin absolu non
    inscriptible et `get_settings()` tente de créer les dossiers du projet. On
    le rabat sur un tmp local (aucun Qdrant n'est ouvert par ces tests). Même
    garde que `tests/rag/test_resolution_identifiant_exact.py`."""
    import src.config as config

    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "vectordb"))
    config.get_settings.cache_clear()
    config.get_config_technique.cache_clear()
    yield
    config.get_settings.cache_clear()
    config.get_config_technique.cache_clear()


@pytest.fixture
def agent_service() -> EspionAgentService:
    return EspionAgentService()


@pytest.fixture
def ingestion_service() -> EspionIngestionService:
    return EspionIngestionService()


@pytest.fixture
def source_marqueur() -> SourceMarqueur:
    return SourceMarqueur()


@pytest.fixture
def sources(source_marqueur: SourceMarqueur) -> dict[str, Any]:
    """Registre à une entrée : `"local"` → une fabrique qui rend toujours la
    même instance marqueur (traçable côté test)."""
    return {"local": lambda: source_marqueur}


@pytest.fixture
def capturing_sink() -> CapturingSink:
    return CapturingSink()


@pytest.fixture
def build_client(agent_service, ingestion_service, sources):
    """Fabrique de `TestClient`. Sans argument : utilise les espions des
    fixtures. Chaque argument nommé (`agent_service=`, `ingestion_service=`,
    `sources=`, `sink=`) remplace la doublure correspondante.

    Le sink par défaut est `NullTraceSink` : les tests P2.3 n'observent pas de
    trace et rien n'est écrit sur stdout. Passer `sink=CapturingSink()` pour
    inspecter les événements."""

    def _build(**overrides: Any) -> TestClient:
        app = create_app(
            agent_service=overrides.get("agent_service", agent_service),
            ingestion_service=overrides.get("ingestion_service", ingestion_service),
            sources=overrides.get("sources", sources),
            sink=overrides.get("sink", NullTraceSink()),
        )
        return TestClient(app, raise_server_exceptions=False)

    return _build


@pytest.fixture
def client(build_client) -> TestClient:
    return build_client()
