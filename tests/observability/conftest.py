"""
Doublures et fixtures des tests d'observabilité — 100 % hors ligne.

Aucun Ollama, aucun Qdrant, aucun graphe. Les services applicatifs sont
remplacés par des espions ; le sink par un `CapturingSink` en mémoire.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from src.observability.events import ObservabilityEvent


class CapturingSink:
    """`TraceSink` de test : conserve en mémoire tous les événements émis."""

    def __init__(self) -> None:
        self.events: list[ObservabilityEvent] = []

    def emit(self, event: ObservabilityEvent) -> None:
        self.events.append(event)

    def noms(self) -> list[str]:
        return [e.event for e in self.events]

    def par_nom(self, nom: str) -> list[ObservabilityEvent]:
        return [e for e in self.events if e.event == nom]

    def unique(self, nom: str) -> ObservabilityEvent:
        trouves = self.par_nom(nom)
        assert len(trouves) == 1, f"attendu 1 « {nom} », trouvé {len(trouves)}"
        return trouves[0]


class SinkQuiExplose:
    """`TraceSink` défaillant : `emit` lève toujours."""

    def __init__(self) -> None:
        self.appels = 0

    def emit(self, event: ObservabilityEvent) -> None:
        self.appels += 1
        raise RuntimeError("sink cassé — ne doit jamais casser le métier")


@dataclass
class FauxAgentResponse:
    """Sous-ensemble duck-typé du contrat `AgentResponse` (P1.3)."""

    status: str = "success"
    capability: str = "search"
    answer: str = "ok"
    sources: list[dict[str, Any]] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None


@dataclass
class FauxRapportIngestion:
    """Sous-ensemble duck-typé de `RapportIngestion` (socle RAG)."""

    profil: str = "generic"
    debut: str = ""
    duree_secondes: float = 0.0
    fichiers_trouves: int = 0
    fichiers_ignores_inchanges: int = 0
    fichiers_traites: int = 0
    fichiers_en_echec: int = 0
    fichiers_vides: int = 0
    fichiers_ocr: int = 0
    fichiers_supprimes: int = 0
    chunks_indexes: int = 0


class EspionAgentInner:
    """Faux `AgentService` : compte les appels, renvoie/lève ce qui est scripté."""

    def __init__(
        self,
        reponse: Any | None = None,
        *,
        exception: BaseException | None = None,
    ) -> None:
        self.reponse = reponse if reponse is not None else FauxAgentResponse()
        self.exception = exception
        self.appels: list[str] = []

    def query(self, requete: str) -> Any:
        self.appels.append(requete)
        if self.exception is not None:
            raise self.exception
        return self.reponse


class EspionIngestionInner:
    """Faux `IngestionService` : mémorise les appels, renvoie/lève ce qui est
    scripté."""

    def __init__(
        self,
        rapport: Any | None = None,
        *,
        exception: BaseException | None = None,
    ) -> None:
        self.rapport = rapport if rapport is not None else FauxRapportIngestion()
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
    ) -> Any:
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


class SourceLocaleFactice:
    """`DocumentSource` factice — jamais matérialisée (l'espion n'y entre pas).
    Porte un chemin sensible pour prouver qu'il ne fuit jamais dans une trace."""

    racine = "/home/secret/corpus/interne"

    def __str__(self) -> str:  # pragma: no cover - piège anti-fuite
        return self.racine


@pytest.fixture
def sink() -> CapturingSink:
    return CapturingSink()


@pytest.fixture
def source_factice() -> SourceLocaleFactice:
    return SourceLocaleFactice()
