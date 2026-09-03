"""
Modèle d'événements de l'observabilité — enveloppe **versionnée et fermée**.

Aucun `dict[str, Any]` arbitraire : `ObservabilityEvent.attributes` ne peut être
qu'une des dataclasses typées de ce module (ou `None`). Ajouter un champ à une
capacité = éditer la dataclass correspondante **et** l'allow-list de
`src/observability/redaction.py` — rien ne fuit tant que les deux ne sont pas
alignés.

Ce module est **pur** : stdlib uniquement, aucune dépendance à `structlog`,
FastAPI, `src/agent/**`, `src/rag/**`. Il ne lit ni n'écrit aucun état global.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Union

#: Version du contrat d'événement. Incrémentée à toute rupture de forme de
#: l'enveloppe ou des blocs `attributes` (ajout compatible = pas de bump).
SCHEMA_VERSION = "1.0"

# --- Noms d'événements stables ------------------------------------------------
AGENT_EXECUTION_STARTED = "agent_execution_started"
AGENT_EXECUTION_COMPLETED = "agent_execution_completed"
AGENT_EXECUTION_FAILED = "agent_execution_failed"
INGESTION_STARTED = "ingestion_started"
INGESTION_COMPLETED = "ingestion_completed"
INGESTION_FAILED = "ingestion_failed"
HTTP_UNHANDLED_ERROR = "http_unhandled_error"

EVENEMENTS = (
    AGENT_EXECUTION_STARTED,
    AGENT_EXECUTION_COMPLETED,
    AGENT_EXECUTION_FAILED,
    INGESTION_STARTED,
    INGESTION_COMPLETED,
    INGESTION_FAILED,
    HTTP_UNHANDLED_ERROR,
)

# --- Issues (`outcome`) ----------------------------------------------------
OUTCOME_SUCCESS = "success"
OUTCOME_REFUSAL = "refusal"
OUTCOME_PARTIAL = "partial"
OUTCOME_ERROR = "error"
OUTCOMES = (OUTCOME_SUCCESS, OUTCOME_REFUSAL, OUTCOME_PARTIAL, OUTCOME_ERROR)


@dataclass(frozen=True)
class AgentExecutionAttributes:
    """Bloc `attributes` d'une exécution agent — dérivé **uniquement** du
    contrat public `AgentResponse` (jamais de l'état interne du graphe).

    `documents` / `citations` sont des tuples (enveloppe immuable). `error_*`
    n'est renseigné que pour un `outcome="error"` ; `refusal_code` que pour un
    `outcome="refusal"`. `error_message` / `error_stack` sont **redactés** à la
    sérialisation (voir `redaction.py`)."""

    capability: str | None = None
    documents: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    source_count: int = 0
    refusal_code: str | None = None
    error_category: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    error_stack: str | None = None


@dataclass(frozen=True)
class IngestionAttributes:
    """Bloc `attributes` d'une ingestion — compteurs recopiés depuis
    `RapportIngestion` (miroir 1:1, aucun chemin local). `error_*` uniquement
    pour un `outcome="error"` ; `error_message` / `error_stack` redactés."""

    source: str | None = None
    profile: str | None = None
    files_found: int = 0
    files_processed: int = 0
    files_failed: int = 0
    files_skipped: int = 0
    files_empty: int = 0
    files_ocr: int = 0
    files_deleted: int = 0
    chunks_indexed: int = 0
    error_category: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    error_stack: str | None = None


@dataclass(frozen=True)
class HttpErrorAttributes:
    """Bloc `attributes` d'un `http_unhandled_error` : le gestionnaire HTTP
    unique (`src/api/errors.py`) a rattrapé une exception non prévue. Aucun
    détail métier — seulement la catégorie et le type d'erreur ; `error_message`
    / `error_stack` redactés."""

    error_category: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    error_stack: str | None = None


#: Union fermée des blocs `attributes` autorisés.
EventAttributes = Union[
    AgentExecutionAttributes,
    IngestionAttributes,
    HttpErrorAttributes,
]


@dataclass(frozen=True)
class ObservabilityEvent:
    """
    Enveloppe commune, immuable et fermée, de tout signal d'observabilité.

    Champs :
    - `schema_version` : version du contrat (`"1.0"`).
    - `event`          : un nom de `EVENEMENTS`.
    - `request_id`     : identifiant de corrélation exploitable par le support
                         (`X-Request-Id`). `None` seulement hors contexte connu.
    - `operation_id`   : identifiant d'exécution serveur (`X-Execution-Id`).
                         Stable pour toute la durée d'une opération (started →
                         completed / failed).
    - `started_at` / `finished_at` : timestamps ISO-8601 UTC. `finished_at` est
                         `None` pour un événement `*_started`.
    - `duration_ms`    : durée totale mesurée (`time.perf_counter`), `None` pour
                         un `*_started` ou un `http_unhandled_error`.
    - `outcome`        : une valeur de `OUTCOMES`, `None` pour un `*_started`.
    - `attributes`     : bloc typé propre à l'opération, ou `None`.
    """

    event: str
    operation_id: str
    schema_version: str = SCHEMA_VERSION
    request_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    duration_ms: float | None = None
    outcome: str | None = None
    attributes: EventAttributes | None = None


def noms_de_champs(dc: type) -> tuple[str, ...]:
    """Noms de champs d'une dataclass d'attributs — utilisé par les tests pour
    vérifier que l'allow-list de redaction couvre bien toute la surface."""
    return tuple(f.name for f in fields(dc))


__all__ = [
    "SCHEMA_VERSION",
    "AGENT_EXECUTION_STARTED",
    "AGENT_EXECUTION_COMPLETED",
    "AGENT_EXECUTION_FAILED",
    "INGESTION_STARTED",
    "INGESTION_COMPLETED",
    "INGESTION_FAILED",
    "HTTP_UNHANDLED_ERROR",
    "EVENEMENTS",
    "OUTCOME_SUCCESS",
    "OUTCOME_REFUSAL",
    "OUTCOME_PARTIAL",
    "OUTCOME_ERROR",
    "OUTCOMES",
    "AgentExecutionAttributes",
    "IngestionAttributes",
    "HttpErrorAttributes",
    "EventAttributes",
    "ObservabilityEvent",
    "noms_de_champs",
]
