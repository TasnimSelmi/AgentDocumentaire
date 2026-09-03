"""
Couche d'observabilité transverse (P2.4).

Trace les requêtes agent et les ingestions — durée, statut, capacité,
documents / citations, erreurs techniques **sécurisées**, identifiant de
corrélation — **sans** modifier le comportement métier et **sans** toucher aux
couches gelées (`src/rag/**`, `src/agent/**`, `src/tools/**`, `src/sources/**`).
`AgentResponse` est strictement inchangé ; aucune trace interne du graphe
LangGraph n'est exposée.

Chaîne : `HTTP → CorrelationMiddleware → routes → Instrumented*Service →
service existant → cœur gelé`, et en parallèle `Instrumented*Service →
ObservabilityEvent → TraceSink → LoggingTraceSink → JSON stdout`.

Générique et indépendant du corpus. Extensible plus tard vers OpenTelemetry /
ELK / Loki / Splunk / base d'audit en fournissant un autre `TraceSink` —
aucune de ces infrastructures n'est implémentée ici.
"""

from src.observability.correlation import (
    CorrelationIds,
    current_execution_id,
    current_request_id,
    portee_correlation,
    request_id_valide,
)
from src.observability.events import (
    AGENT_EXECUTION_COMPLETED,
    AGENT_EXECUTION_FAILED,
    AGENT_EXECUTION_STARTED,
    EVENEMENTS,
    HTTP_UNHANDLED_ERROR,
    INGESTION_COMPLETED,
    INGESTION_FAILED,
    INGESTION_STARTED,
    OUTCOME_ERROR,
    OUTCOME_PARTIAL,
    OUTCOME_REFUSAL,
    OUTCOME_SUCCESS,
    OUTCOMES,
    SCHEMA_VERSION,
    AgentExecutionAttributes,
    HttpErrorAttributes,
    IngestionAttributes,
    ObservabilityEvent,
)
from src.observability.instrumentation import (
    InstrumentedAgentService,
    InstrumentedIngestionService,
    emettre_http_unhandled_error,
)
from src.observability.integration import ObservabilityRuntime, install_observability
from src.observability.middleware import CorrelationMiddleware
from src.observability.redaction import (
    evenement_vers_log,
    scrub_stack,
    scrub_texte,
)
from src.observability.sinks import LoggingTraceSink, NullTraceSink, TraceSink

__all__ = [
    # Enveloppe & contrat d'événements
    "SCHEMA_VERSION",
    "ObservabilityEvent",
    "AgentExecutionAttributes",
    "IngestionAttributes",
    "HttpErrorAttributes",
    "EVENEMENTS",
    "OUTCOMES",
    "OUTCOME_SUCCESS",
    "OUTCOME_REFUSAL",
    "OUTCOME_PARTIAL",
    "OUTCOME_ERROR",
    "AGENT_EXECUTION_STARTED",
    "AGENT_EXECUTION_COMPLETED",
    "AGENT_EXECUTION_FAILED",
    "INGESTION_STARTED",
    "INGESTION_COMPLETED",
    "INGESTION_FAILED",
    "HTTP_UNHANDLED_ERROR",
    # Corrélation
    "CorrelationIds",
    "current_request_id",
    "current_execution_id",
    "portee_correlation",
    "request_id_valide",
    "CorrelationMiddleware",
    # Sinks
    "TraceSink",
    "NullTraceSink",
    "LoggingTraceSink",
    # Redaction
    "evenement_vers_log",
    "scrub_texte",
    "scrub_stack",
    # Instrumentation & intégration
    "InstrumentedAgentService",
    "InstrumentedIngestionService",
    "emettre_http_unhandled_error",
    "install_observability",
    "ObservabilityRuntime",
]
