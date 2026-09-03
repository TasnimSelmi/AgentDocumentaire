"""
Destinations de traces — `TraceSink` (Protocol) et deux implémentations.

- `NullTraceSink`    : ne fait rien. Défaut sûr quand l'observabilité est
                       désactivée.
- `LoggingTraceSink` : **seul** endroit du paquet autorisé à dépendre de
                       `structlog`. Émet une ligne JSON structurée sur un flux
                       (stdout par défaut).

Aucun `get_sink()` / `set_sink()`, aucun sink global mutable : le sink est
**injecté** explicitement dans les wrappers (`install_observability`). Deux
`create_app()` dans le même process peuvent donc porter des sinks différents
sans interférence.

`LoggingTraceSink` construit un logger `structlog` **local** via
`structlog.wrap_logger` : il n'appelle jamais `structlog.configure()` et ne
touche pas au root logger `logging`. Sa construction est idempotente (aucun
état de module partagé). Si `emit()` échoue, l'exception est absorbée — une
requête métier ne casse jamais à cause d'une trace.
"""

from __future__ import annotations

import sys
from typing import Any, Protocol, TextIO, runtime_checkable

from src.observability.events import ObservabilityEvent
from src.observability.redaction import evenement_vers_log


@runtime_checkable
class TraceSink(Protocol):
    """Contrat minimal d'une destination de traces."""

    def emit(self, event: ObservabilityEvent) -> None:
        ...


class NullTraceSink:
    """Sink inerte — utile quand l'observabilité est désactivée ou en test."""

    def emit(self, event: ObservabilityEvent) -> None:  # noqa: D401 - no-op
        return None


class LoggingTraceSink:
    """
    Sink JSON structuré sur `stream` (par défaut `sys.stdout`).

    `stream` est capturé à la construction : en test, instancier le sink
    **après** la mise en place de la capture (ou passer un `io.StringIO`).
    """

    def __init__(self, *, stream: TextIO | None = None) -> None:
        import structlog

        self._stream: TextIO = stream if stream is not None else sys.stdout
        self._log = structlog.wrap_logger(
            structlog.PrintLogger(file=self._stream),
            processors=[
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True, key="logged_at"),
                structlog.processors.JSONRenderer(sort_keys=True),
            ],
            wrapper_class=structlog.BoundLogger,
        )

    def emit(self, event: ObservabilityEvent) -> None:
        try:
            charge: dict[str, Any] = evenement_vers_log(event)
            nom = charge.pop("event", "observability_event")
            self._log.info(nom, **charge)
        except Exception:  # noqa: BLE001 — une trace ne casse jamais le métier
            return None


__all__ = ["TraceSink", "NullTraceSink", "LoggingTraceSink"]
