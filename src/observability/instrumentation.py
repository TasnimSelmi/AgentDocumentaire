"""
Wrappers d'instrumentation — `InstrumentedAgentService` / `InstrumentedIngestionService`.

Ils **enveloppent** les services applicatifs existants sans modifier leur
comportement métier :

- même signature publique (`query(str)` / `sync(source, *, ...)`) ;
- `inner` est appelé **exactement une fois** ;
- l'objet métier original (`AgentResponse` / `RapportIngestion`) est renvoyé
  **tel quel** — l'observabilité ne le touche jamais ;
- `duration_ms` est mesuré avec `time.perf_counter()` ; les timestamps sont
  en UTC ;
- un événement `*_started` est émis si `emit_start=True`, puis un `*_completed`
  ou `*_failed` ;
- une émission de trace qui échoue n'interrompt jamais l'appel métier.

Hors HTTP, `portee_correlation()` fabrique un contexte de corrélation complet
pour toute la durée de l'appel, puis le réinitialise.
"""

from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone
from typing import Any, Protocol

from src.observability.correlation import (
    CorrelationIds,
    current_execution_id,
    current_request_id,
    nouvel_id,
    portee_correlation,
)
from src.observability.events import (
    AGENT_EXECUTION_COMPLETED,
    AGENT_EXECUTION_FAILED,
    AGENT_EXECUTION_STARTED,
    HTTP_UNHANDLED_ERROR,
    INGESTION_COMPLETED,
    INGESTION_FAILED,
    INGESTION_STARTED,
    OUTCOME_ERROR,
    OUTCOME_PARTIAL,
    OUTCOME_REFUSAL,
    OUTCOME_SUCCESS,
    AgentExecutionAttributes,
    HttpErrorAttributes,
    IngestionAttributes,
    ObservabilityEvent,
)
from src.observability.sinks import TraceSink

try:  # `src/sources` est gelé : on l'importe (lecture seule), sans le modifier.
    from src.sources.base import ErreurSource as _ErreurSource
except Exception:  # pragma: no cover - défensif si le paquet bouge
    _ErreurSource = ()  # type: ignore[assignment]

# Codes de statut de `AgentResponse` — recopiés en constantes locales pour ne
# créer aucune dépendance d'import vers la couche agentique gelée.
_STATUT_SUCCES = "success"
_STATUT_REFUS = "refusal"
_STATUT_ERREUR = "error"

_ERR_CATEGORIE_AGENT = "agent_error"
_ERR_CATEGORIE_INATTENDUE = "unexpected"
_ERR_CATEGORIE_SOURCE = "source_unavailable"
_ERR_CATEGORIE_HTTP = "http_unhandled"


# ---------------------------------------------------------------------------
# Protocoles des services enveloppés (structurels — aucun import dur).
# ---------------------------------------------------------------------------
class _AgentServiceLike(Protocol):
    def query(self, requete: str) -> Any: ...


class _IngestionServiceLike(Protocol):
    def sync(
        self,
        source: Any,
        *,
        reinitialiser: bool = ...,
        limite: int | None = ...,
        inferer: bool = ...,
        nom_profil: str | None = ...,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Utilitaires temps / émission.
# ---------------------------------------------------------------------------
def _maintenant() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.isoformat()


def _emettre(sink: TraceSink, event: ObservabilityEvent) -> None:
    """Émission défensive : une trace ne casse jamais l'appel métier."""
    try:
        sink.emit(event)
    except Exception:  # noqa: BLE001
        return None


def _unique(valeurs: Any) -> list[str]:
    vus: set[str] = set()
    sortie: list[str] = []
    for valeur in valeurs or ():
        texte = str(valeur) if valeur is not None else ""
        if texte and texte not in vus:
            vus.add(texte)
            sortie.append(texte)
    return sortie


def _format_stack(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


# ---------------------------------------------------------------------------
# Dérivation des attributs.
# ---------------------------------------------------------------------------
def _attributs_agent(reponse: Any) -> AgentExecutionAttributes:
    statut = getattr(reponse, "status", "") or ""
    erreur = getattr(reponse, "error", None) or {}
    sources = list(getattr(reponse, "sources", None) or [])
    metadata = getattr(reponse, "metadata", None) or {}

    documents = _unique(metadata.get("documents_resolus")) or _unique(
        s.get("document") for s in sources if isinstance(s, dict)
    )
    est_erreur = statut == _STATUT_ERREUR
    est_refus = statut == _STATUT_REFUS

    return AgentExecutionAttributes(
        capability=(getattr(reponse, "capability", "") or None),
        documents=tuple(documents),
        citations=tuple(str(c) for c in (getattr(reponse, "citations", None) or [])),
        source_count=len(sources),
        refusal_code=(erreur.get("code") if est_refus else None),
        error_category=(_ERR_CATEGORIE_AGENT if est_erreur else None),
        error_code=(erreur.get("code") if est_erreur else None),
        error_message=(erreur.get("message") if est_erreur else None),
        error_stack=None,
    )


def _issue_agent(reponse: Any) -> tuple[str, str]:
    """(`event`, `outcome`) pour un `AgentResponse` renvoyé sans exception."""
    statut = getattr(reponse, "status", "") or ""
    if statut == _STATUT_SUCCES:
        return AGENT_EXECUTION_COMPLETED, OUTCOME_SUCCESS
    if statut == _STATUT_REFUS:
        return AGENT_EXECUTION_COMPLETED, OUTCOME_REFUSAL
    return AGENT_EXECUTION_FAILED, OUTCOME_ERROR


def _attributs_ingestion(rapport: Any, source: Any) -> IngestionAttributes:
    return IngestionAttributes(
        source=type(source).__name__,
        profile=(getattr(rapport, "profil", "") or None),
        files_found=int(getattr(rapport, "fichiers_trouves", 0) or 0),
        files_processed=int(getattr(rapport, "fichiers_traites", 0) or 0),
        files_failed=int(getattr(rapport, "fichiers_en_echec", 0) or 0),
        files_skipped=int(getattr(rapport, "fichiers_ignores_inchanges", 0) or 0),
        files_empty=int(getattr(rapport, "fichiers_vides", 0) or 0),
        files_ocr=int(getattr(rapport, "fichiers_ocr", 0) or 0),
        files_deleted=int(getattr(rapport, "fichiers_supprimes", 0) or 0),
        chunks_indexed=int(getattr(rapport, "chunks_indexes", 0) or 0),
    )


def _attributs_ingestion_erreur(
    source: Any, exc: BaseException, categorie: str
) -> IngestionAttributes:
    return IngestionAttributes(
        source=type(source).__name__,
        profile=None,
        error_category=categorie,
        error_code=type(exc).__name__,
        error_message=str(exc),
        error_stack=_format_stack(exc),
    )


# ---------------------------------------------------------------------------
# Wrappers.
# ---------------------------------------------------------------------------
class InstrumentedAgentService:
    """Enveloppe observante de `AgentService` (ou toute façade `query(str)`)."""

    def __init__(
        self,
        inner: _AgentServiceLike,
        sink: TraceSink,
        *,
        emit_start: bool = True,
    ) -> None:
        self._inner = inner
        self._sink = sink
        self._emit_start = emit_start

    @property
    def inner(self) -> _AgentServiceLike:
        """Service enveloppé — introspection (tests, intégration)."""
        return self._inner

    def query(self, requete: str) -> Any:
        with portee_correlation() as ids:
            debut = _maintenant()
            horloge = time.perf_counter()

            if self._emit_start:
                _emettre(self._sink, _event_started(AGENT_EXECUTION_STARTED, ids, debut))

            try:
                reponse = self._inner.query(requete)
            except Exception as exc:  # noqa: BLE001 — on trace puis on relève
                _emettre(
                    self._sink,
                    _event_termine(
                        AGENT_EXECUTION_FAILED,
                        ids,
                        debut,
                        horloge,
                        OUTCOME_ERROR,
                        AgentExecutionAttributes(
                            error_category=_ERR_CATEGORIE_INATTENDUE,
                            error_code=type(exc).__name__,
                            error_message=str(exc),
                            error_stack=_format_stack(exc),
                        ),
                    ),
                )
                raise

            nom_event, issue = _issue_agent(reponse)
            _emettre(
                self._sink,
                _event_termine(
                    nom_event, ids, debut, horloge, issue, _attributs_agent(reponse)
                ),
            )
            return reponse


class InstrumentedIngestionService:
    """Enveloppe observante de `IngestionService` (façade `sync(source, ...)`)."""

    def __init__(
        self,
        inner: _IngestionServiceLike,
        sink: TraceSink,
        *,
        emit_start: bool = True,
    ) -> None:
        self._inner = inner
        self._sink = sink
        self._emit_start = emit_start

    @property
    def inner(self) -> _IngestionServiceLike:
        """Service enveloppé — introspection (tests, intégration)."""
        return self._inner

    def sync(
        self,
        source: Any,
        *,
        reinitialiser: bool = False,
        limite: int | None = None,
        inferer: bool = True,
        nom_profil: str | None = None,
    ) -> Any:
        with portee_correlation() as ids:
            debut = _maintenant()
            horloge = time.perf_counter()

            if self._emit_start:
                _emettre(self._sink, _event_started(INGESTION_STARTED, ids, debut))

            try:
                rapport = self._inner.sync(
                    source,
                    reinitialiser=reinitialiser,
                    limite=limite,
                    inferer=inferer,
                    nom_profil=nom_profil,
                )
            except Exception as exc:  # noqa: BLE001
                est_source = (
                    isinstance(exc, _ErreurSource)
                    if _ErreurSource
                    else type(exc).__name__ == "ErreurSource"
                )
                categorie = (
                    _ERR_CATEGORIE_SOURCE if est_source else _ERR_CATEGORIE_INATTENDUE
                )
                _emettre(
                    self._sink,
                    _event_termine(
                        INGESTION_FAILED,
                        ids,
                        debut,
                        horloge,
                        OUTCOME_ERROR,
                        _attributs_ingestion_erreur(source, exc, categorie),
                    ),
                )
                raise

            echecs = int(getattr(rapport, "fichiers_en_echec", 0) or 0)
            issue = OUTCOME_PARTIAL if echecs > 0 else OUTCOME_SUCCESS
            _emettre(
                self._sink,
                _event_termine(
                    INGESTION_COMPLETED,
                    ids,
                    debut,
                    horloge,
                    issue,
                    _attributs_ingestion(rapport, source),
                ),
            )
            return rapport


# ---------------------------------------------------------------------------
# Fabriques d'événements.
# ---------------------------------------------------------------------------
def _event_started(
    nom: str, ids: CorrelationIds, debut: datetime
) -> ObservabilityEvent:
    return ObservabilityEvent(
        event=nom,
        operation_id=ids.execution_id,
        request_id=ids.request_id,
        started_at=_iso(debut),
        finished_at=None,
        duration_ms=None,
        outcome=None,
        attributes=None,
    )


def _event_termine(
    nom: str,
    ids: CorrelationIds,
    debut: datetime,
    horloge: float,
    outcome: str,
    attributes: Any,
) -> ObservabilityEvent:
    fin = _maintenant()
    duree_ms = (time.perf_counter() - horloge) * 1000.0
    return ObservabilityEvent(
        event=nom,
        operation_id=ids.execution_id,
        request_id=ids.request_id,
        started_at=_iso(debut),
        finished_at=_iso(fin),
        duration_ms=duree_ms,
        outcome=outcome,
        attributes=attributes,
    )


def emettre_http_unhandled_error(
    sink: TraceSink,
    exc: BaseException,
    *,
    request_id: str | None = None,
    operation_id: str | None = None,
) -> None:
    """
    Émet `http_unhandled_error` depuis le gestionnaire d'exception HTTP unique
    (`src/api/errors.py`). Les ids sont fournis par l'appelant (lus sur
    `request.state`, posés par le middleware) ; à défaut, on retombe sur les
    `ContextVar` puis sur un id neuf. Ne lève jamais.
    """
    try:
        moment = _iso(_maintenant())
        event = ObservabilityEvent(
            event=HTTP_UNHANDLED_ERROR,
            operation_id=(
                operation_id or current_execution_id() or request_id or nouvel_id()
            ),
            request_id=(request_id or current_request_id()),
            started_at=moment,
            finished_at=moment,
            duration_ms=None,
            outcome=OUTCOME_ERROR,
            attributes=HttpErrorAttributes(
                error_category=_ERR_CATEGORIE_HTTP,
                error_code=type(exc).__name__,
                error_message=str(exc),
                error_stack=_format_stack(exc),
            ),
        )
        _emettre(sink, event)
    except Exception:  # noqa: BLE001
        return None


__all__ = [
    "InstrumentedAgentService",
    "InstrumentedIngestionService",
    "emettre_http_unhandled_error",
]
