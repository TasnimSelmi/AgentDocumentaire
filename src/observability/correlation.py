"""
Corrélation transverse — deux `ContextVar` (`request_id`, `execution_id`) et un
gestionnaire de contexte pour les appels hors HTTP.

`request_id`   : identifiant fourni par le client (`X-Request-Id`) s'il est
                 valide, sinon généré. Exploitable par le support.
`execution_id` : **toujours** généré côté serveur (`X-Execution-Id`). Unique par
                 exécution.

Ce module ne connaît ni FastAPI, ni l'agent, ni le RAG. Le middleware ASGI
(`middleware.py`) et les wrappers d'instrumentation (`instrumentation.py`) sont
ses seuls appelants. Le reset des `ContextVar` est **toujours** fait dans un
`finally` — aucune fuite d'un contexte de requête vers le suivant, y compris
sous concurrence asyncio (chaque tâche a sa propre copie de contexte) et dans le
threadpool synchrone de Starlette (le contexte est copié vers le worker).
"""

from __future__ import annotations

import re
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Iterator

#: En-têtes de corrélation (minuscules — comparaison ASGI insensible à la casse).
HEADER_REQUEST_ID = "x-request-id"
HEADER_EXECUTION_ID = "x-execution-id"

#: Contrainte d'acceptation d'un `X-Request-Id` entrant.
REQUEST_ID_MAX_LEN = 128
_REQUEST_ID_RE = re.compile(r"\A[A-Za-z0-9._-]{1," + str(REQUEST_ID_MAX_LEN) + r"}\Z")

_request_id: ContextVar[str | None] = ContextVar("observability_request_id", default=None)
_execution_id: ContextVar[str | None] = ContextVar(
    "observability_execution_id", default=None
)


@dataclass(frozen=True)
class CorrelationIds:
    """Paire d'identifiants de corrélation active pour une exécution."""

    request_id: str
    execution_id: str


def nouvel_id() -> str:
    """Identifiant opaque neuf — `uuid4().hex` (32 caractères hexadécimaux)."""
    return uuid.uuid4().hex


def request_id_valide(valeur: str | None) -> str | None:
    """Renvoie `valeur` si c'est un `X-Request-Id` acceptable
    (`[A-Za-z0-9._-]`, longueur ≤ 128), sinon `None`."""
    if not valeur or not isinstance(valeur, str):
        return None
    return valeur if _REQUEST_ID_RE.match(valeur) else None


def current_request_id() -> str | None:
    """`request_id` du contexte courant, ou `None` hors contexte."""
    return _request_id.get()


def current_execution_id() -> str | None:
    """`execution_id` du contexte courant, ou `None` hors contexte."""
    return _execution_id.get()


def lier_correlation(request_id: str, execution_id: str) -> tuple[Token, Token]:
    """Lie les deux `ContextVar` et renvoie les tokens de reset. L'appelant
    **doit** passer les tokens à `delier_correlation` dans un `finally`."""
    return _request_id.set(request_id), _execution_id.set(execution_id)


def delier_correlation(tokens: tuple[Token, Token]) -> None:
    """Restaure l'état des `ContextVar` d'avant `lier_correlation`."""
    jeton_request, jeton_execution = tokens
    _request_id.reset(jeton_request)
    _execution_id.reset(jeton_execution)


@contextmanager
def portee_correlation(
    *,
    request_id: str | None = None,
    execution_id: str | None = None,
) -> Iterator[CorrelationIds]:
    """
    Garantit une paire d'identifiants de corrélation pour le bloc.

    - Sous HTTP : le middleware a déjà lié les `ContextVar` ; si les ids
      demandés coïncident avec ceux du contexte, **rien** n'est relié ni
      réinitialisé — l'exécution partage les ids de la requête.
    - Hors HTTP (script, tâche planifiée, test) : un contexte **complet** est
      créé (ids fournis ou générés), lié pour la durée du bloc, puis
      **toujours** réinitialisé dans le `finally`.
    """
    req_courant = current_request_id()
    exec_courant = current_execution_id()

    req = request_id or req_courant or nouvel_id()
    exe = execution_id or exec_courant or nouvel_id()

    if req == req_courant and exe == exec_courant:
        yield CorrelationIds(req, exe)
        return

    tokens = lier_correlation(req, exe)
    try:
        yield CorrelationIds(req, exe)
    finally:
        delier_correlation(tokens)


__all__ = [
    "HEADER_REQUEST_ID",
    "HEADER_EXECUTION_ID",
    "REQUEST_ID_MAX_LEN",
    "CorrelationIds",
    "nouvel_id",
    "request_id_valide",
    "current_request_id",
    "current_execution_id",
    "lier_correlation",
    "delier_correlation",
    "portee_correlation",
]
