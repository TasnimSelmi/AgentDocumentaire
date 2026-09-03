"""
`CorrelationMiddleware` — middleware **ASGI pur** (pas `BaseHTTPMiddleware`).

Pour chaque requête HTTP :

1. lit `X-Request-Id` entrant, l'accepte **uniquement** s'il respecte
   `[A-Za-z0-9._-]` et une longueur ≤ 128 ; sinon en génère un (`uuid4().hex`) ;
2. génère **toujours** un `X-Execution-Id` côté serveur ;
3. lie les deux ids aux `ContextVar` de `correlation.py` ;
4. recopie les deux ids dans `scope["state"]` (lecture via `request.state`) ;
5. ajoute `X-Request-Id` et `X-Execution-Id` aux en-têtes de réponse ;
6. réinitialise **toujours** les `ContextVar` dans un `finally`.

Le middleware ne produit **aucun** événement métier, n'avale **aucune**
exception (seul un `try/finally` de reset l'entoure) et ne contient **aucune**
logique Agent / RAG.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, MutableMapping

from src.observability.correlation import (
    HEADER_EXECUTION_ID,
    HEADER_REQUEST_ID,
    delier_correlation,
    lier_correlation,
    nouvel_id,
    request_id_valide,
)

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

_REQ = HEADER_REQUEST_ID.encode("latin-1")
_EXE = HEADER_EXECUTION_ID.encode("latin-1")


class CorrelationMiddleware:
    """Middleware ASGI de corrélation. `__init__(app)` — signature ASGI standard,
    compatible `app.add_middleware(CorrelationMiddleware)`."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        entrant = _lire_entete(scope, _REQ)
        request_id = request_id_valide(entrant) or nouvel_id()
        execution_id = nouvel_id()

        etat = scope.setdefault("state", {})
        etat["request_id"] = request_id
        etat["execution_id"] = execution_id

        async def send_avec_entetes(message: Message) -> None:
            if message["type"] == "http.response.start":
                entetes = [
                    (cle, valeur)
                    for (cle, valeur) in message.get("headers") or []
                    if cle.lower() not in (_REQ, _EXE)
                ]
                entetes.append((_REQ, request_id.encode("latin-1")))
                entetes.append((_EXE, execution_id.encode("latin-1")))
                message = {**message, "headers": entetes}
            await send(message)

        tokens = lier_correlation(request_id, execution_id)
        try:
            await self.app(scope, receive, send_avec_entetes)
        finally:
            delier_correlation(tokens)


def _lire_entete(scope: Scope, nom: bytes) -> str | None:
    """Première valeur d'en-tête `nom` (bytes ASGI) décodée, ou `None`."""
    for cle, valeur in scope.get("headers") or []:
        if cle.lower() == nom:
            try:
                return valeur.decode("latin-1")
            except Exception:  # noqa: BLE001
                return None
    return None


__all__ = ["CorrelationMiddleware"]
