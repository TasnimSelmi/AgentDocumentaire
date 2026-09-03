"""
Traduction des issues métier / techniques en statuts HTTP, et **nettoyage**
du corps de réponse.

Principes (design P2.3 validé) :

- un **refus fonctionnel** de l'agent (`status="refusal"`) n'est pas une
  panne : `200`, corps = contrat public intact ;
- une **requête invalide** signalée par l'autorité unique `AgentService`
  (`status="error"`, `code="requete_invalide"`) → `422`, corps intact
  (message API fixe, sans donnée sensible) ;
- toute **autre erreur** `status="error"` = échec technique → `500`, et le
  bloc `error` est **remplacé** par un message générique : jamais de
  traceback, d'exception brute, de chemin local, de secret ou de détail
  d'infrastructure dans la réponse HTTP ;
- `ErreurSource` (source documentaire indisponible) → `503` (dépendance
  temporairement indisponible), détail générique ;
- toute exception inattendue → `500`, détail générique.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.agent.response import STATUT_ERREUR, STATUT_REFUS, STATUT_SUCCES, AgentResponse
from src.agent.service import CODE_REQUETE_INVALIDE
from src.observability import emettre_http_unhandled_error
from src.sources.base import ErreurSource

HTTP_OK = 200
HTTP_REQUETE_INVALIDE = 422
HTTP_ERREUR_INTERNE = 500
HTTP_SOURCE_INDISPONIBLE = 503

#: Bloc `error` public substitué à tout échec technique de `/query`.
ERREUR_INTERNE_PUBLIQUE: dict[str, str] = {
    "code": "internal_error",
    "message": "Une erreur interne est survenue.",
}
_DETAIL_SOURCE_INDISPONIBLE = "Source documentaire temporairement indisponible."
_DETAIL_ERREUR_INTERNE = "Erreur interne."


def statut_http_pour(reponse: AgentResponse) -> int:
    """Statut HTTP d'un `AgentResponse` — sans jamais transformer un refus
    métier en panne."""
    if reponse.status in (STATUT_SUCCES, STATUT_REFUS):
        return HTTP_OK
    if reponse.status == STATUT_ERREUR:
        code = (reponse.error or {}).get("code")
        if code == CODE_REQUETE_INVALIDE:
            return HTTP_REQUETE_INVALIDE
        return HTTP_ERREUR_INTERNE
    # `status` hors nomenclature : traité comme un échec technique.
    return HTTP_ERREUR_INTERNE


def corps_reponse_query(reponse: AgentResponse) -> dict[str, Any]:
    """Corps JSON de `/query`. Renvoie `AgentResponse.vers_dict()` **tel
    quel**, sauf pour un échec technique où le bloc `error` est masqué (voir
    module). Aucune autre clé n'est touchée."""
    corps = reponse.vers_dict()
    if reponse.status == STATUT_ERREUR:
        code = (reponse.error or {}).get("code")
        if code != CODE_REQUETE_INVALIDE:
            corps["error"] = dict(ERREUR_INTERNE_PUBLIQUE)
    return corps


def _ids_correlation(request: Request) -> tuple[str | None, str | None]:
    """Ids de corrélation posés par `CorrelationMiddleware` sur `request.state`.

    Le gestionnaire `Exception` (500) est traité par `ServerErrorMiddleware`,
    **en dehors** du middleware de corrélation : ses `ContextVar` sont déjà
    réinitialisés et son `send` n'est pas enveloppé. `request.state` reste la
    source fiable, et ces en-têtes sont réappliqués explicitement ici."""
    etat = getattr(request, "state", None)
    return getattr(etat, "request_id", None), getattr(etat, "execution_id", None)


def _entetes_correlation(request: Request) -> dict[str, str]:
    request_id, execution_id = _ids_correlation(request)
    entetes: dict[str, str] = {}
    if request_id:
        entetes["X-Request-Id"] = request_id
    if execution_id:
        entetes["X-Execution-Id"] = execution_id
    return entetes


def enregistrer_gestionnaires_erreurs(app: FastAPI) -> None:
    """Branche les gestionnaires d'exception non liés à `/query`."""

    @app.exception_handler(ErreurSource)
    async def _sur_erreur_source(request: Request, _exc: ErreurSource) -> JSONResponse:
        return JSONResponse(
            status_code=HTTP_SOURCE_INDISPONIBLE,
            content={"detail": _DETAIL_SOURCE_INDISPONIBLE},
            headers=_entetes_correlation(request),
        )

    @app.exception_handler(Exception)
    async def _sur_exception(request: Request, exc: Exception) -> JSONResponse:
        # P2.4 : signal d'observabilité `http_unhandled_error` — jamais de
        # détail technique renvoyé au client (réponse identique à P2.3).
        try:
            runtime = getattr(request.app.state, "observability", None)
            if runtime is not None and getattr(runtime, "enabled", False):
                request_id, execution_id = _ids_correlation(request)
                emettre_http_unhandled_error(
                    runtime.sink,
                    exc,
                    request_id=request_id,
                    operation_id=execution_id,
                )
        except Exception:  # noqa: BLE001 — l'observabilité ne casse jamais la réponse
            pass

        return JSONResponse(
            status_code=HTTP_ERREUR_INTERNE,
            content={"detail": _DETAIL_ERREUR_INTERNE},
            headers=_entetes_correlation(request),
        )


__all__ = [
    "HTTP_OK",
    "HTTP_REQUETE_INVALIDE",
    "HTTP_ERREUR_INTERNE",
    "HTTP_SOURCE_INDISPONIBLE",
    "ERREUR_INTERNE_PUBLIQUE",
    "statut_http_pour",
    "corps_reponse_query",
    "enregistrer_gestionnaires_erreurs",
]
