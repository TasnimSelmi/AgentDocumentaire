"""
Les trois routes du MVP. Chacune : validation HTTP (Pydantic) → un appel de
service → adaptation HTTP. Aucune logique documentaire.

Les collaborateurs sont lus sur `request.app.state` (injectés par
`create_app`), ce qui permet aux tests de fournir des doublures sans toucher
au réseau.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from src.agent.service import AgentService
from src.api.errors import corps_reponse_query, statut_http_pour
from src.api.schemas import HealthResponse, IngestionRequest, QueryRequest, rapport_vers_dict
from src.sources import IngestionService

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness. Ne sonde ni Ollama, ni Qdrant, ni la source documentaire."""
    return HealthResponse()


@router.post("/query", tags=["agent"])
def query(corps: QueryRequest, request: Request) -> JSONResponse:
    """Délègue à `AgentService.query` et renvoie `AgentResponse.vers_dict()`.

    - `success` / `refusal` → `200` (un refus métier n'est pas une panne) ;
    - `error` + `requete_invalide` → `422` ;
    - autre `error` → `500`, bloc `error` masqué.
    """
    service: AgentService = request.app.state.agent_service
    reponse = service.query(corps.query)
    return JSONResponse(
        status_code=statut_http_pour(reponse),
        content=corps_reponse_query(reponse),
    )


@router.post("/ingestion", tags=["ingestion"])
def ingestion(corps: IngestionRequest, request: Request) -> JSONResponse:
    """Résout la source autorisée par son nom logique puis délègue à
    `IngestionService.sync`. `ErreurSource` → `503` (via gestionnaire). Le
    corps de succès est le `RapportIngestion` du socle, intact — y compris
    lorsqu'il compte des échecs par fichier (`fichiers_en_echec > 0`)."""
    registre = request.app.state.sources
    service: IngestionService = request.app.state.ingestion_service

    fabrique = registre.get(corps.source)
    if fabrique is None:
        raise HTTPException(status_code=422, detail=f"Source inconnue : {corps.source!r}.")

    rapport = service.sync(
        fabrique(),
        reinitialiser=corps.reinitialiser,
        limite=corps.limite,
    )
    return JSONResponse(status_code=200, content=rapport_vers_dict(rapport))


__all__ = ["router"]
