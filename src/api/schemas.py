"""
Schémas HTTP de la couche API — **entrées uniquement**, plus un helper de
sérialisation pour `RapportIngestion`.

Décision P2.3 (design validé) : aucun modèle Pydantic ne reproduit
`AgentResponse`. Le contrat public de `/query` reste `AgentResponse.vers_dict()`
(types natifs, garanti par P1.3). `/ingestion` renvoie le bilan du socle tel
quel via `rapport_vers_dict`.

`extra="forbid"` sur les deux corps de requête : toute clé inconnue — en
particulier une clé du type `path` / `dossier` / `racine` — provoque un `422`.
Le client ne choisit **jamais** un chemin du système de fichiers serveur ; il
désigne au plus une *source autorisée* par son nom logique (cf.
`src/api/dependencies.py`).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.rag.ingestion import RapportIngestion


class QueryRequest(BaseModel):
    """Corps de `POST /query`. La validation « chaîne non vide » n'est PAS
    faite ici : `AgentService` reste l'autorité unique (une requête vide ou
    blanche ressort en `AgentResponse(status="error", code="requete_invalide")`,
    que la route traduit en `422`)."""

    model_config = ConfigDict(extra="forbid")

    query: str


class IngestionRequest(BaseModel):
    """Corps de `POST /ingestion`. `source` est un **nom logique** résolu
    côté backend contre un registre de fabriques `DocumentSource` ; il ne
    transporte aucun chemin. `inferer` et `nom_profil` ne sont volontairement
    pas exposés par l'API MVP."""

    model_config = ConfigDict(extra="forbid")

    source: str = "local"
    reinitialiser: bool = False
    limite: int | None = Field(default=None, ge=1)


class HealthResponse(BaseModel):
    """Corps de `GET /health` — liveness pur, aucune dépendance sondée."""

    status: Literal["ok"] = "ok"


def rapport_vers_dict(rapport: RapportIngestion) -> dict[str, Any]:
    """Bilan d'ingestion en dict JSON-sûr — miroir fidèle du dataclass du
    socle (`RapportIngestion`), aucun champ retiré ni renommé."""
    return asdict(rapport)


__all__ = [
    "QueryRequest",
    "IngestionRequest",
    "HealthResponse",
    "rapport_vers_dict",
]
