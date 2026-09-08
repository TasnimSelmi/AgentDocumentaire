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

from src.rag.corpus import ResumeCorpus
from src.rag.ingestion import RapportIngestion


class QueryRequest(BaseModel):
    """Corps de `POST /query`. La validation « chaîne non vide » n'est PAS
    faite ici : `AgentService` reste l'autorité unique (une requête vide ou
    blanche ressort en `AgentResponse(status="error", code="requete_invalide")`,
    que la route traduit en `422`).

    `corpus_id` : contexte applicatif logique (`src.rag.corpus`), jamais un
    nom de collection ni un chemin. Optionnel pour compatibilité avec les
    appels existants (repli sur le corpus « default » côté service) ; une
    UI multi-corpus doit le fournir systématiquement. Un `corpus_id` mal
    formé ou non déclaré ressort en `AgentResponse(status="error",
    code="corpus_invalide")`, traduit en `422`."""

    model_config = ConfigDict(extra="forbid")

    query: str
    corpus_id: str = "default"


class IngestionRequest(BaseModel):
    """Corps de `POST /ingestion`. `source` est un **nom logique** résolu
    côté backend contre un registre de fabriques `DocumentSource` ; il ne
    transporte aucun chemin. `inferer` et `nom_profil` ne sont volontairement
    pas exposés par l'API MVP.

    `corpus_id` : corpus logique cible (`src.rag.corpus`) — détermine la
    collection Qdrant, le registre de fichiers et le profil utilisés,
    jamais fournis directement par le client. Optionnel, replié sur
    « default » (comportement historique inchangé). `reinitialiser` ne
    détruit alors que CE corpus, jamais un autre.

    `source` optionnel : la source EFFECTIVE est toujours celle déclarée pour
    ce corpus (`ConfigCorpus.source`, cf. `POST /corpora`), jamais une valeur
    arbitraire injectée par le client — un corpus ne peut être synchronisé
    qu'avec la source avec laquelle il a été déclaré, pour qu'un changement de
    source ne fasse jamais lire au registre de fichiers du corpus une
    suppression fictive. Si fourni, doit correspondre exactement à cette
    source déclarée (sinon `422`) ; c'est une vérification, jamais un choix."""

    model_config = ConfigDict(extra="forbid")

    source: str | None = None
    corpus_id: str = "default"
    reinitialiser: bool = False
    limite: int | None = Field(default=None, ge=1)


class HealthResponse(BaseModel):
    """Corps de `GET /health` — liveness pur, aucune dépendance sondée."""

    status: Literal["ok"] = "ok"


class CorpusCreateRequest(BaseModel):
    """Corps de `POST /corpora`. `profile_name` ici désigne le profil
    TECHNIQUE (schéma de classification/extraction, `config/schemas/*.yaml`)
    — optionnel, replié sur `generic`. Le profil de DOMAINE (vocabulaire
    métier) se déclare séparément, via `/corpora/{corpus_id}/profile`."""

    model_config = ConfigDict(extra="forbid")

    corpus_id: str
    name: str | None = None
    source: str = "local"
    profile_name: str | None = None


class DomainProfileProposeRequest(BaseModel):
    """Corps de `POST /corpora/{corpus_id}/profile`. Ne persiste rien : la
    proposition reste éditable côté client jusqu'à validation."""

    model_config = ConfigDict(extra="forbid")

    domain: str
    output_language: str = "fr"


class ImportUrlRequest(BaseModel):
    """Corps de `POST /corpora/{corpus_id}/import-url`. `url` : document
    public accessible en http(s) uniquement — jamais `file://`/`ftp://`, ni
    un chemin filesystem. Validée côté serveur (schéma, résolution DNS,
    plages privées/loopback/link-local interdites, revalidées à chaque
    redirection) avant toute requête réseau — voir `src.api.url_import`."""

    model_config = ConfigDict(extra="forbid")

    url: str


class DomainProfileValidateRequest(BaseModel):
    """Corps de `POST /corpora/{corpus_id}/profile/validate`. Porte le
    contenu (potentiellement corrigé par l'utilisateur) de la proposition à
    persister — jamais recalculé côté serveur à ce stade."""

    model_config = ConfigDict(extra="forbid")

    domain: str
    description: str
    keywords: list[str]
    output_language: str = "fr"


def rapport_vers_dict(rapport: RapportIngestion) -> dict[str, Any]:
    """Bilan d'ingestion en dict JSON-sûr — miroir fidèle du dataclass du
    socle (`RapportIngestion`), aucun champ retiré ni renommé."""
    return asdict(rapport)


def resume_corpus_vers_dict(resume: ResumeCorpus) -> dict[str, Any]:
    """Résumé d'un corpus en dict JSON-sûr — noms de champs alignés sur le
    contrat attendu par le frontend (`corpus_id`, `nom`, `status`, ...)."""
    return {
        "corpus_id": resume.corpus_id,
        "nom": resume.nom,
        "source": resume.source,
        "status": resume.statut,
        "document_count": resume.document_count,
        "chunk_count": resume.chunk_count,
        "domain": resume.domain,
        "profile_name": resume.profile_name,
        "created_at": resume.created_at,
        "last_sync_at": resume.last_sync_at,
    }


__all__ = [
    "QueryRequest",
    "IngestionRequest",
    "HealthResponse",
    "CorpusCreateRequest",
    "ImportUrlRequest",
    "DomainProfileProposeRequest",
    "DomainProfileValidateRequest",
    "rapport_vers_dict",
    "resume_corpus_vers_dict",
]
