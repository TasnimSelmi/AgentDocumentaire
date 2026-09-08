"""
Les trois routes du MVP. Chacune : validation HTTP (Pydantic) → un appel de
service → adaptation HTTP. Aucune logique documentaire.

Les collaborateurs sont lus sur `request.app.state` (injectés par
`create_app`), ce qui permet aux tests de fournir des doublures sans toucher
au réseau.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from src.agent.service import AgentService
from src.api.errors import corps_reponse_query, statut_http_pour
from src.api.schemas import (
    CorpusCreateRequest,
    DomainProfileProposeRequest,
    DomainProfileValidateRequest,
    HealthResponse,
    ImportUrlRequest,
    IngestionRequest,
    QueryRequest,
    rapport_vers_dict,
    resume_corpus_vers_dict,
)
from src.api.uploads import UploadInvalide, ecrire_lot, valider_lot
from src.api.url_import import ImportUrlInvalide, telecharger
from src.config import CorpusDejaExistant, CorpusProtege, get_config_technique
from src.profiling import DomainProfileGenerationError, save_domain_profile, suggest_domain_profile
from src.profiling.models import DomainProfile
from src.rag.corpus import (
    CorpusInconnu,
    ErreurCorpusInvalide,
    declarer_corpus,
    dossier_managed_pour_corpus,
    enregistrer_rapport_ingestion,
    lister_resumes_corpus,
    resoudre_corpus,
    resume_corpus,
    supprimer_corpus,
)
from src.sources import IngestionService

logger = logging.getLogger(__name__)

#: Description honnête des sources logiques connues, pour `GET /sources` —
#: jamais un connecteur inventé. Toute source absente de
#: `request.app.state.sources` n'apparaît jamais dans la réponse, quel que
#: soit son statut ici.
_DESCRIPTIONS_SOURCES: dict[str, dict[str, str]] = {
    "local": {"name": "Dossier local du serveur", "type": "local"},
    "managed": {
        "name": "Documents envoyés ou importés pour ce corpus",
        "type": "managed",
    },
}

router = APIRouter()


def _resoudre_ou_404(corpus_id: str):
    """Valide/résout un `corpus_id` de chemin d'URL : `422` si mal formé,
    `404` s'il est absent du registre — jamais une collection/chemin fourni
    par le client, jamais une écriture Qdrant tentée pour vérifier."""
    try:
        return resume_corpus(corpus_id)
    except ErreurCorpusInvalide as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except CorpusInconnu as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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
    reponse = service.query(corps.query, corpus_id=corps.corpus_id)
    return JSONResponse(
        status_code=statut_http_pour(reponse),
        content=corps_reponse_query(reponse),
    )


@router.post("/ingestion", tags=["ingestion"])
def ingestion(corps: IngestionRequest, request: Request) -> JSONResponse:
    """Résout le corpus logique cible et sa source DÉCLARÉE, puis délègue à
    `IngestionService.sync`. `corpus_id` mal formé ou non déclaré → `422`,
    exactement comme une source inconnue — jamais d'écriture Qdrant tentée.

    La source effective est TOUJOURS celle enregistrée pour ce corpus
    (`ConfigCorpus.source`), jamais choisie librement par le client : un
    `source` fourni qui ne correspond pas à cette valeur est un `422`,
    jamais un simple avertissement — un corpus synchronisé tour à tour
    contre deux sources différentes ferait lire par erreur au registre de
    fichiers de CE corpus une suppression de documents qui n'ont en réalité
    jamais disparu (voir `src.sources.base` sur l'invariant de snapshot).

    `ErreurSource` → `503` (via gestionnaire). Le corps de succès est le
    `RapportIngestion` du socle, intact — y compris lorsqu'il compte des
    échecs par fichier (`fichiers_en_echec > 0`)."""
    registre = request.app.state.sources
    service: IngestionService = request.app.state.ingestion_service

    try:
        contexte = resoudre_corpus(corps.corpus_id)
    except (ErreurCorpusInvalide, CorpusInconnu) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if corps.source is not None and corps.source != contexte.source:
        raise HTTPException(
            status_code=422,
            detail=(
                f"La source fournie ({corps.source!r}) ne correspond pas à "
                f"la source déclarée pour ce corpus ({contexte.source!r})."
            ),
        )

    fabrique = registre.get(contexte.source)
    if fabrique is None:
        raise HTTPException(status_code=422, detail=f"Source inconnue : {contexte.source!r}.")

    rapport = service.sync(
        fabrique(corps.corpus_id),
        reinitialiser=corps.reinitialiser,
        limite=corps.limite,
        corpus_id=corps.corpus_id,
    )
    # Trace minimale, persistée dans le registre du corpus (jamais un
    # historique complet) : dérive le statut fonctionnel de `GET /corpora`
    # sans rejouer l'ingestion. Ne doit jamais faire échouer une ingestion
    # par ailleurs réussie.
    try:
        enregistrer_rapport_ingestion(corps.corpus_id, rapport)
    except Exception:  # noqa: BLE001
        pass
    return JSONResponse(status_code=200, content=rapport_vers_dict(rapport))


@router.get("/corpora", tags=["corpora"])
def lister_corpora() -> JSONResponse:
    """Liste tous les corpus déclarés avec leurs statistiques réelles
    (Qdrant + registre) — jamais calculées par un LLM."""
    resumes = lister_resumes_corpus()
    return JSONResponse(status_code=200, content=[resume_corpus_vers_dict(r) for r in resumes])


@router.get("/corpora/{corpus_id}", tags=["corpora"])
def obtenir_corpus(corpus_id: str) -> JSONResponse:
    """Détail d'un corpus — alimente la carte sélectionnée de la page
    Gestion des corpus et le panneau « Corpus & indexation » de la page
    Interrogation. `404` si `corpus_id` n'est pas déclaré."""
    resume = _resoudre_ou_404(corpus_id)
    return JSONResponse(status_code=200, content=resume_corpus_vers_dict(resume))


@router.post("/corpora", tags=["corpora"])
def creer_corpus(corps: CorpusCreateRequest) -> JSONResponse:
    """Déclare un nouveau corpus logique. Ne crée AUCUNE collection Qdrant
    (création différée à la première ingestion, voir `src.rag.corpus`) :
    l'état renvoyé est donc toujours `indexation_required`."""
    try:
        resume = declarer_corpus(
            corps.corpus_id,
            nom=corps.name,
            source=corps.source,
            profil=corps.profile_name,
        )
    except ErreurCorpusInvalide as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except CorpusDejaExistant as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(status_code=201, content=resume_corpus_vers_dict(resume))


@router.delete("/corpora/{corpus_id}", tags=["corpora"], status_code=204)
def supprimer_corpus_route(corpus_id: str) -> Response:
    """
    Supprime un corpus logique : sa collection Qdrant, son registre de
    fichiers, son profil de domaine persisté, puis son entrée de registre —
    dans cet ordre (`src.rag.corpus.supprimer_corpus`), jamais un autre
    corpus. `422` si `corpus_id` est mal formé, `404` s'il n'est pas
    déclaré, `403` pour « default » (protégé — cible implicite de tout
    appel sans `corpus_id`, cf. rétrocompatibilité). Une erreur de
    suppression (Qdrant, disque) n'est jamais masquée mais jamais exposée
    en détail au client : `500` générique, trace journalisée côté serveur.
    """
    _resoudre_ou_404(corpus_id)
    try:
        supprimer_corpus(corpus_id)
    except CorpusProtege as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — jamais de trace exposée au client
        logger.exception("Échec de la suppression du corpus %r", corpus_id)
        raise HTTPException(
            status_code=500, detail="La suppression du corpus a échoué."
        ) from exc
    return Response(status_code=204)


@router.get("/sources", tags=["sources"])
def lister_sources(request: Request) -> JSONResponse:
    """
    Sources logiques RÉELLEMENT enregistrées côté backend
    (`request.app.state.sources`) — jamais un connecteur inventé, jamais un
    credential ni un chemin. Aujourd'hui : `local` (dossier serveur partagé,
    historique) et `managed` (stockage propre à un corpus, alimenté par
    upload/import URL) — aucune source distante (GED/SharePoint/API) tant
    qu'aucune n'est effectivement enregistrée ici.
    """
    registre = request.app.state.sources
    contenu = [
        {
            "source_id": source_id,
            "name": _DESCRIPTIONS_SOURCES.get(source_id, {}).get("name", source_id),
            "type": _DESCRIPTIONS_SOURCES.get(source_id, {}).get("type", "custom"),
            "available": True,
        }
        for source_id in sorted(registre)
    ]
    return JSONResponse(status_code=200, content=contenu)


@router.post("/corpora/{corpus_id}/upload", tags=["corpora"])
async def upload_dossier(
    corpus_id: str,
    files: list[UploadFile] = File(...),
    paths: list[str] = Form(...),
) -> JSONResponse:
    """
    Matérialise un lot de fichiers déjà sélectionnés côté client (upload de
    dossier) dans le stockage géré de CE corpus — additif, jamais un
    remplacement du contenu déjà présent. `files`/`paths` sont appariés par
    position (même ordre, même longueur) : `paths[i]` est le chemin relatif
    déclaré du fichier `files[i]` au sein du dossier sélectionné.

    N'indexe rien : `POST /ingestion` (`source` déclarée du corpus =
    `"managed"`) reste l'unique porte d'entrée d'indexation
    (`IngestionService.sync`) — aucune logique d'ingestion dupliquée ici.

    `404` corpus inconnu. `422` : corpus non déclaré en source `managed`,
    aucun fichier, désaccord `files`/`paths`, trop de fichiers, un format non
    supporté, un fichier vide ou trop volumineux, une taille totale excessive,
    ou un chemin relatif invalide (absolu, `..`, hors périmètre) — dans tous
    ces cas, validation intégrale du lot AVANT toute écriture : rien n'est
    écrit si un seul fichier est refusé.
    """
    resume = _resoudre_ou_404(corpus_id)
    if resume.source != "managed":
        raise HTTPException(
            status_code=422,
            detail="Ce corpus n'est pas déclaré avec la source 'managed' (upload/import).",
        )
    if len(files) != len(paths):
        raise HTTPException(
            status_code=422, detail="Fichiers et chemins relatifs incohérents."
        )

    cfg = get_config_technique().ingestion
    contenus = [(chemin, await f.read()) for f, chemin in zip(files, paths)]

    try:
        valides = valider_lot(
            contenus,
            extensions_autorisees=set(cfg.extensions_supportees),
            taille_max_fichier_octets=cfg.taille_max_mo * 1024 * 1024,
        )
    except UploadInvalide as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        ecrire_lot(valides, racine=dossier_managed_pour_corpus(corpus_id))
    except OSError as exc:
        logger.exception("Échec d'écriture de l'upload pour le corpus %r", corpus_id)
        raise HTTPException(status_code=500, detail="L'upload a échoué.") from exc

    return JSONResponse(
        status_code=200,
        content={
            "files_received": len(valides),
            "total_size_bytes": sum(len(f.contenu) for f in valides),
        },
    )


@router.post("/corpora/{corpus_id}/import-url", tags=["corpora"])
def importer_url(corpus_id: str, corps: ImportUrlRequest) -> JSONResponse:
    """
    Télécharge un document depuis une URL publique (validation SSRF stricte,
    voir `src.api.url_import`) et le matérialise dans le stockage géré de CE
    corpus — additif, comme l'upload. N'indexe rien (même porte d'entrée
    d'indexation que l'upload : `POST /ingestion`).

    `404` corpus inconnu. `422` : corpus non déclaré en source `managed`,
    schéma non http(s), hôte résolu vers une plage réseau interdite, format
    non supporté, redirection invalide/en boucle, taille dépassée, échec de
    téléchargement — jamais un détail réseau brut exposé au client.
    """
    resume = _resoudre_ou_404(corpus_id)
    if resume.source != "managed":
        raise HTTPException(
            status_code=422,
            detail="Ce corpus n'est pas déclaré avec la source 'managed' (upload/import).",
        )

    cfg = get_config_technique().ingestion
    try:
        nom_fichier, contenu = telecharger(
            corps.url,
            extensions_autorisees=set(cfg.extensions_supportees),
            taille_max_octets=cfg.taille_max_mo * 1024 * 1024,
        )
    except ImportUrlInvalide as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        racine = dossier_managed_pour_corpus(corpus_id)
        racine.mkdir(parents=True, exist_ok=True)
        (racine / nom_fichier).write_bytes(contenu)
    except OSError as exc:
        logger.exception("Échec d'écriture de l'import URL pour le corpus %r", corpus_id)
        raise HTTPException(status_code=500, detail="L'import a échoué.") from exc

    return JSONResponse(
        status_code=200,
        content={"file_saved": nom_fichier, "size_bytes": len(contenu)},
    )


@router.post("/corpora/{corpus_id}/profile", tags=["corpora"])
def proposer_profil(corpus_id: str, corps: DomainProfileProposeRequest) -> JSONResponse:
    """
    Lance une VRAIE proposition de profil de domaine (`suggest_domain_profile`,
    appel LLM réel) pour CE corpus uniquement. Ne persiste rien : la
    proposition reste modifiable côté client jusqu'à
    `POST /corpora/{corpus_id}/profile/validate`. `404` si le corpus n'est
    pas déclaré.
    """
    _resoudre_ou_404(corpus_id)
    try:
        profil = suggest_domain_profile(corps.domain, corps.output_language)
    except DomainProfileGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return JSONResponse(
        status_code=200,
        content={
            "domain": profil.domain,
            "description": profil.description,
            "keywords": profil.keywords,
            "output_language": profil.output_language,
        },
    )


@router.post("/corpora/{corpus_id}/profile/validate", tags=["corpora"])
def valider_profil(corpus_id: str, corps: DomainProfileValidateRequest) -> JSONResponse:
    """
    Persiste le profil de domaine (potentiellement corrigé par
    l'utilisateur) et l'associe à CE corpus — jamais un
    `active_domain_profile` global mutable : l'association vit dans le
    registre du corpus (`profil_domaine`), lue explicitement par chaque
    requête portant ce `corpus_id` (`src.agent.session.construire_session`).
    `404` si le corpus n'est pas déclaré.
    """
    _resoudre_ou_404(corpus_id)
    try:
        profil = DomainProfile(
            profile_name=corpus_id,
            domain=corps.domain,
            description=corps.description,
            keywords=corps.keywords,
            output_language=corps.output_language,
        )
        save_domain_profile(profil, overwrite=True)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    from src.config import mettre_a_jour_corpus

    mettre_a_jour_corpus(corpus_id, profil_domaine=profil.profile_name)
    resume = resume_corpus(corpus_id)
    return JSONResponse(status_code=200, content=resume_corpus_vers_dict(resume))


__all__ = ["router"]
