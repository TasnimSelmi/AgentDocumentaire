"""
Résolution du corpus logique actif — frontière d'isolation multi-corpus.

1 `corpus_id` -> 1 collection Qdrant + 1 registre de fichiers + 1 profil,
strictement isolés. `corpus_id` est le SEUL identifiant qu'un appelant
externe (frontend, API) peut fournir : le backend en dérive ici, et nulle
part ailleurs, le nom de collection Qdrant et le chemin du registre —
jamais l'inverse. Un client ne fournit jamais de nom de collection, de
`qdrant_path`, ni de chemin de registre arbitraire.

Aucun état mutable global : `resoudre_corpus()` reconstruit un
`ContexteCorpus` autonome à chaque appel, sûr à partager entre requêtes
concurrentes portant sur des corpus différents.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.config import (
    ConfigCorpus,
    CorpusDejaExistant,
    CorpusInconnu,
    CorpusProtege,
    DerniereIngestion,
    ErreurCorpusInvalide,
    Profil,
    creer_corpus as _declarer_corpus,
    get_config_technique,
    get_corpus,
    get_profil,
    get_settings,
    lister_corpus,
    mettre_a_jour_corpus,
    supprimer_corpus_config,
    valider_corpus_id,
)

#: Corpus de compatibilité : préserve la collection/le registre historiques
#: (ceux d'avant l'introduction du multi-corpus), jamais un nom arbitraire.
CORPUS_DEFAUT = "default"


@dataclass(frozen=True)
class ContexteCorpus:
    """Tout ce dont l'ingestion et la recherche ont besoin pour rester
    strictement scopées à UN corpus — jamais construit à partir d'une
    entrée fournie directement par un client."""

    corpus_id: str
    nom_collection: str
    chemin_registre: Path
    profil: Profil
    profil_domaine_nom: str | None
    source: str


def nom_collection_pour_corpus(corpus_id: str) -> str:
    """
    Dérive le nom de collection Qdrant depuis `corpus_id` — jamais choisi
    par le client. `corpus_id="default"` préserve la collection
    historiquement configurée (`qdrant.nom_collection`) ; tout autre corpus
    obtient une collection dédiée, nommée de façon déterministe et stable.
    """
    base = get_config_technique().qdrant.nom_collection
    if corpus_id == CORPUS_DEFAUT:
        return base
    return f"{base}__{corpus_id}"


def chemin_registre_pour_corpus(corpus_id: str) -> Path:
    """
    Chemin du `RegistreFichiers` dédié à ce corpus. `corpus_id="default"`
    préserve le chemin historique (`Settings.chemin_registre`,
    `qdrant_path/registry.json`) ; tout autre corpus obtient son propre
    fichier, sous `qdrant_path/registries/<corpus_id>.json` — jamais le
    même fichier comparé au périmètre d'un autre corpus (voir l'audit
    multi-corpus : c'est exactement ce qui causait une suppression croisée).
    """
    if corpus_id == CORPUS_DEFAUT:
        return get_settings().chemin_registre
    return get_settings().qdrant_path / "registries" / f"{corpus_id}.json"


def dossier_managed_pour_corpus(corpus_id: str) -> Path:
    """
    Répertoire de stockage géré par le backend pour CE corpus — cible de tout
    upload de dossier ou import par URL (source logique `"managed"`), jamais
    un chemin fourni par le client. Sous `Settings.corpora_dir`, jamais
    `src/` ni `config/`. N'est créé sur disque qu'au premier upload/import
    réel (`Path.mkdir` par l'appelant) : sa seule existence sur disque, pas
    une entrée de registre, signale qu'un contenu y a déjà été déposé.
    """
    return get_settings().corpora_dir / corpus_id / "documents"


def resoudre_corpus(corpus_id: str | None) -> ContexteCorpus:
    """
    Point d'entrée UNIQUE de résolution `corpus_id -> contexte complet`.

    Valide le format de `corpus_id` (`ErreurCorpusInvalide` sinon), vérifie
    qu'il est déclaré dans `config/corpus.yaml` (`CorpusInconnu` sinon),
    puis dérive collection, registre et profil. `corpus_id=None` résout le
    corpus « default », pour compatibilité avec le code qui ne connaît pas
    encore le multi-corpus.
    """
    corpus_id = valider_corpus_id(corpus_id or CORPUS_DEFAUT)
    entree: ConfigCorpus = get_corpus(corpus_id)

    return ContexteCorpus(
        corpus_id=corpus_id,
        nom_collection=nom_collection_pour_corpus(corpus_id),
        chemin_registre=chemin_registre_pour_corpus(corpus_id),
        profil=get_profil(entree.profil),
        profil_domaine_nom=entree.profil_domaine,
        source=entree.source,
    )


#: Statut fonctionnel exposé à l'API — SEULE la lecture des signaux réels
#: (collection Qdrant, dernier rapport d'ingestion connu, profil de domaine
#: associé) le détermine ; jamais stocké tel quel (voir `ConfigCorpus`).
StatutCorpus = Literal[
    "indexation_required", "profiling_required", "ready", "partial", "error"
]


@dataclass(frozen=True)
class ResumeCorpus:
    """Vue agrégée d'UN corpus pour l'API `/corpora` — uniquement des
    données réelles (Qdrant, registre), jamais calculées par un LLM."""

    corpus_id: str
    nom: str
    source: str
    statut: StatutCorpus
    document_count: int
    chunk_count: int
    domain: str | None
    profile_name: str | None
    created_at: str | None
    last_sync_at: str | None


def _statut_et_stats(corpus_id: str, entree: ConfigCorpus) -> tuple[StatutCorpus, int, int]:
    """Dérive `(statut, document_count, chunk_count)` depuis Qdrant + le
    dernier rapport d'ingestion connu — jamais depuis un état stocké tel
    quel. Import différé de `vectorstore` : évite tout cycle avec
    `src.rag.retrieval` (qui importe déjà ce module)."""
    from src.rag import vectorstore

    contexte = resoudre_corpus(corpus_id)
    infos = vectorstore.info_collection(nom_collection=contexte.nom_collection)
    if not infos.get("existe") or not infos.get("points"):
        return "indexation_required", 0, 0

    chunk_count = int(infos.get("points") or 0)
    document_count = len(vectorstore.lister_documents(nom_collection=contexte.nom_collection))

    derniere = entree.derniere_ingestion
    if derniere is not None and derniere.statut == "echec":
        return "error", document_count, chunk_count
    if entree.profil_domaine is None:
        return "profiling_required", document_count, chunk_count
    if derniere is not None and derniere.statut == "partiel":
        return "partial", document_count, chunk_count
    return "ready", document_count, chunk_count


def resume_corpus(corpus_id: str) -> ResumeCorpus:
    """Vue complète d'UN corpus — alimente `GET /corpora/{corpus_id}` et
    chaque carte de `GET /corpora`. Lève `CorpusInconnu` si non déclaré."""
    corpus_id = valider_corpus_id(corpus_id)
    entree = get_corpus(corpus_id)
    statut, document_count, chunk_count = _statut_et_stats(corpus_id, entree)

    domaine: str | None = None
    if entree.profil_domaine:
        try:
            from src.profiling import load_domain_profile

            domaine = load_domain_profile(entree.profil_domaine).domain
        except Exception:  # noqa: BLE001 — une référence orpheline n'affiche pas de domaine, ne casse rien
            domaine = None

    return ResumeCorpus(
        corpus_id=corpus_id,
        nom=entree.nom or corpus_id,
        source=entree.source,
        statut=statut,
        document_count=document_count,
        chunk_count=chunk_count,
        domain=domaine,
        profile_name=entree.profil_domaine,
        created_at=entree.cree_le,
        last_sync_at=derniere.a if (derniere := entree.derniere_ingestion) else None,
    )


def lister_resumes_corpus() -> list[ResumeCorpus]:
    """Vue agrégée de tous les corpus déclarés — alimente `GET /corpora`."""
    return [resume_corpus(cid) for cid in lister_corpus()]


def declarer_corpus(
    corpus_id: str, *, nom: str | None = None, source: str = "local", profil: str | None = None
) -> ResumeCorpus:
    """Déclare un nouveau corpus logique (`POST /corpora`) puis renvoie son
    résumé — immédiatement en état « indexation requise », sans écriture
    Qdrant (voir `src.config.creer_corpus`). Lève `ErreurCorpusInvalide`,
    `CorpusDejaExistant`."""
    _declarer_corpus(corpus_id, nom=nom, source=source, profil=profil)
    return resume_corpus(corpus_id)


def supprimer_corpus(corpus_id: str) -> None:
    """
    Supprime un corpus logique et UNIQUEMENT son périmètre : sa collection
    Qdrant, son registre de fichiers, son profil de domaine persisté, puis
    son entrée dans `config/corpus.yaml` — dans cet ordre précis, pour
    qu'un échec intermédiaire ne laisse jamais une entrée de registre
    pointer vers une collection déjà supprimée, ni l'inverse une collection
    orpheline sans trace dans le registre.

    Ordre :
      1. validation du format + existence (via `resoudre_corpus`/`get_corpus`) ;
      2. suppression de la collection Qdrant ;
      3. suppression du registre de fichiers propre à ce corpus ;
      4. suppression du stockage géré (uploads / imports URL) propre à ce
         corpus, s'il existe — évite qu'un `corpus_id` réutilisé plus tard
         hérite silencieusement de fichiers d'un corpus supprimé ;
      5. suppression du profil de domaine persisté, si associé ;
      6. suppression de l'entrée `config/corpus.yaml` (dernière étape :
         ne retire la trace qu'une fois le reste effectivement nettoyé).

    Lève `ErreurCorpusInvalide` / `CorpusInconnu` (mêmes règles que les
    autres opérations), `CorpusProtege` si `corpus_id == CORPUS_DEFAUT` —
    « default » reste la cible implicite de tout appel sans `corpus_id`
    (compatibilité ascendante) et ne peut donc pas être supprimé physiquement.
    Une erreur de suppression (Qdrant, disque) n'est jamais masquée : elle
    remonte telle quelle à l'appelant (la route HTTP la journalise sans en
    exposer le détail au client).
    """
    corpus_id = valider_corpus_id(corpus_id)
    if corpus_id == CORPUS_DEFAUT:
        raise CorpusProtege(
            "Le corpus 'default' ne peut pas être supprimé : il reste la "
            "cible implicite de tout appel sans corpus_id."
        )
    entree = get_corpus(corpus_id)  # CorpusInconnu si absent, avant tout effet de bord.
    contexte = resoudre_corpus(corpus_id)

    from src.rag import vectorstore

    vectorstore.supprimer_collection(nom_collection=contexte.nom_collection)

    if contexte.chemin_registre.exists():
        contexte.chemin_registre.unlink()

    dossier_managed = dossier_managed_pour_corpus(corpus_id).parent  # <corpus_id>/
    if dossier_managed.exists():
        import shutil

        shutil.rmtree(dossier_managed)

    if entree.profil_domaine:
        from src.profiling import delete_domain_profile

        delete_domain_profile(entree.profil_domaine)

    supprimer_corpus_config(corpus_id)


def enregistrer_rapport_ingestion(corpus_id: str, rapport: Any) -> None:
    """
    Persiste un résumé du `RapportIngestion` dans le registre de CE corpus,
    seule trace conservée entre deux appels — pas un historique, pas une
    machine à états : `_statut_et_stats` recalcule le statut fonctionnel à
    chaque lecture à partir de ce résumé + de l'état réel de la collection.
    """
    trouves = int(getattr(rapport, "fichiers_trouves", 0) or 0)
    echecs = int(getattr(rapport, "fichiers_en_echec", 0) or 0)
    if trouves > 0 and echecs >= trouves:
        statut: Literal["succes", "partiel", "echec"] = "echec"
    elif echecs > 0:
        statut = "partiel"
    else:
        statut = "succes"

    derniere = DerniereIngestion(
        statut=statut,
        fichiers_trouves=trouves,
        fichiers_traites=int(getattr(rapport, "fichiers_traites", 0) or 0),
        fichiers_en_echec=echecs,
        fichiers_ignores_inchanges=int(getattr(rapport, "fichiers_ignores_inchanges", 0) or 0),
        chunks_indexes=int(getattr(rapport, "chunks_indexes", 0) or 0),
        duree_secondes=float(getattr(rapport, "duree_secondes", 0.0) or 0.0),
        a=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    )
    mettre_a_jour_corpus(corpus_id, derniere_ingestion=derniere)


__all__ = [
    "CORPUS_DEFAUT",
    "ContexteCorpus",
    "CorpusDejaExistant",
    "CorpusInconnu",
    "CorpusProtege",
    "ErreurCorpusInvalide",
    "StatutCorpus",
    "ResumeCorpus",
    "resoudre_corpus",
    "nom_collection_pour_corpus",
    "chemin_registre_pour_corpus",
    "dossier_managed_pour_corpus",
    "resume_corpus",
    "lister_resumes_corpus",
    "declarer_corpus",
    "supprimer_corpus",
    "enregistrer_rapport_ingestion",
]
