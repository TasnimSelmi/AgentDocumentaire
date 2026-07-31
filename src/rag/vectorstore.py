"""
Couche Qdrant : création de collection, indexation, recherche hybride.

La collection utilise des vecteurs nommés : chaque point porte à la fois
un vecteur "dense" (1024d) et un vecteur "sparse" (lexical BGE-M3).
La fusion RRF des deux est faite côté Qdrant via l'API Query — aucun
index BM25 séparé à maintenir côté Python.

Les champs filtrables sont déclarés dans le profil YAML ; ce module lit
cette liste et crée les index de payload correspondants. Aucun nom de
champ métier n'apparaît dans le code.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable

from qdrant_client import QdrantClient, models

from src.config import (
    Champ,
    Profil,
    get_config_technique,
    get_profil,
    get_settings,
)
from src.rag.embeddings import VecteurSparse

logger = logging.getLogger(__name__)

NOM_VECTEUR_DENSE = "dense"
NOM_VECTEUR_SPARSE = "sparse"

# Namespace fixe : garantit qu'un même (doc_id, chunk) produit toujours
# le même identifiant de point, donc qu'une réindexation écrase au lieu
# de dupliquer.
_NAMESPACE = uuid.UUID("6f1a0c2e-4b7d-4f3a-9c11-8d2e5a7b1c40")


# ===========================================================================
# Structures
# ===========================================================================

@dataclass
class ChunkIndexable:
    """Une unité prête à être écrite dans Qdrant."""
    doc_id: str
    chunk_index: int
    texte: str
    dense: list[float]
    sparse: VecteurSparse
    payload: dict[str, Any] = field(default_factory=dict)

    def point_id(self) -> str:
        return str(uuid.uuid5(_NAMESPACE, f"{self.doc_id}:{self.chunk_index}"))


@dataclass
class Resultat:
    """Un chunk retourné par une recherche."""
    point_id: str
    score: float
    texte: str
    payload: dict[str, Any]

    @property
    def doc_id(self) -> str:
        return self.payload.get("doc_id", "")

    @property
    def source(self) -> str:
        return self.payload.get("source", "")

    @property
    def page(self) -> int | None:
        return self.payload.get("page")


# ===========================================================================
# Client
# ===========================================================================

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """
    Client Qdrant, mode local (fichier) ou serveur selon .env.

    En mode local, Qdrant pose un verrou sur le dossier : un seul
    processus à la fois. C'est acceptable en développement ; passer en
    mode serveur ne demande qu'un changement de variable d'environnement.
    """
    global _client
    if _client is not None:
        return _client

    s = get_settings()
    if s.qdrant_mode == "local":
        s.qdrant_path.mkdir(parents=True, exist_ok=True)
        _client = QdrantClient(path=str(s.qdrant_path))
        logger.info("Qdrant local : %s", s.qdrant_path)
    else:
        _client = QdrantClient(
            url=s.qdrant_url,
            api_key=s.qdrant_api_key or None,
        )
        logger.info("Qdrant serveur : %s", s.qdrant_url)

    return _client


def fermer_client() -> None:
    """Libère le verrou du mode local. À appeler en fin de script."""
    global _client
    if _client is not None:
        _client.close()
        _client = None


# ===========================================================================
# Collection
# ===========================================================================

_DISTANCES = {
    "cosine": models.Distance.COSINE,
    "euclid": models.Distance.EUCLID,
    "dot": models.Distance.DOT,
}


def _schema_payload(champ: Champ) -> models.PayloadSchemaType:
    """Type d'index Qdrant déduit du type déclaré dans le YAML."""
    if "date" in champ.type:
        return models.PayloadSchemaType.DATETIME
    if "entier" in champ.type:
        return models.PayloadSchemaType.INTEGER
    if "nombre" in champ.type:
        return models.PayloadSchemaType.FLOAT
    if "booleen" in champ.type:
        return models.PayloadSchemaType.BOOL
    return models.PayloadSchemaType.KEYWORD


def creer_collection(reinitialiser: bool = False, profil: Profil | None = None) -> None:
    """
    Crée la collection si absente, avec ses index de payload.

    Les index sont indispensables : sans eux, un filtre sur 2000 documents
    force un parcours complet et la latence s'effondre.
    """
    client = get_client()
    cfg = get_config_technique().qdrant
    profil = profil or get_profil()
    nom = cfg.nom_collection

    if reinitialiser and client.collection_exists(nom):
        client.delete_collection(nom)
        logger.info("Collection '%s' supprimée.", nom)

    if client.collection_exists(nom):
        return

    client.create_collection(
        collection_name=nom,
        vectors_config={
            NOM_VECTEUR_DENSE: models.VectorParams(
                size=cfg.taille_vecteur_dense,
                distance=_DISTANCES[cfg.distance],
            )
        },
        sparse_vectors_config=(
            {NOM_VECTEUR_SPARSE: models.SparseVectorParams()}
            if cfg.sparse_active
            else {}
        ),
    )
    logger.info("Collection '%s' créée.", nom)

    # Index techniques, toujours présents.
    for champ_technique, schema in (
        ("doc_id", models.PayloadSchemaType.KEYWORD),
        ("categorie", models.PayloadSchemaType.KEYWORD),
        ("source", models.PayloadSchemaType.KEYWORD),
    ):
        client.create_payload_index(nom, champ_technique, field_schema=schema)

    # Index métier, pilotés par le profil YAML.
    for champ in profil.champs_metadonnees:
        if champ.filtrable:
            client.create_payload_index(nom, champ.nom, field_schema=_schema_payload(champ))
            logger.debug("Index payload : %s (%s)", champ.nom, champ.type)


def info_collection() -> dict[str, Any]:
    client = get_client()
    nom = get_config_technique().qdrant.nom_collection
    if not client.collection_exists(nom):
        return {"existe": False, "nom": nom}

    infos = client.get_collection(nom)
    return {
        "existe": True,
        "nom": nom,
        "points": infos.points_count,
        "statut": str(infos.status),
    }


# ===========================================================================
# Écriture
# ===========================================================================

def indexer(chunks: list[ChunkIndexable]) -> int:
    """
    Écrit les chunks par lots. L'upsert écrase les points de même
    identifiant : réindexer un fichier modifié ne crée pas de doublons.
    """
    if not chunks:
        return 0

    client = get_client()
    cfg = get_config_technique().qdrant
    taille_lot = cfg.taille_lot_upsert
    total = 0

    for debut in range(0, len(chunks), taille_lot):
        lot = chunks[debut : debut + taille_lot]
        points = []

        for chunk in lot:
            vecteurs: dict[str, Any] = {NOM_VECTEUR_DENSE: chunk.dense}
            if cfg.sparse_active and not chunk.sparse.est_vide():
                vecteurs[NOM_VECTEUR_SPARSE] = models.SparseVector(
                    indices=chunk.sparse.indices,
                    values=chunk.sparse.valeurs,
                )

            payload = dict(chunk.payload)
            payload.update(
                {
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index,
                    "texte": chunk.texte,
                }
            )

            points.append(
                models.PointStruct(id=chunk.point_id(), vector=vecteurs, payload=payload)
            )

        client.upsert(collection_name=cfg.nom_collection, points=points, wait=True)
        total += len(points)

    return total


def supprimer_document(doc_id: str) -> None:
    """Retire tous les chunks d'un document. Appelé avant réindexation."""
    client = get_client()
    client.delete(
        collection_name=get_config_technique().qdrant.nom_collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            )
        ),
        wait=True,
    )


# ===========================================================================
# Filtres
# ===========================================================================

def construire_filtre(criteres: dict[str, Any] | None) -> models.Filter | None:
    """
    Traduit un dictionnaire de critères en filtre Qdrant.

    Formes acceptées :
        {"champ": "valeur"}                    -> égalité
        {"champ": ["a", "b"]}                  -> l'un ou l'autre
        {"champ": {"gte": x, "lte": y}}        -> plage (nombre ou date)

    L'agent construira ce dictionnaire à partir de champs_filtrables(),
    donc à partir du YAML — jamais de noms codés en dur.
    """
    if not criteres:
        return None

    conditions: list[models.Condition] = []

    for cle, valeur in criteres.items():
        if valeur is None:
            continue

        if isinstance(valeur, dict) and {"gte", "lte", "gt", "lt"} & valeur.keys():
            bornes = {k: v for k, v in valeur.items() if k in {"gte", "lte", "gt", "lt"}}
            est_date = any(isinstance(v, str) and "-" in v for v in bornes.values())
            plage = models.DatetimeRange(**bornes) if est_date else models.Range(**bornes)
            conditions.append(models.FieldCondition(key=cle, range=plage))

        elif isinstance(valeur, (list, tuple, set)):
            conditions.append(
                models.FieldCondition(key=cle, match=models.MatchAny(any=list(valeur)))
            )

        else:
            conditions.append(
                models.FieldCondition(key=cle, match=models.MatchValue(value=valeur))
            )

    return models.Filter(must=conditions) if conditions else None


# ===========================================================================
# Recherche
# ===========================================================================

def _vers_resultats(points: Iterable[Any]) -> list[Resultat]:
    return [
        Resultat(
            point_id=str(p.id),
            score=float(p.score),
            texte=(p.payload or {}).get("texte", ""),
            payload=p.payload or {},
        )
        for p in points
    ]


def rechercher(
    dense: list[float],
    sparse: VecteurSparse | None = None,
    filtre: models.Filter | None = None,
    limite: int | None = None,
) -> list[Resultat]:
    """
    Recherche hybride : deux pré-requêtes (dense et sparse) fusionnées
    par RRF côté Qdrant.

    RRF classe chaque résultat par son rang dans chaque liste plutôt que
    par son score brut — les scores dense et lexical n'étant pas sur la
    même échelle, les comparer directement n'aurait pas de sens.

    Bascule automatiquement en dense seul si le sparse est absent ou
    désactivé.
    """
    client = get_client()
    cfg = get_config_technique()
    nom = cfg.qdrant.nom_collection
    limite = limite or cfg.recherche.top_k_final

    hybride = (
        cfg.qdrant.sparse_active
        and sparse is not None
        and not sparse.est_vide()
    )

    if not hybride:
        reponse = client.query_points(
            collection_name=nom,
            query=dense,
            using=NOM_VECTEUR_DENSE,
            query_filter=filtre,
            limit=limite,
            with_payload=True,
        )
        return _vers_resultats(reponse.points)

    reponse = client.query_points(
        collection_name=nom,
        prefetch=[
            models.Prefetch(
                query=dense,
                using=NOM_VECTEUR_DENSE,
                filter=filtre,
                limit=cfg.recherche.top_k_dense,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse.indices,
                    values=sparse.valeurs,
                ),
                using=NOM_VECTEUR_SPARSE,
                filter=filtre,
                limit=cfg.recherche.top_k_sparse,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limite,
        with_payload=True,
    )
    return _vers_resultats(reponse.points)


def parcourir(
    filtre: models.Filter | None = None,
    limite: int = 100,
) -> list[Resultat]:
    """
    Récupère des points par filtre seul, sans vecteur.
    Utile pour lister les documents d'une catégorie ou alimenter
    un tableau de bord.
    """
    client = get_client()
    points, _ = get_client().scroll(
        collection_name=get_config_technique().qdrant.nom_collection,
        scroll_filter=filtre,
        limit=limite,
        with_payload=True,
        with_vectors=False,
    )
    return [
        Resultat(
            point_id=str(p.id),
            score=0.0,
            texte=(p.payload or {}).get("texte", ""),
            payload=p.payload or {},
        )
        for p in points
    ]


def compter(filtre: models.Filter | None = None) -> int:
    client = get_client()
    return client.count(
        collection_name=get_config_technique().qdrant.nom_collection,
        count_filter=filtre,
        exact=True,
    ).count


# ===========================================================================
# Vérification manuelle : python -m src.rag.vectorstore
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from src.rag.embeddings import encoder, encoder_requete

    print("--- Création de la collection ---")
    creer_collection(reinitialiser=True)
    print(info_collection())

    textes = [
        "Le présent accord fixe les modalités de règlement et les délais applicables.",
        "La facture correspondante devra être acquittée avant la date d'échéance.",
        "Le manuel décrit la procédure de maintenance préventive des équipements.",
    ]

    print("\n--- Indexation ---")
    enc = encoder(textes)
    chunks = [
        ChunkIndexable(
            doc_id="doc_demo",
            chunk_index=i,
            texte=t,
            dense=enc.dense[i],
            sparse=enc.sparse[i],
            payload={"source": "demo.txt", "page": 1, "categorie": "autre"},
        )
        for i, t in enumerate(textes)
    ]
    print(f"{indexer(chunks)} chunks indexés — total : {compter()}")

    print("\n--- Recherche hybride ---")
    requete = "Quand faut-il payer ?"
    d, s = encoder_requete(requete)
    for r in rechercher(d, s, limite=3):
        print(f"  [{r.score:.4f}] {r.texte[:60]}…")

    print("\n--- Recherche filtrée ---")
    filtre = construire_filtre({"categorie": "autre", "source": "demo.txt"})
    for r in rechercher(d, s, filtre=filtre, limite=2):
        print(f"  [{r.score:.4f}] p.{r.page} — {r.texte[:50]}…")

    fermer_client()