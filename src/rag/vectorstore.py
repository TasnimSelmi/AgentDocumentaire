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
from typing import Any, Iterable, Sequence

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


def _assurer_indexes_payload(
    client: QdrantClient,
    nom_collection: str,
    profil: Profil,
) -> None:
    """Crée les index de payload absents, sans recréer ceux déjà présents."""
    infos = client.get_collection(nom_collection)
    schema_existant = getattr(infos, "payload_schema", None) or {}
    champs_indexes = set(schema_existant)

    index_requis: list[tuple[str, models.PayloadSchemaType]] = [
        ("doc_id", models.PayloadSchemaType.KEYWORD),
        ("nom_fichier", models.PayloadSchemaType.KEYWORD),
        ("categorie", models.PayloadSchemaType.KEYWORD),
        ("source", models.PayloadSchemaType.KEYWORD),
        # Nécessaires à l'expansion du contexte (parent et voisins).
        ("chunk_index", models.PayloadSchemaType.INTEGER),
        ("parent_id", models.PayloadSchemaType.KEYWORD),
        ("table_id", models.PayloadSchemaType.KEYWORD),
    ]
    index_requis.extend(
        (champ.nom, _schema_payload(champ))
        for champ in profil.champs_metadonnees
        if champ.filtrable
    )

    for nom_champ, schema in index_requis:
        if nom_champ in champs_indexes:
            continue
        client.create_payload_index(
            collection_name=nom_collection,
            field_name=nom_champ,
            field_schema=schema,
            wait=True,
        )
        logger.debug("Index payload créé : %s (%s)", nom_champ, schema)


def creer_collection(reinitialiser: bool = False, profil: Profil | None = None) -> None:
    """
    Crée la collection si absente et garantit ses index de payload.

    Les index sont indispensables : sans eux, un filtre sur un corpus
    volumineux force un parcours complet et dégrade fortement la latence.
    Les index manquants sont aussi ajoutés à une collection déjà existante.
    """
    client = get_client()
    cfg = get_config_technique().qdrant
    profil = profil or get_profil()
    nom = cfg.nom_collection

    if reinitialiser and client.collection_exists(nom):
        client.delete_collection(nom)
        logger.info("Collection '%s' supprimée.", nom)

    if not client.collection_exists(nom):
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

    _assurer_indexes_payload(client, nom, profil)


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


def recuperer_contexte(
    doc_id: str,
    *,
    parent_id: str | None = None,
    indices: Sequence[int] | None = None,
    limite: int = 50,
) -> list[Resultat]:
    """
    Récupère les chunks voisins d'un chunk déjà trouvé.

    Deux modes, dans cet ordre de préférence :

    - ``parent_id`` : tous les chunks du même groupe logique (un tableau
      complet, une section) ;
    - ``indices`` : les chunks du même document dont ``chunk_index`` figure
      dans la liste, ce qui couvre le voisinage immédiat.

    Le filtre porte toujours sur ``doc_id`` : un voisin ne peut jamais
    provenir d'un autre document.
    """
    if not doc_id or (parent_id is None and not indices):
        return []

    conditions: list[models.FieldCondition] = [
        models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
    ]
    if parent_id is not None:
        conditions.append(
            models.FieldCondition(
                key="parent_id", match=models.MatchValue(value=parent_id)
            )
        )
    else:
        conditions.append(
            models.FieldCondition(
                key="chunk_index", match=models.MatchAny(any=list(indices or []))
            )
        )

    return parcourir(models.Filter(must=conditions), limite=limite)


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


# Champs d'identité toujours lus, même si l'appelant n'en demande pas d'autres.
_CHAMPS_IDENTITE_MINIMAUX = ("doc_id", "nom_fichier", "source")
 
 
def lister_documents(
    champs: Sequence[str] | None = None,
    limite: int = 20_000,
) -> list[dict[str, Any]]:
    """
    Renvoie l'identité des documents distincts présents dans la collection.
 
    Un balayage des payloads suffit : seuls les champs d'identité sont lus,
    sans vecteur ni texte, ce qui reste peu coûteux même sur un gros corpus.
    Le résultat alimente le catalogue documentaire de retrieval.py, qui en
    dérive son vocabulaire discriminant.
 
    ``champs`` est la liste des clés à remonter. retrieval.py y passe les
    conventions de nommage qu'il sait exploiter, complétées par les champs de
    métadonnées du profil actif : aucune clé n'est donc figée ici, et un
    corpus disposant de ses propres noms de champs reste exploitable.
    Les clés absentes d'un payload sont simplement ignorées.
    """
    demandes = list(dict.fromkeys([*_CHAMPS_IDENTITE_MINIMAUX, *(champs or ())]))
 
    if limite < 1:
        raise ValueError("limite doit être supérieure ou égale à 1.")

    client = get_client()
    nom_collection = get_config_technique().qdrant.nom_collection
    if not client.collection_exists(nom_collection):
        return []

    documents: dict[str, dict[str, Any]] = {}
    decalage: Any = None

    while len(documents) < limite:
        taille_page = min(1024, limite - len(documents))
        points, decalage = client.scroll(
            collection_name=nom_collection,
            limit=taille_page,
            offset=decalage,
            with_payload=demandes,
            with_vectors=False,
        )
        if not points:
            break
 
        for point in points:
            payload = point.payload or {}
            cle = str(
                payload.get("doc_id")
                or payload.get("nom_fichier")
                or payload.get("source")
                or ""
            ).strip()
            if not cle or cle in documents:
                continue
            documents[cle] = {
                nom: valeur
                for nom, valeur in payload.items()
                if valeur is not None and valeur != ""
            }
 
        if decalage is None:
            break
 
    return list(documents.values())


def parcourir_tout(
    filtre: models.Filter | None,
    *,
    taille_page: int = 1024,
) -> list[Resultat]:
    """
    Récupère TOUS les points correspondant à un filtre, sans vecteur.

    Contrairement à ``parcourir()``, borné par ``limite`` (utile pour un
    aperçu ou un voisinage volontairement plafonné), cette fonction épuise
    toutes les pages du scroll Qdrant : un document dont le nombre de chunks
    dépasse une seule page n'en perd donc silencieusement aucun. Réservée à
    la lecture intégrale d'un document (voir ``retrieval.charger_document``) ;
    ``parcourir()`` reste inchangée pour ses appelants existants.
    """
    if taille_page < 1:
        raise ValueError("taille_page doit être supérieure ou égale à 1.")

    client = get_client()
    nom_collection = get_config_technique().qdrant.nom_collection

    resultats: list[Resultat] = []
    decalage: Any = None

    while True:
        points, decalage = client.scroll(
            collection_name=nom_collection,
            scroll_filter=filtre,
            limit=taille_page,
            offset=decalage,
            with_payload=True,
            with_vectors=False,
        )
        resultats.extend(
            Resultat(
                point_id=str(p.id),
                score=0.0,
                texte=(p.payload or {}).get("texte", ""),
                payload=p.payload or {},
            )
            for p in points
        )
        if not points or decalage is None:
            break

    return resultats


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