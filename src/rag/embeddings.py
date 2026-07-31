"""
Vectorisation via BGE-M3 et reranking via BGE-reranker-v2-m3.

BGE-M3 produit en un seul passage :
  - un vecteur dense   (1024 dimensions)  -> similarité sémantique
  - un vecteur sparse  (lexical appris)   -> correspondance de termes

Les deux sont stockés ensemble dans Qdrant sous forme de vecteurs nommés.
C'est ce qui permet la recherche hybride sans maintenir d'index BM25 séparé.

Les modèles pèsent plusieurs Go : ils sont chargés une seule fois
(singletons) et jamais rechargés pendant la vie du processus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from src.config import get_settings

logger = logging.getLogger(__name__)

# Longueur maximale acceptée par BGE-M3. Bien au-delà de nos chunks (~1000),
# donc aucune troncature silencieuse — c'est l'intérêt principal du modèle.
LONGUEUR_MAX = 8192


# ===========================================================================
# Structures
# ===========================================================================

@dataclass
class VecteurSparse:
    """
    Représentation creuse : seuls les termes actifs sont stockés.
    Format directement consommable par Qdrant.
    """
    indices: list[int]
    valeurs: list[float]

    def est_vide(self) -> bool:
        return not self.indices


@dataclass
class Encodage:
    """Résultat d'un encodage : dense et sparse alignés sur les textes d'entrée."""
    dense: list[list[float]]
    sparse: list[VecteurSparse]

    def __len__(self) -> int:
        return len(self.dense)


# ===========================================================================
# Chargement des modèles (singletons)
# ===========================================================================

@lru_cache(maxsize=1)
def _modele_bge() -> Any:
    """
    Charge BGE-M3. Premier appel : téléchargement (~2,2 Go) puis mise en
    cache locale par HuggingFace. Appels suivants : instantané.
    """
    from FlagEmbedding import BGEM3FlagModel

    s = get_settings()
    logger.info("Chargement de %s sur %s…", s.embedding_model, s.embedding_device)

    # La signature a changé entre versions de FlagEmbedding ('device' -> 'devices').
    try:
        modele = BGEM3FlagModel(
            s.embedding_model,
            use_fp16=(s.embedding_device == "cuda"),
            devices=s.embedding_device,
        )
    except TypeError:
        modele = BGEM3FlagModel(
            s.embedding_model,
            use_fp16=(s.embedding_device == "cuda"),
            device=s.embedding_device,
        )

    logger.info("Modèle d'embedding prêt.")
    return modele


@lru_cache(maxsize=1)
def _modele_reranker() -> Any:
    """Charge le cross-encoder de reranking. Utilisé uniquement à la requête."""
    from FlagEmbedding import FlagReranker

    s = get_settings()
    logger.info("Chargement du reranker %s…", s.reranker_model)

    try:
        modele = FlagReranker(
            s.reranker_model,
            use_fp16=(s.embedding_device == "cuda"),
            devices=s.embedding_device,
        )
    except TypeError:
        modele = FlagReranker(
            s.reranker_model,
            use_fp16=(s.embedding_device == "cuda"),
            device=s.embedding_device,
        )

    logger.info("Reranker prêt.")
    return modele


def precharger_modeles(avec_reranker: bool = True) -> None:
    """
    Force le chargement en amont plutôt qu'au premier usage.
    Évite qu'une ingestion de 2000 fichiers paraisse figée pendant
    le téléchargement initial.
    """
    _modele_bge()
    if avec_reranker and get_settings().reranker_enabled:
        _modele_reranker()


# ===========================================================================
# Encodage
# ===========================================================================

def _convertir_sparse(poids: dict[str, float]) -> VecteurSparse:
    """
    BGE-M3 renvoie {token_id (str): poids}. Qdrant attend deux listes
    parallèles d'entiers et de flottants.
    """
    if not poids:
        return VecteurSparse(indices=[], valeurs=[])

    indices, valeurs = [], []
    for token_id, poids_token in poids.items():
        indices.append(int(token_id))
        valeurs.append(float(poids_token))
    return VecteurSparse(indices=indices, valeurs=valeurs)


def encoder(
    textes: list[str],
    avec_sparse: bool = True,
    taille_lot: int | None = None,
) -> Encodage:
    """
    Encode une liste de textes en dense (+ sparse).

    Les textes vides sont acceptés mais produisent des vecteurs nuls :
    l'appelant reste responsable de ne pas indexer de chunk vide.
    """
    if not textes:
        return Encodage(dense=[], sparse=[])

    s = get_settings()
    modele = _modele_bge()
    lot = taille_lot or s.embedding_batch_size

    sortie = modele.encode(
        textes,
        batch_size=lot,
        max_length=LONGUEUR_MAX,
        return_dense=True,
        return_sparse=avec_sparse,
        return_colbert_vecs=False,
    )

    dense = [vecteur.tolist() for vecteur in sortie["dense_vecs"]]

    if avec_sparse:
        sparse = [_convertir_sparse(p) for p in sortie["lexical_weights"]]
    else:
        sparse = [VecteurSparse([], []) for _ in textes]

    return Encodage(dense=dense, sparse=sparse)


def encoder_requete(texte: str, avec_sparse: bool = True) -> tuple[list[float], VecteurSparse]:
    """
    Encode une requête unique.

    BGE-M3 ne demande aucun préfixe ('query:' / 'passage:'), contrairement
    à la famille E5 — requête et document passent par le même chemin.
    """
    resultat = encoder([texte], avec_sparse=avec_sparse)
    return resultat.dense[0], resultat.sparse[0]


def encoder_dense_seul(textes: list[str]) -> list[list[float]]:
    """
    Signature attendue par normalization.py pour la résolution d'entités.
    Le sparse est inutile pour comparer deux noms courts, et l'omettre
    accélère sensiblement le traitement.
    """
    return encoder(textes, avec_sparse=False).dense


# ===========================================================================
# Reranking
# ===========================================================================

def reranker(
    requete: str,
    documents: list[str],
    top_k: int | None = None,
) -> list[tuple[int, float]]:
    """
    Réordonne des documents par pertinence face à la requête.

    Renvoie [(indice_original, score)] trié par score décroissant.
    L'indice permet à l'appelant de retrouver ses métadonnées sans
    que ce module ait à les connaître.

    Le cross-encoder lit la paire (requête, document) ensemble, là où
    l'embedding les encode séparément : bien plus précis, mais trop
    coûteux pour être appliqué à toute la base — d'où le pipeline
    recherche large puis reranking sur les candidats.
    """
    if not documents:
        return []

    if not get_settings().reranker_enabled:
        return [(i, 0.0) for i in range(len(documents))][:top_k]

    modele = _modele_reranker()
    paires = [[requete, doc] for doc in documents]
    scores = modele.compute_score(paires, normalize=True)

    if isinstance(scores, float):
        scores = [scores]

    classement = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return classement[:top_k] if top_k else classement


# ===========================================================================
# Diagnostic
# ===========================================================================

def dimension_dense() -> int:
    """Dimension du vecteur dense — doit correspondre à default.yaml."""
    return len(encoder(["test"], avec_sparse=False).dense[0])


# ===========================================================================
# Vérification manuelle : python -m src.rag.embeddings
# ===========================================================================

if __name__ == "__main__":
    import time

    from src.config import get_config_technique

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    textes = [
        "Le document précise les modalités de règlement et les délais applicables.",
        "The agreement specifies the payment terms and applicable deadlines.",
        "La recette traditionnelle demande une cuisson lente à feu doux.",
    ]

    print("Chargement du modèle (long au premier lancement)…")
    debut = time.time()
    resultat = encoder(textes)
    print(f"Encodage de {len(textes)} textes en {time.time() - debut:.1f}s\n")

    dim_attendue = get_config_technique().qdrant.taille_vecteur_dense
    dim_reelle = len(resultat.dense[0])
    statut = "OK" if dim_reelle == dim_attendue else "INCOHÉRENT"
    print(f"Dimension dense : {dim_reelle} (attendue {dim_attendue}) -> {statut}")
    print(f"Termes sparse   : {[len(v.indices) for v in resultat.sparse]}\n")

    print("--- Similarité dense (cosinus) ---")
    from src.rag.normalization import _cosinus

    print(f"  FR vs EN (même sens)   : {_cosinus(resultat.dense[0], resultat.dense[1]):.3f}")
    print(f"  FR vs sujet différent  : {_cosinus(resultat.dense[0], resultat.dense[2]):.3f}")

    print("\n--- Reranking ---")
    requete = "Quels sont les délais de paiement ?"
    for indice, score in reranker(requete, textes):
        print(f"  [{score:.3f}] {textes[indice][:60]}…")