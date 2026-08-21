"""
Tests de l'expansion du contexte (parent et voisins).

La récupération Qdrant est injectée : aucun serveur, aucune collection.
"""

from __future__ import annotations

from src.rag.retrieval import etendre_contexte
from src.rag.vectorstore import Resultat


def _resultat(doc: str, index: int, *, parent: str | None = None, texte: str = "") -> Resultat:
    payload = {"doc_id": doc, "chunk_index": index}
    if parent:
        payload["parent_id"] = parent
    return Resultat(
        point_id=f"{doc}-{index}",
        score=1.0,
        texte=texte or f"contenu {doc} {index}",
        payload=payload,
    )


class FauxDepot:
    """Imite `recuperer_contexte` à partir d'un corpus en mémoire."""

    def __init__(self, chunks: list[Resultat]):
        self.chunks = chunks
        self.appels: list[dict] = []

    def __call__(self, doc_id, *, parent_id=None, indices=None, limite=50):
        self.appels.append({"doc_id": doc_id, "parent_id": parent_id, "indices": indices})
        resultats = [c for c in self.chunks if c.payload.get("doc_id") == doc_id]
        if parent_id is not None:
            return [c for c in resultats if c.payload.get("parent_id") == parent_id]
        return [c for c in resultats if c.payload.get("chunk_index") in set(indices or [])]


def _etendre(retenus, depot, **surcharges):
    parametres = {
        "rayon": 1,
        "max_chunks_ajoutes": 12,
        "taille_max_contexte": 100_000,
        "recuperer": depot,
    }
    parametres.update(surcharges)
    return etendre_contexte(retenus, **parametres)


def test_voisins_recuperes_autour_du_chunk():
    corpus = [_resultat("A", i) for i in range(5)]
    depot = FauxDepot(corpus)

    etendus = _etendre([(corpus[2], 0.9)], depot)
    indices = [r.payload["chunk_index"] for r, _ in etendus]

    assert indices == [2, 1, 3]
    assert depot.appels[0]["indices"] == [1, 3]


def test_parent_prioritaire_sur_les_voisins():
    corpus = [_resultat("A", i, parent="A:t1") for i in range(3)]
    corpus.append(_resultat("A", 9))  # hors parent
    depot = FauxDepot(corpus)

    etendus = _etendre([(corpus[1], 0.9)], depot)
    indices = [r.payload["chunk_index"] for r, _ in etendus]

    assert indices == [1, 0, 2]  # tout le parent, rien d'autre
    assert depot.appels[0]["parent_id"] == "A:t1"


def test_aucun_doublon():
    corpus = [_resultat("A", i) for i in range(5)]
    depot = FauxDepot(corpus)

    etendus = _etendre([(corpus[1], 0.9), (corpus[2], 0.8)], depot)
    point_ids = [r.point_id for r, _ in etendus]

    assert len(point_ids) == len(set(point_ids))


def test_voisins_limites_au_meme_document():
    corpus = [_resultat("A", i) for i in range(3)] + [_resultat("B", i) for i in range(3)]
    depot = FauxDepot(corpus)

    etendus = _etendre([(corpus[1], 0.9)], depot)

    assert {r.payload["doc_id"] for r, _ in etendus} == {"A"}
    assert all(appel["doc_id"] == "A" for appel in depot.appels)


def test_un_parent_nest_recupere_quune_fois():
    corpus = [_resultat("A", i, parent="A:t1") for i in range(4)]
    depot = FauxDepot(corpus)

    _etendre([(corpus[0], 0.9), (corpus[1], 0.8)], depot)

    assert len(depot.appels) == 1


def test_ordre_naturel_preserve():
    corpus = [_resultat("A", i, parent="A:s1") for i in range(4)]
    depot = FauxDepot(corpus)

    etendus = _etendre([(corpus[2], 0.9)], depot)
    ajoutes = [r.payload["chunk_index"] for r, score in etendus if score is None]

    assert ajoutes == sorted(ajoutes)


def test_scores_des_ajouts_a_none():
    corpus = [_resultat("A", i) for i in range(3)]
    etendus = _etendre([(corpus[1], 0.9)], FauxDepot(corpus))

    assert etendus[0][1] == 0.9
    assert all(score is None for _, score in etendus[1:])


def test_plafond_de_chunks():
    corpus = [_resultat("A", i, parent="A:t1") for i in range(20)]
    etendus = _etendre([(corpus[0], 0.9)], FauxDepot(corpus), max_chunks_ajoutes=3)

    assert len(etendus) == 4  # le chunk d'origine + 3 ajouts


def test_plafond_de_caracteres():
    corpus = [_resultat("A", i, parent="A:t1", texte="x" * 100) for i in range(10)]
    etendus = _etendre([(corpus[0], 0.9)], FauxDepot(corpus), taille_max_contexte=350)

    assert sum(len(r.texte) for r, _ in etendus) <= 350


def test_expansion_desactivee():
    corpus = [_resultat("A", i) for i in range(3)]
    etendus = _etendre([(corpus[1], 0.9)], FauxDepot(corpus), rayon=0, max_chunks_ajoutes=0)

    assert etendus == [(corpus[1], 0.9)]


def test_chunk_sans_index_ignore():
    resultat = Resultat(point_id="A-x", score=1.0, texte="t", payload={"doc_id": "A"})
    etendus = _etendre([(resultat, 0.9)], FauxDepot([]))

    assert len(etendus) == 1


def test_erreur_de_recuperation_non_bloquante():
    def depot_casse(*args, **kwargs):
        raise RuntimeError("qdrant indisponible")

    corpus = [_resultat("A", 1)]
    etendus = _etendre([(corpus[0], 0.9)], depot_casse)

    assert etendus == [(corpus[0], 0.9)]