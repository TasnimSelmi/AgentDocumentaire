"""
Tests de la primitive de lecture documentaire complète (`charger_document`).

Cette primitive n'est pas une recherche : aucun Qdrant réel, aucun embedding,
aucun reranker, aucun LLM. `parcourir_tout` (vectorstore) et `parcourir_tout`
tel qu'appelé depuis `charger_document` (retrieval) sont tous deux injectés
à partir d'un corpus en mémoire, sur le modèle de `test_expansion_contexte.py`.
"""

from __future__ import annotations

import types

import pytest

from src.rag import retrieval
from src.rag import vectorstore
from src.rag.retrieval import CollectionIndisponible, DocumentInconnu, charger_document
from src.rag.vectorstore import Resultat


# ===========================================================================
# Fabriques
# ===========================================================================


def _resultat(doc_id: str, chunk_index: int | None, *, point_id: str | None = None, texte: str = "") -> Resultat:
    payload: dict = {"doc_id": doc_id, "page": 1}
    if chunk_index is not None:
        payload["chunk_index"] = chunk_index
    return Resultat(
        point_id=point_id or f"{doc_id}-{chunk_index}",
        score=0.0,
        texte=texte or f"contenu {doc_id} {chunk_index}",
        payload=payload,
    )


def _faux_parcourir_tout(corpus: list[Resultat]):
    """
    Imite le comportement d'un vrai Qdrant filtré par ``doc_id`` : lit la clé
    et la valeur du filtre réellement construit par ``construire_filtre`` et
    ne renvoie que les points du corpus qui correspondent.
    """

    def _parcourir(filtre, *, taille_page: int = 1024):
        condition = filtre.must[0]
        assert condition.key == "doc_id"
        cible = condition.match.value
        return [r for r in corpus if r.payload.get("doc_id") == cible]

    return _parcourir


@pytest.fixture(autouse=True)
def _collection_disponible(monkeypatch):
    """Par défaut, la collection existe : chaque test le désactive s'il le teste."""
    monkeypatch.setattr(retrieval, "info_collection", lambda: {"existe": True, "points": 1})


# ===========================================================================
# Test 1 — récupération complète
# ===========================================================================


def test_recuperation_complete_tous_les_chunks(monkeypatch):
    corpus = [_resultat("A", i) for i in range(7)]
    monkeypatch.setattr(retrieval, "parcourir_tout", _faux_parcourir_tout(corpus))

    passages = charger_document("A")

    assert len(passages) == 7


# ===========================================================================
# Test 2 — ordre
# ===========================================================================


def test_ordre_reconstruit_depuis_chunk_index(monkeypatch):
    # Qdrant renvoie volontairement dans le désordre.
    corpus = [_resultat("A", i) for i in [4, 1, 3, 2, 0]]
    monkeypatch.setattr(retrieval, "parcourir_tout", _faux_parcourir_tout(corpus))

    passages = charger_document("A")

    assert [p.chunk_index for p in passages] == [0, 1, 2, 3, 4]
    assert [p.rang for p in passages] == [1, 2, 3, 4, 5]


# ===========================================================================
# Test 3 — isolation documentaire
# ===========================================================================


def test_isolation_documentaire(monkeypatch):
    corpus = [_resultat("A", i) for i in range(3)] + [_resultat("B", i) for i in range(2)]
    monkeypatch.setattr(retrieval, "parcourir_tout", _faux_parcourir_tout(corpus))

    passages = charger_document("A")

    assert len(passages) == 3
    assert {p.doc_id for p in passages} == {"A"}


# ===========================================================================
# Test 4 — pagination (couche vectorstore, scroll Qdrant simulé)
# ===========================================================================


class _FauxClientPagine:
    """Simule le scroll paginé de Qdrant sans jamais s'y connecter."""

    def __init__(self, points):
        self._points = list(points)
        self.appels: list[dict] = []

    def scroll(self, *, collection_name, scroll_filter, limit, offset, with_payload, with_vectors):
        self.appels.append({"limit": limit, "offset": offset})
        debut = offset or 0
        page = self._points[debut : debut + limit]
        fin = debut + limit
        suite = fin if fin < len(self._points) else None
        return page, suite


class _FauxPoint:
    def __init__(self, id_, payload):
        self.id = id_
        self.payload = payload


def test_pagination_recupere_tous_les_chunks_au_dela_dune_page(monkeypatch):
    n = 250
    points = [_FauxPoint(f"p{i}", {"doc_id": "A", "chunk_index": i}) for i in range(n)]
    faux_client = _FauxClientPagine(points)

    monkeypatch.setattr(vectorstore, "get_client", lambda: faux_client)
    monkeypatch.setattr(
        vectorstore,
        "get_config_technique",
        lambda: types.SimpleNamespace(qdrant=types.SimpleNamespace(nom_collection="test")),
    )

    resultats = vectorstore.parcourir_tout(filtre=None, taille_page=64)

    assert len(resultats) == n
    assert {r.payload["chunk_index"] for r in resultats} == set(range(n))
    # Aucune page ne pouvait contenir les 250 points : plusieurs appels ont été nécessaires.
    assert len(faux_client.appels) > 1


# ===========================================================================
# Test 5 — document inconnu
# ===========================================================================


def test_document_inconnu_leve_une_exception(monkeypatch):
    corpus = [_resultat("A", 0)]
    monkeypatch.setattr(retrieval, "parcourir_tout", _faux_parcourir_tout(corpus))

    with pytest.raises(DocumentInconnu):
        charger_document("Z")


def test_doc_id_vide_leve_une_exception(monkeypatch):
    with pytest.raises(DocumentInconnu):
        charger_document("   ")


def test_collection_indisponible_court_circuite_avant_toute_lecture(monkeypatch):
    monkeypatch.setattr(retrieval, "info_collection", lambda: {"existe": False})

    appele = False

    def _parcourir_jamais_appele(*args, **kwargs):
        nonlocal appele
        appele = True
        return []

    monkeypatch.setattr(retrieval, "parcourir_tout", _parcourir_jamais_appele)

    with pytest.raises(CollectionIndisponible):
        charger_document("A")

    assert appele is False


# ===========================================================================
# Test 6 — aucun appel au pipeline RAG (recherche, embeddings, reranker)
# ===========================================================================


def test_aucun_appel_a_la_recherche_semantique_ni_au_reranker(monkeypatch):
    corpus = [_resultat("A", i) for i in range(3)]
    monkeypatch.setattr(retrieval, "parcourir_tout", _faux_parcourir_tout(corpus))

    def _echec(*args, **kwargs):
        raise AssertionError("charger_document ne doit appeler aucune primitive de recherche.")

    monkeypatch.setattr(retrieval, "rechercher", _echec)
    monkeypatch.setattr(retrieval, "encoder_requete", _echec)
    monkeypatch.setattr(retrieval, "reranker", _echec)
    monkeypatch.setattr(retrieval, "rechercher_passages", _echec)

    passages = charger_document("A")

    assert len(passages) == 3


# ===========================================================================
# Métadonnée d'ordre absente : comportement déterministe, jamais silencieux
# ===========================================================================


def test_chunk_index_absent_relegue_en_fin_de_liste_avec_avertissement(monkeypatch, caplog):
    corpus = [
        _resultat("A", 0),
        _resultat("A", None, point_id="A-orphelin"),
        _resultat("A", 1),
    ]
    monkeypatch.setattr(retrieval, "parcourir_tout", _faux_parcourir_tout(corpus))

    with caplog.at_level("WARNING"):
        passages = charger_document("A")

    assert [p.chunk_index for p in passages[:2]] == [0, 1]
    assert passages[2].point_id == "A-orphelin"
    assert any("chunk_index" in enregistrement.message for enregistrement in caplog.records)


# ===========================================================================
# Métadonnées et provenance conservées
# ===========================================================================


def test_metadonnees_et_provenance_conservees(monkeypatch):
    resultat = _resultat("A", 0)
    resultat.payload.update({"nom_fichier": "rapport.pdf", "source": "docs/rapport.pdf", "categorie": "autre"})
    monkeypatch.setattr(retrieval, "parcourir_tout", _faux_parcourir_tout([resultat]))

    passage = charger_document("A")[0]

    assert passage.doc_id == "A"
    assert passage.nom_fichier == "rapport.pdf"
    assert passage.source == "docs/rapport.pdf"
    assert passage.page == 1
    assert passage.texte == resultat.texte
