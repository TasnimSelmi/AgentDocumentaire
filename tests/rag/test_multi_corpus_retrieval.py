"""
Isolation multi-corpus — couche retrieval (`catalogue`, `charger_document`,
`rechercher_passages`).

Aucun Qdrant réel, aucun embedding réel (`faux_qdrant`, `tests/rag/conftest.py`) :
le VRAI code de `src.rag.retrieval` / `src.rag.vectorstore` tourne contre un
état en mémoire organisé par collection. Deux corpus, "corpus_a" et
"corpus_b", chacun avec un document nommé IDENTIQUEMENT ("rapport.pdf") mais
un contenu et un secret distincts — exactement le scénario de l'audit
multi-corpus.
"""

from __future__ import annotations

import pytest

from src.config import ConfigCorpus, get_profil
from src.rag import retrieval, vectorstore
from src.rag.embeddings import VecteurSparse
from src.rag.retrieval import CatalogueDocuments, DocumentInconnu
from src.rag.vectorstore import ChunkIndexable
from tests.rag.conftest import cabler_registre_corpus

ALPHA = "ALPHA_SECRET_123"
BETA = "BETA_SECRET_456"

_REGISTRE = {
    "default": ConfigCorpus(),
    "corpus_a": ConfigCorpus(profil="generic", source="local"),
    "corpus_b": ConfigCorpus(profil="generic", source="local"),
}


def _indexer(nom_collection: str, corpus_id: str, doc_id: str, texte: str, page: int = 1) -> None:
    vectorstore.creer_collection(nom_collection=nom_collection)
    chunk = ChunkIndexable(
        doc_id=doc_id,
        chunk_index=0,
        texte=texte,
        dense=[0.1],
        sparse=VecteurSparse(indices=[], valeurs=[]),
        payload={
            "nom_fichier": "rapport.pdf",
            "source": "rapport.pdf",
            "categorie": "autre",
            "page": page,
        },
        corpus_id=corpus_id,
    )
    vectorstore.indexer([chunk], nom_collection=nom_collection)


@pytest.fixture()
def deux_corpus(faux_qdrant, monkeypatch):
    """
    corpus_a : doc_id="A1", rapport.pdf, contenu ALPHA_SECRET_123
    corpus_b : doc_id="B1", rapport.pdf (MÊME NOM), contenu BETA_SECRET_456
    """
    cabler_registre_corpus(monkeypatch, _REGISTRE)
    from src.rag.corpus import nom_collection_pour_corpus

    col_a = nom_collection_pour_corpus("corpus_a")
    col_b = nom_collection_pour_corpus("corpus_b")
    _indexer(col_a, "corpus_a", "A1", ALPHA)
    _indexer(col_b, "corpus_b", "B1", BETA)
    retrieval.reinitialiser_catalogue()
    return col_a, col_b


# ===========================================================================
# CatalogueDocuments — isolation
# ===========================================================================


def test_catalogue_corpus_a_ne_voit_pas_corpus_b(deux_corpus):
    cat_a = retrieval.catalogue(corpus_id="corpus_a", forcer=True)
    assert {f.document_id for f in cat_a.fiches} == {"A1"}


def test_catalogue_corpus_b_ne_voit_pas_corpus_a(deux_corpus):
    cat_b = retrieval.catalogue(corpus_id="corpus_b", forcer=True)
    assert {f.document_id for f in cat_b.fiches} == {"B1"}


def test_catalogue_meme_nom_de_fichier_resout_le_bon_document_par_corpus(deux_corpus):
    """§13 — même nom ("rapport.pdf") dans A et B : chaque corpus résout
    UNIQUEMENT son propre document, aucune ambiguïté globale."""
    cat_a = retrieval.catalogue(corpus_id="corpus_a", forcer=True)
    cat_b = retrieval.catalogue(corpus_id="corpus_b", forcer=True)

    fiche_a = cat_a.par_identifiant("rapport.pdf")
    fiche_b = cat_b.par_identifiant("rapport.pdf")

    assert fiche_a is not None and fiche_a.document_id == "A1"
    assert fiche_b is not None and fiche_b.document_id == "B1"


def test_catalogue_cache_bien_isole_par_corpus(deux_corpus):
    """Le cache module-level (`_catalogues`) ne doit jamais renvoyer le
    catalogue d'un autre corpus, même sans `forcer=True`."""
    cat_a = retrieval.catalogue(corpus_id="corpus_a")
    cat_b = retrieval.catalogue(corpus_id="corpus_b")
    assert cat_a is not cat_b
    assert {f.document_id for f in cat_a.fiches} == {"A1"}
    assert {f.document_id for f in cat_b.fiches} == {"B1"}


# ===========================================================================
# charger_document — isolation
# ===========================================================================


def test_charger_document_corpus_a_charge_bien_alpha(deux_corpus):
    passages = retrieval.charger_document("A1", corpus_id="corpus_a")
    assert len(passages) == 1
    assert ALPHA in passages[0].texte


def test_charger_document_ne_peut_jamais_charger_un_doc_id_dun_autre_corpus(deux_corpus):
    """doc_id="B1" existe réellement, mais seulement dans corpus_b : invisible
    depuis corpus_a (collection distincte)."""
    with pytest.raises(DocumentInconnu):
        retrieval.charger_document("B1", corpus_id="corpus_a")

    with pytest.raises(DocumentInconnu):
        retrieval.charger_document("A1", corpus_id="corpus_b")


def test_charger_document_corpus_b_charge_bien_beta(deux_corpus):
    passages = retrieval.charger_document("B1", corpus_id="corpus_b")
    assert len(passages) == 1
    assert BETA in passages[0].texte


# ===========================================================================
# rechercher_passages (SEARCH) — isolation des requêtes
# ===========================================================================


@pytest.fixture(autouse=True)
def _sans_embedding_reel(monkeypatch):
    """Encodage/reranking factices : aucun modèle réel chargé. Le filtrage
    testé ici porte sur le payload (doc_id/corpus/collection), jamais sur la
    pertinence sémantique."""
    monkeypatch.setattr(
        retrieval, "encoder_requete", lambda texte, avec_sparse=True: ([0.1], VecteurSparse([], []))
    )
    monkeypatch.setattr(
        retrieval, "reranker", lambda requete, documents, top_k=None: [
            (i, 1.0) for i in range(len(documents))
        ]
    )


def test_recherche_corpus_a_ne_recupere_jamais_beta(deux_corpus):
    rapport = retrieval.rechercher_passages("question", corpus_id="corpus_a")
    textes = " ".join(p.texte for p in rapport.passages)
    assert ALPHA in textes
    assert BETA not in textes


def test_recherche_corpus_b_ne_recupere_jamais_alpha(deux_corpus):
    rapport = retrieval.rechercher_passages("question", corpus_id="corpus_b")
    textes = " ".join(p.texte for p in rapport.passages)
    assert BETA in textes
    assert ALPHA not in textes


def test_recherche_corpus_defaut_est_isolee_des_deux(deux_corpus, faux_qdrant):
    """Le corpus "default" (collection historique) reste vide : aucune fuite
    depuis corpus_a/corpus_b, qui vivent dans des collections dédiées."""
    from src.rag.retrieval import CollectionIndisponible

    with pytest.raises(CollectionIndisponible):
        retrieval.rechercher_passages("question", corpus_id="default")


# ===========================================================================
# Corpus invalide / inconnu
# ===========================================================================


def test_corpus_id_invalide_leve_une_erreur_propre(faux_qdrant, monkeypatch):
    from src.config import ErreurCorpusInvalide

    cabler_registre_corpus(monkeypatch, _REGISTRE)
    with pytest.raises(ErreurCorpusInvalide):
        retrieval.catalogue(corpus_id="../../etc/passwd")


def test_corpus_id_inconnu_leve_une_erreur_propre(faux_qdrant, monkeypatch):
    from src.config import CorpusInconnu

    cabler_registre_corpus(monkeypatch, _REGISTRE)
    with pytest.raises(CorpusInconnu):
        retrieval.catalogue(corpus_id="rh")


# ===========================================================================
# Profil par corpus
# ===========================================================================


def test_profil_resolu_par_corpus_jamais_de_mutation_globale(faux_qdrant, monkeypatch):
    """§8 — le profil est résolu depuis le contexte du corpus, jamais depuis
    une variable globale mutée. Deux corpus avec des profils différents
    obtiennent chacun le leur, sans effet de bord l'un sur l'autre."""
    registre = {
        "default": ConfigCorpus(),
        "finance": ConfigCorpus(profil="generic", profil_domaine="finance", source="local"),
        "rh": ConfigCorpus(profil="generic", profil_domaine="rh", source="local"),
    }
    cabler_registre_corpus(monkeypatch, registre)

    from src.rag.corpus import resoudre_corpus

    ctx_finance = resoudre_corpus("finance")
    ctx_rh = resoudre_corpus("rh")

    assert ctx_finance.profil_domaine_nom == "finance"
    assert ctx_rh.profil_domaine_nom == "rh"
    # Résoudre "rh" n'a pas altéré le contexte déjà résolu pour "finance".
    assert ctx_finance.profil_domaine_nom == "finance"
    assert ctx_finance.nom_collection != ctx_rh.nom_collection
