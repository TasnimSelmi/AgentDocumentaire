"""
Suppression d'un corpus logique (`src.rag.corpus.supprimer_corpus`).

Aucun Qdrant réel (`faux_qdrant`), aucun LLM réel : la collection, le
registre de fichiers et le profil de domaine sont de VRAIS artefacts
(collection en mémoire via `FauxQdrantClient`, fichiers réels sous des
répertoires temporaires) — seul le VRAI code de suppression tourne contre
eux. `config/corpus.yaml` réel jamais touché (registre en mémoire).
"""

from __future__ import annotations

import pytest

from src.config import ConfigCorpus, CorpusInconnu, CorpusProtege, ErreurCorpusInvalide
from src.rag import retrieval, vectorstore
from src.rag.corpus import (
    CORPUS_DEFAUT,
    chemin_registre_pour_corpus,
    dossier_managed_pour_corpus,
    nom_collection_pour_corpus,
    supprimer_corpus,
)
from src.rag.embeddings import VecteurSparse
from src.rag.vectorstore import ChunkIndexable
from tests.rag.conftest import cabler_registre_corpus_inscriptible, faux_qdrant

__all__ = ["faux_qdrant"]  # ré-export explicite : fixture importée, pas un import mort


def _indexer_un_point(nom_collection: str, corpus_id: str) -> None:
    vectorstore.creer_collection(nom_collection=nom_collection)
    chunk = ChunkIndexable(
        doc_id="D1",
        chunk_index=0,
        texte="contenu",
        dense=[0.1],
        sparse=VecteurSparse(indices=[], valeurs=[]),
        payload={"nom_fichier": "rapport.pdf", "source": "rapport.pdf", "categorie": "autre", "page": 1},
        corpus_id=corpus_id,
    )
    vectorstore.indexer([chunk], nom_collection=nom_collection)


@pytest.fixture()
def deux_corpus(faux_qdrant, monkeypatch):
    registre = cabler_registre_corpus_inscriptible(
        monkeypatch,
        {
            "default": ConfigCorpus(),
            "corpus_a": ConfigCorpus(profil="generic", profil_domaine="corpus_a", source="local"),
            "corpus_b": ConfigCorpus(profil="generic", profil_domaine="corpus_b", source="local"),
        },
    )
    col_a = nom_collection_pour_corpus("corpus_a")
    col_b = nom_collection_pour_corpus("corpus_b")
    _indexer_un_point(col_a, "corpus_a")
    _indexer_un_point(col_b, "corpus_b")
    chemin_registre_pour_corpus("corpus_a").parent.mkdir(parents=True, exist_ok=True)
    chemin_registre_pour_corpus("corpus_a").write_text("{}", encoding="utf-8")
    chemin_registre_pour_corpus("corpus_b").write_text("{}", encoding="utf-8")
    retrieval.reinitialiser_catalogue()
    return registre, col_a, col_b


# ===========================================================================
# Validation / protection
# ===========================================================================


def test_supprimer_corpus_default_est_protege(faux_qdrant, monkeypatch):
    cabler_registre_corpus_inscriptible(monkeypatch, {"default": ConfigCorpus()})
    with pytest.raises(CorpusProtege):
        supprimer_corpus(CORPUS_DEFAUT)


def test_supprimer_corpus_default_ne_supprime_rien(faux_qdrant, monkeypatch):
    registre = cabler_registre_corpus_inscriptible(monkeypatch, {"default": ConfigCorpus()})
    with pytest.raises(CorpusProtege):
        supprimer_corpus("default")
    assert "default" in registre


def test_supprimer_corpus_inconnu_leve_corpus_inconnu(faux_qdrant, monkeypatch):
    cabler_registre_corpus_inscriptible(monkeypatch, {"default": ConfigCorpus()})
    with pytest.raises(CorpusInconnu):
        supprimer_corpus("rh")


def test_supprimer_corpus_id_invalide_leve_erreur_format(faux_qdrant, monkeypatch):
    cabler_registre_corpus_inscriptible(monkeypatch, {"default": ConfigCorpus()})
    with pytest.raises(ErreurCorpusInvalide):
        supprimer_corpus("Invalide Avec Espaces")


# ===========================================================================
# Suppression effective — collection, registre, entrée corpus.yaml
# ===========================================================================


def test_supprime_la_collection_qdrant(deux_corpus):
    registre, col_a, _ = deux_corpus
    assert vectorstore.info_collection(nom_collection=col_a)["existe"]
    supprimer_corpus("corpus_a")
    assert not vectorstore.info_collection(nom_collection=col_a)["existe"]


def test_supprime_le_registre_de_fichiers(deux_corpus):
    registre, col_a, _ = deux_corpus
    chemin_a = chemin_registre_pour_corpus("corpus_a")
    assert chemin_a.exists()
    supprimer_corpus("corpus_a")
    assert not chemin_a.exists()


def test_supprime_le_stockage_gere_upload_import(deux_corpus):
    dossier_a = dossier_managed_pour_corpus("corpus_a")
    dossier_a.mkdir(parents=True, exist_ok=True)
    (dossier_a / "upload.pdf").write_bytes(b"contenu")

    supprimer_corpus("corpus_a")

    assert not dossier_a.exists()
    assert not dossier_a.parent.exists()  # <corpus_id>/ entier, pas seulement documents/


def test_supprime_le_stockage_gere_ninquiete_pas_dun_corpus_sans_upload(deux_corpus):
    """Un corpus qui n'a jamais reçu d'upload/import n'a pas de dossier géré
    — sa suppression ne doit pas échouer pour autant."""
    dossier_a = dossier_managed_pour_corpus("corpus_a")
    assert not dossier_a.exists()
    supprimer_corpus("corpus_a")  # ne doit lever aucune exception


def test_supprimer_a_ne_touche_pas_le_stockage_gere_de_b(deux_corpus):
    dossier_a = dossier_managed_pour_corpus("corpus_a")
    dossier_b = dossier_managed_pour_corpus("corpus_b")
    dossier_a.mkdir(parents=True, exist_ok=True)
    dossier_b.mkdir(parents=True, exist_ok=True)
    (dossier_a / "a.pdf").write_bytes(b"A")
    (dossier_b / "b.pdf").write_bytes(b"B")

    supprimer_corpus("corpus_a")

    assert not dossier_a.exists()
    assert (dossier_b / "b.pdf").exists()


def test_supprime_lentree_du_registre_corpus_yaml(deux_corpus):
    registre, _, _ = deux_corpus
    assert "corpus_a" in registre
    supprimer_corpus("corpus_a")
    assert "corpus_a" not in registre


def test_supprime_le_profil_de_domaine_persiste(faux_qdrant, monkeypatch, tmp_path):
    monkeypatch.setenv("DOMAIN_PROFILES_DIR", str(tmp_path / "profiles"))
    import src.config as config_module

    config_module.get_settings.cache_clear()

    from src.profiling import save_domain_profile
    from src.profiling.models import DomainProfile

    profil = DomainProfile(
        profile_name="corpus_a", domain="Test", description="d", keywords=["a", "b", "c"]
    )
    save_domain_profile(profil)
    chemin_profil = tmp_path / "profiles" / "corpus_a.yaml"
    assert chemin_profil.exists()

    cabler_registre_corpus_inscriptible(
        monkeypatch, {"default": ConfigCorpus(), "corpus_a": ConfigCorpus(profil_domaine="corpus_a")}
    )
    vectorstore.creer_collection(nom_collection=nom_collection_pour_corpus("corpus_a"))

    supprimer_corpus("corpus_a")

    assert not chemin_profil.exists()
    config_module.get_settings.cache_clear()


# ===========================================================================
# Isolation A/B
# ===========================================================================


def test_supprimer_a_laisse_b_totalement_intact(deux_corpus):
    registre, col_a, col_b = deux_corpus
    supprimer_corpus("corpus_a")

    assert vectorstore.info_collection(nom_collection=col_b)["existe"]
    assert chemin_registre_pour_corpus("corpus_b").exists()
    assert "corpus_b" in registre
    assert registre["corpus_b"].profil_domaine == "corpus_b"


def test_supprimer_b_laisse_a_totalement_intact(deux_corpus):
    registre, col_a, col_b = deux_corpus
    supprimer_corpus("corpus_b")

    assert vectorstore.info_collection(nom_collection=col_a)["existe"]
    assert chemin_registre_pour_corpus("corpus_a").exists()
    assert "corpus_a" in registre
