"""
Isolation multi-corpus — pipeline d'ingestion réel (`ingerer`).

Aucun Qdrant réel (`faux_qdrant`), aucun embedding réel, aucun LLM réel
(`inferer=False` + `encoder`/`precharger_modeles` doublés) : le VRAI pipeline
`src.rag.ingestion.ingerer` tourne, y compris le calcul réel de `doc_id` /
`point_id`, le vrai `RegistreFichiers`, la vraie logique de suppression des
fichiers absents — c'est exactement le chemin qui portait le bug de
suppression croisée démontré par l'audit multi-corpus (§5).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import ConfigCorpus, get_profil
from src.rag import ingestion, vectorstore
from src.rag.corpus import chemin_registre_pour_corpus, nom_collection_pour_corpus
from tests.rag.conftest import cabler_registre_corpus

_REGISTRE = {
    "default": ConfigCorpus(),
    "corpus_a": ConfigCorpus(profil="generic", source="local"),
    "corpus_b": ConfigCorpus(profil="generic", source="local"),
}


@pytest.fixture(autouse=True)
def _pipeline_sans_ml(monkeypatch):
    """Ingestion réelle, sans aucun modèle chargé : `encoder`/
    `precharger_modeles` doublés par des fonctions triviales et rapides."""
    from src.rag.embeddings import Encodage, VecteurSparse

    def _faux_encoder(textes, avec_sparse=True):
        return Encodage(
            dense=[[0.1] for _ in textes],
            sparse=[VecteurSparse(indices=[], valeurs=[]) for _ in textes],
        )

    monkeypatch.setattr(ingestion, "precharger_modeles", lambda avec_reranker=False: None)
    monkeypatch.setattr(ingestion, "encoder", _faux_encoder)
    monkeypatch.setattr(ingestion, "encoder_dense_seul", lambda textes: [[0.1] for _ in textes])


@pytest.fixture()
def corpus_registre(faux_qdrant, monkeypatch):
    cabler_registre_corpus(monkeypatch, _REGISTRE)
    return faux_qdrant


def _dossier(tmp_path: Path, nom: str) -> Path:
    d = tmp_path / nom
    d.mkdir()
    return d


def _ecrire(dossier: Path, nom_fichier: str, contenu: str) -> Path:
    chemin = dossier / nom_fichier
    chemin.write_text(contenu, encoding="utf-8")
    return chemin


# ===========================================================================
# Corpus « default » — compatibilité ascendante
# ===========================================================================


def test_corpus_default_utilise_la_collection_historique(corpus_registre, tmp_path):
    from src.config import get_config_technique

    dossier = _dossier(tmp_path, "docs_default")
    _ecrire(dossier, "a.txt", "contenu A")

    rapport = ingestion.ingerer(dossier=dossier, inferer=False, corpus_id="default")

    assert rapport.corpus_id == "default"
    assert rapport.fichiers_traites == 1
    nom = nom_collection_pour_corpus("default")
    assert nom == get_config_technique().qdrant.nom_collection
    assert vectorstore.compter(nom_collection=nom) == 1


# ===========================================================================
# Deux corpus coexistants, isolation stricte
# ===========================================================================


def test_deux_corpus_coexistent_dans_des_collections_distinctes(corpus_registre, tmp_path):
    dossier_a = _dossier(tmp_path, "docs_a")
    dossier_b = _dossier(tmp_path, "docs_b")
    _ecrire(dossier_a, "a.txt", "contenu du corpus A")
    _ecrire(dossier_b, "b.txt", "contenu du corpus B")

    rapport_a = ingestion.ingerer(dossier=dossier_a, inferer=False, corpus_id="corpus_a")
    rapport_b = ingestion.ingerer(dossier=dossier_b, inferer=False, corpus_id="corpus_b")

    assert rapport_a.fichiers_traites == 1
    assert rapport_b.fichiers_traites == 1

    col_a = nom_collection_pour_corpus("corpus_a")
    col_b = nom_collection_pour_corpus("corpus_b")
    assert col_a != col_b
    assert vectorstore.compter(nom_collection=col_a) == 1
    assert vectorstore.compter(nom_collection=col_b) == 1


def test_corpus_id_present_dans_le_payload(corpus_registre, tmp_path):
    dossier = _dossier(tmp_path, "docs_a")
    _ecrire(dossier, "a.txt", "contenu du corpus A")
    ingestion.ingerer(dossier=dossier, inferer=False, corpus_id="corpus_a")

    col_a = nom_collection_pour_corpus("corpus_a")
    points = vectorstore.parcourir(limite=10, nom_collection=col_a)
    assert len(points) == 1
    assert points[0].payload["corpus_id"] == "corpus_a"


# ===========================================================================
# §6 — collisions d'identifiants documentaires
# ===========================================================================


def test_meme_nom_fichier_contenu_different_doc_id_distincts(corpus_registre, tmp_path):
    dossier_a = _dossier(tmp_path, "docs_a")
    dossier_b = _dossier(tmp_path, "docs_b")
    _ecrire(dossier_a, "rapport.txt", "contenu A")
    _ecrire(dossier_b, "rapport.txt", "contenu B, totalement différent")

    ingestion.ingerer(dossier=dossier_a, inferer=False, corpus_id="corpus_a")
    ingestion.ingerer(dossier=dossier_b, inferer=False, corpus_id="corpus_b")

    col_a = nom_collection_pour_corpus("corpus_a")
    col_b = nom_collection_pour_corpus("corpus_b")
    doc_id_a = vectorstore.parcourir(limite=10, nom_collection=col_a)[0].payload["doc_id"]
    doc_id_b = vectorstore.parcourir(limite=10, nom_collection=col_b)[0].payload["doc_id"]

    assert doc_id_a != doc_id_b


def test_meme_nom_fichier_contenu_byte_identique_doc_id_distincts(corpus_registre, tmp_path):
    """Cas critique de l'audit (§6) : même chemin relatif, contenu
    STRICTEMENT identique (même hash) dans deux corpus différents ->
    `corpus_id` inclus dans le hash de `doc_id` empêche toute collision,
    donc tout écrasement croisé au niveau des `point_id` Qdrant."""
    dossier_a = _dossier(tmp_path, "docs_a")
    dossier_b = _dossier(tmp_path, "docs_b")
    contenu_identique = "contenu strictement identique, octet pour octet"
    _ecrire(dossier_a, "rapport.txt", contenu_identique)
    _ecrire(dossier_b, "rapport.txt", contenu_identique)

    ingestion.ingerer(dossier=dossier_a, inferer=False, corpus_id="corpus_a")
    ingestion.ingerer(dossier=dossier_b, inferer=False, corpus_id="corpus_b")

    col_a = nom_collection_pour_corpus("corpus_a")
    col_b = nom_collection_pour_corpus("corpus_b")
    point_a = vectorstore.parcourir(limite=10, nom_collection=col_a)[0]
    point_b = vectorstore.parcourir(limite=10, nom_collection=col_b)[0]

    assert point_a.payload["hash_contenu"] == point_b.payload["hash_contenu"]  # même contenu
    assert point_a.payload["doc_id"] != point_b.payload["doc_id"]  # jamais le même doc_id
    assert point_a.point_id != point_b.point_id  # jamais le même point Qdrant

    # Preuve directe sur la fonction de calcul elle-même.
    empreinte = ingestion.empreinte_fichier(dossier_a / "rapport.txt")
    signature = ingestion.calculer_signature_pipeline(get_profil("generic"), False)
    doc_id_a = ingestion.identifiant_version_document(
        chemin=dossier_a / "rapport.txt",
        racine_documents=dossier_a,
        empreinte=empreinte,
        signature_pipeline=signature,
        corpus_id="corpus_a",
    )
    doc_id_b = ingestion.identifiant_version_document(
        chemin=dossier_b / "rapport.txt",
        racine_documents=dossier_b,
        empreinte=empreinte,
        signature_pipeline=signature,
        corpus_id="corpus_b",
    )
    assert doc_id_a != doc_id_b


# ===========================================================================
# §5 — reproduction exacte du bug de suppression croisée, puis preuve du correctif
# ===========================================================================


def test_synchroniser_corpus_b_ne_supprime_jamais_corpus_a(corpus_registre, tmp_path):
    """
    1. indexer corpus A ;
    2. indexer corpus B ;
    3. vérifier que A existe toujours intégralement (comparaison AVANT/APRÈS) ;
    4. supprimer un fichier de B, resynchroniser B ;
    5. vérifier qu'aucun document de A n'est touché.
    """
    dossier_a = _dossier(tmp_path, "docs_a")
    dossier_b = _dossier(tmp_path, "docs_b")
    _ecrire(dossier_a, "a1.txt", "document A1")
    _ecrire(dossier_a, "a2.txt", "document A2")
    fichier_b1 = _ecrire(dossier_b, "b1.txt", "document B1")
    _ecrire(dossier_b, "b2.txt", "document B2")

    ingestion.ingerer(dossier=dossier_a, inferer=False, corpus_id="corpus_a")

    col_a = nom_collection_pour_corpus("corpus_a")
    assert vectorstore.compter(nom_collection=col_a) == 2
    registre_a_avant = chemin_registre_pour_corpus("corpus_a").read_text(encoding="utf-8")

    # 2. Indexer un corpus B totalement distinct.
    ingestion.ingerer(dossier=dossier_b, inferer=False, corpus_id="corpus_b")

    # 3. A doit être intact : même nombre de points, même registre.
    assert vectorstore.compter(nom_collection=col_a) == 2
    assert chemin_registre_pour_corpus("corpus_a").read_text(encoding="utf-8") == registre_a_avant

    # 4. Supprimer un fichier de B, resynchroniser B.
    fichier_b1.unlink()
    rapport_b2 = ingestion.ingerer(dossier=dossier_b, inferer=False, corpus_id="corpus_b")
    assert rapport_b2.fichiers_supprimes == 1

    col_b = nom_collection_pour_corpus("corpus_b")
    assert vectorstore.compter(nom_collection=col_b) == 1  # seul b2 reste

    # 5. A reste totalement intact.
    assert vectorstore.compter(nom_collection=col_a) == 2
    assert chemin_registre_pour_corpus("corpus_a").read_text(encoding="utf-8") == registre_a_avant


def test_registre_de_a_et_de_b_sont_des_fichiers_distincts(corpus_registre):
    chemin_a = chemin_registre_pour_corpus("corpus_a")
    chemin_b = chemin_registre_pour_corpus("corpus_b")
    chemin_default = chemin_registre_pour_corpus("default")
    assert chemin_a != chemin_b != chemin_default


# ===========================================================================
# §11 — reinitialiser scoping
# ===========================================================================


def test_reinitialiser_corpus_a_ne_touche_jamais_corpus_b(corpus_registre, tmp_path):
    dossier_a = _dossier(tmp_path, "docs_a")
    dossier_b = _dossier(tmp_path, "docs_b")
    _ecrire(dossier_a, "a.txt", "document A")
    _ecrire(dossier_b, "b.txt", "document B")

    ingestion.ingerer(dossier=dossier_a, inferer=False, corpus_id="corpus_a")
    ingestion.ingerer(dossier=dossier_b, inferer=False, corpus_id="corpus_b")

    col_a = nom_collection_pour_corpus("corpus_a")
    col_b = nom_collection_pour_corpus("corpus_b")
    registre_b_avant = chemin_registre_pour_corpus("corpus_b").read_text(encoding="utf-8")

    # reinitialiser=True sur corpus_a UNIQUEMENT.
    rapport = ingestion.ingerer(dossier=dossier_a, inferer=False, corpus_id="corpus_a", reinitialiser=True)

    assert rapport.fichiers_traites == 1
    assert vectorstore.compter(nom_collection=col_a) == 1  # recréé, réingéré

    # corpus_b : collection, registre et requête restent intacts.
    assert vectorstore.compter(nom_collection=col_b) == 1
    assert chemin_registre_pour_corpus("corpus_b").read_text(encoding="utf-8") == registre_b_avant
    points_b = vectorstore.parcourir(limite=10, nom_collection=col_b)
    assert points_b[0].payload["corpus_id"] == "corpus_b"
