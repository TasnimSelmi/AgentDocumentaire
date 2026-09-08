"""
`GET /corpora`, `GET /corpora/{corpus_id}`, `POST /corpora`, et exposition du
profilage de domaine (`POST /corpora/{corpus_id}/profile[/validate]`).

Aucun Qdrant réel (`faux_qdrant`), aucun LLM réel (`suggest_domain_profile` /
`save_domain_profile` sont remplacés dans `src.api.routes`) : seul le VRAI
code de `src.rag.corpus` / `src.config` tourne, contre un registre en mémoire
(`cabler_registre_corpus[_inscriptible]`, déjà utilisés par les tests
multi-corpus de `src/rag`).
"""

from __future__ import annotations

import pytest

from src.config import ConfigCorpus
from src.rag import retrieval, vectorstore
from src.rag.embeddings import VecteurSparse
from src.rag.vectorstore import ChunkIndexable
from tests.rag.conftest import (
    cabler_registre_corpus,
    cabler_registre_corpus_inscriptible,
    faux_qdrant,
)

__all__ = ["faux_qdrant"]  # ré-export explicite : fixture importée, pas un import mort


def _indexer_un_point(nom_collection: str, corpus_id: str) -> None:
    vectorstore.creer_collection(nom_collection=nom_collection)
    chunk = ChunkIndexable(
        doc_id="D1",
        chunk_index=0,
        texte="contenu",
        dense=[0.1],
        sparse=VecteurSparse(indices=[], valeurs=[]),
        payload={
            "nom_fichier": "rapport.pdf",
            "source": "rapport.pdf",
            "categorie": "autre",
            "page": 1,
        },
        corpus_id=corpus_id,
    )
    vectorstore.indexer([chunk], nom_collection=nom_collection)


# ===========================================================================
# GET /corpora, GET /corpora/{corpus_id}
# ===========================================================================


def test_liste_corpora_reflete_le_registre(build_client, faux_qdrant, monkeypatch):
    cabler_registre_corpus(
        monkeypatch,
        {"default": ConfigCorpus(), "finance": ConfigCorpus(nom="Finance")},
    )
    reponse = build_client().get("/corpora")
    assert reponse.status_code == 200
    corps = {c["corpus_id"]: c for c in reponse.json()}
    assert set(corps) == {"default", "finance"}
    assert corps["finance"]["nom"] == "Finance"
    assert corps["finance"]["status"] == "indexation_required"
    assert corps["finance"]["document_count"] == 0
    assert corps["finance"]["chunk_count"] == 0


def test_detail_corpus_inconnu_est_404(build_client, faux_qdrant, monkeypatch):
    cabler_registre_corpus(monkeypatch, {"default": ConfigCorpus()})
    reponse = build_client().get("/corpora/rh")
    assert reponse.status_code == 404


def test_detail_corpus_id_mal_forme_est_422(build_client, faux_qdrant):
    reponse = build_client().get("/corpora/Invalide%20Espace")
    assert reponse.status_code == 422


def test_detail_corpus_indexe_est_ready_avec_profil(build_client, faux_qdrant, monkeypatch):
    from src.rag.corpus import nom_collection_pour_corpus

    cabler_registre_corpus(
        monkeypatch, {"default": ConfigCorpus(), "rh": ConfigCorpus(profil_domaine="rh_profil")}
    )
    monkeypatch.setattr(
        "src.profiling.load_domain_profile",
        lambda nom: type("P", (), {"domain": "Ressources humaines"})(),
    )
    _indexer_un_point(nom_collection_pour_corpus("rh"), "rh")
    retrieval.reinitialiser_catalogue()

    reponse = build_client().get("/corpora/rh")
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["status"] == "ready"
    assert corps["document_count"] == 1
    assert corps["chunk_count"] == 1
    assert corps["domain"] == "Ressources humaines"


def test_detail_corpus_indexe_sans_profil_est_profiling_required(
    build_client, faux_qdrant, monkeypatch
):
    from src.rag.corpus import nom_collection_pour_corpus

    cabler_registre_corpus(monkeypatch, {"default": ConfigCorpus(), "rh": ConfigCorpus()})
    _indexer_un_point(nom_collection_pour_corpus("rh"), "rh")
    retrieval.reinitialiser_catalogue()

    reponse = build_client().get("/corpora/rh")
    assert reponse.json()["status"] == "profiling_required"


def test_isolation_a_b_les_compteurs_ne_se_melangent_jamais(
    build_client, faux_qdrant, monkeypatch
):
    from src.rag.corpus import nom_collection_pour_corpus

    cabler_registre_corpus(
        monkeypatch,
        {
            "default": ConfigCorpus(),
            "corpus_a": ConfigCorpus(profil_domaine="p"),
            "corpus_b": ConfigCorpus(profil_domaine="p"),
        },
    )
    monkeypatch.setattr(
        "src.profiling.load_domain_profile",
        lambda nom: type("P", (), {"domain": "d"})(),
    )
    _indexer_un_point(nom_collection_pour_corpus("corpus_a"), "corpus_a")
    retrieval.reinitialiser_catalogue()

    client = build_client()
    a = client.get("/corpora/corpus_a").json()
    b = client.get("/corpora/corpus_b").json()
    assert a["document_count"] == 1 and a["status"] == "ready"
    assert b["document_count"] == 0 and b["status"] == "indexation_required"


# ===========================================================================
# POST /corpora
# ===========================================================================


def test_creation_corpus_est_immediatement_indexation_requise(
    build_client, faux_qdrant, monkeypatch
):
    cabler_registre_corpus_inscriptible(monkeypatch, {"default": ConfigCorpus()})
    reponse = build_client().post(
        "/corpora", json={"corpus_id": "juridique", "name": "Juridique", "source": "local"}
    )
    assert reponse.status_code == 201
    corps = reponse.json()
    assert corps["corpus_id"] == "juridique"
    assert corps["status"] == "indexation_required"
    assert corps["nom"] == "Juridique"


def test_creation_corpus_doublon_est_409(build_client, faux_qdrant, monkeypatch):
    cabler_registre_corpus_inscriptible(monkeypatch, {"default": ConfigCorpus()})
    client = build_client()
    client.post("/corpora", json={"corpus_id": "juridique"})
    reponse = client.post("/corpora", json={"corpus_id": "juridique"})
    assert reponse.status_code == 409


def test_creation_corpus_id_invalide_est_422(build_client, faux_qdrant, monkeypatch):
    cabler_registre_corpus_inscriptible(monkeypatch, {"default": ConfigCorpus()})
    reponse = build_client().post("/corpora", json={"corpus_id": "Invalide Espace"})
    assert reponse.status_code == 422


def test_creation_corpus_ne_cree_aucune_collection_qdrant(build_client, faux_qdrant, monkeypatch):
    """Création logique != indexation : aucune écriture Qdrant à la déclaration."""
    cabler_registre_corpus_inscriptible(monkeypatch, {"default": ConfigCorpus()})
    build_client().post("/corpora", json={"corpus_id": "juridique"})
    from src.rag.corpus import nom_collection_pour_corpus

    infos = vectorstore.info_collection(nom_collection=nom_collection_pour_corpus("juridique"))
    assert not infos.get("existe")


def test_creation_corpus_champ_inconnu_est_422(build_client, faux_qdrant, monkeypatch):
    cabler_registre_corpus_inscriptible(monkeypatch, {"default": ConfigCorpus()})
    reponse = build_client().post(
        "/corpora", json={"corpus_id": "juridique", "collection": "autre"}
    )
    assert reponse.status_code == 422


# ===========================================================================
# POST /ingestion — trace persistée, statut fonctionnel dérivé
# ===========================================================================


def test_ingestion_reussie_persiste_le_statut_ready(
    build_client, ingestion_service, faux_qdrant, monkeypatch
):
    from src.rag.corpus import nom_collection_pour_corpus
    from src.rag.ingestion import RapportIngestion

    cabler_registre_corpus_inscriptible(
        monkeypatch, {"default": ConfigCorpus(), "rh": ConfigCorpus(profil_domaine="p")}
    )
    monkeypatch.setattr(
        "src.profiling.load_domain_profile",
        lambda nom: type("P", (), {"domain": "d"})(),
    )
    ingestion_service.rapport = RapportIngestion(
        profil="generic", fichiers_trouves=2, fichiers_traites=2, chunks_indexes=5
    )
    _indexer_un_point(nom_collection_pour_corpus("rh"), "rh")
    retrieval.reinitialiser_catalogue()

    client = build_client(ingestion_service=ingestion_service)
    r = client.post("/ingestion", json={"source": "local", "corpus_id": "rh"})
    assert r.status_code == 200

    detail = client.get("/corpora/rh").json()
    assert detail["status"] == "ready"
    assert detail["last_sync_at"] is not None


def test_ingestion_partielle_persiste_le_statut_partial(
    build_client, ingestion_service, faux_qdrant, monkeypatch
):
    from src.rag.corpus import nom_collection_pour_corpus
    from src.rag.ingestion import RapportIngestion

    cabler_registre_corpus_inscriptible(
        monkeypatch, {"default": ConfigCorpus(), "rh": ConfigCorpus(profil_domaine="p")}
    )
    monkeypatch.setattr(
        "src.profiling.load_domain_profile",
        lambda nom: type("P", (), {"domain": "d"})(),
    )
    ingestion_service.rapport = RapportIngestion(
        profil="generic", fichiers_trouves=3, fichiers_traites=2, fichiers_en_echec=1, chunks_indexes=4
    )
    _indexer_un_point(nom_collection_pour_corpus("rh"), "rh")
    retrieval.reinitialiser_catalogue()

    client = build_client(ingestion_service=ingestion_service)
    client.post("/ingestion", json={"source": "local", "corpus_id": "rh"})

    assert client.get("/corpora/rh").json()["status"] == "partial"


def test_ingestion_echec_total_persiste_le_statut_error(
    build_client, ingestion_service, faux_qdrant, monkeypatch
):
    from src.rag.corpus import nom_collection_pour_corpus
    from src.rag.ingestion import RapportIngestion

    cabler_registre_corpus_inscriptible(
        monkeypatch, {"default": ConfigCorpus(), "rh": ConfigCorpus(profil_domaine="p")}
    )
    monkeypatch.setattr(
        "src.profiling.load_domain_profile",
        lambda nom: type("P", (), {"domain": "d"})(),
    )
    ingestion_service.rapport = RapportIngestion(
        profil="generic", fichiers_trouves=2, fichiers_traites=0, fichiers_en_echec=2
    )
    _indexer_un_point(nom_collection_pour_corpus("rh"), "rh")
    retrieval.reinitialiser_catalogue()

    client = build_client(ingestion_service=ingestion_service)
    client.post("/ingestion", json={"source": "local", "corpus_id": "rh"})

    assert client.get("/corpora/rh").json()["status"] == "error"


def test_ingestion_source_invalide_est_422_registre_non_touche(
    build_client, ingestion_service, faux_qdrant, monkeypatch
):
    registre = cabler_registre_corpus_inscriptible(monkeypatch, {"default": ConfigCorpus()})
    reponse = build_client(ingestion_service=ingestion_service).post(
        "/ingestion", json={"source": "ged_inconnue", "corpus_id": "default"}
    )
    assert reponse.status_code == 422
    assert registre["default"].derniere_ingestion is None


# ===========================================================================
# Profilage de domaine — POST /corpora/{corpus_id}/profile[/validate]
# ===========================================================================


def test_proposer_profil_corpus_inconnu_est_404(build_client, faux_qdrant, monkeypatch):
    cabler_registre_corpus(monkeypatch, {"default": ConfigCorpus()})
    reponse = build_client().post(
        "/corpora/rh/profile", json={"domain": "Ressources humaines"}
    )
    assert reponse.status_code == 404


def test_proposer_profil_renvoie_la_proposition_reelle_sans_rien_persister(
    build_client, faux_qdrant, monkeypatch
):
    from src.profiling.models import DomainProfile

    cabler_registre_corpus_inscriptible(monkeypatch, {"default": ConfigCorpus(), "rh": ConfigCorpus()})
    monkeypatch.setattr(
        "src.api.routes.suggest_domain_profile",
        lambda domain, lang: DomainProfile(
            profile_name="rh",
            domain=domain,
            description="Profil RH généré.",
            keywords=["contrat", "paie", "conge"],
            output_language=lang,
        ),
    )
    reponse = build_client().post(
        "/corpora/rh/profile", json={"domain": "Ressources humaines"}
    )
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["domain"] == "Ressources humaines"
    assert corps["keywords"] == ["contrat", "paie", "conge"]
    # Rien de persisté par la simple proposition.
    assert build_client().get("/corpora/rh").json()["status"] == "indexation_required"


def test_proposer_profil_echec_generation_est_502(build_client, faux_qdrant, monkeypatch):
    from src.profiling import DomainProfileGenerationError

    cabler_registre_corpus_inscriptible(monkeypatch, {"default": ConfigCorpus(), "rh": ConfigCorpus()})

    def _leve(*_a, **_k):
        raise DomainProfileGenerationError("LLM indisponible")

    monkeypatch.setattr("src.api.routes.suggest_domain_profile", _leve)
    reponse = build_client().post(
        "/corpora/rh/profile", json={"domain": "Ressources humaines"}
    )
    assert reponse.status_code == 502


def test_valider_profil_persiste_et_associe_au_corpus(build_client, faux_qdrant, monkeypatch):
    from src.rag.corpus import nom_collection_pour_corpus

    cabler_registre_corpus_inscriptible(monkeypatch, {"default": ConfigCorpus(), "rh": ConfigCorpus()})
    monkeypatch.setattr("src.api.routes.save_domain_profile", lambda profile, overwrite=False: None)
    _indexer_un_point(nom_collection_pour_corpus("rh"), "rh")
    retrieval.reinitialiser_catalogue()

    reponse = build_client().post(
        "/corpora/rh/profile/validate",
        json={
            "domain": "Ressources humaines",
            "description": "Profil RH validé.",
            "keywords": ["contrat", "paie", "conge"],
        },
    )
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["profile_name"] == "rh"
    assert corps["status"] == "ready"


def test_valider_profil_corpus_inconnu_est_404(build_client, faux_qdrant, monkeypatch):
    cabler_registre_corpus(monkeypatch, {"default": ConfigCorpus()})
    reponse = build_client().post(
        "/corpora/rh/profile/validate",
        json={"domain": "d", "description": "d", "keywords": ["a", "b", "c"]},
    )
    assert reponse.status_code == 404


# ===========================================================================
# DELETE /corpora/{corpus_id}
# ===========================================================================


def test_suppression_corpus_reussie_est_204(build_client, faux_qdrant, monkeypatch):
    registre = cabler_registre_corpus_inscriptible(
        monkeypatch, {"default": ConfigCorpus(), "juridique": ConfigCorpus()}
    )
    reponse = build_client().delete("/corpora/juridique")
    assert reponse.status_code == 204
    assert reponse.content == b""
    assert "juridique" not in registre


def test_suppression_corpus_supprime_la_collection_qdrant(build_client, faux_qdrant, monkeypatch):
    from src.rag.corpus import nom_collection_pour_corpus

    cabler_registre_corpus_inscriptible(
        monkeypatch, {"default": ConfigCorpus(), "juridique": ConfigCorpus()}
    )
    col = nom_collection_pour_corpus("juridique")
    _indexer_un_point(col, "juridique")
    assert vectorstore.info_collection(nom_collection=col)["existe"]

    build_client().delete("/corpora/juridique")

    assert not vectorstore.info_collection(nom_collection=col)["existe"]


def test_suppression_corpus_inconnu_est_404(build_client, faux_qdrant, monkeypatch):
    cabler_registre_corpus_inscriptible(monkeypatch, {"default": ConfigCorpus()})
    reponse = build_client().delete("/corpora/rh")
    assert reponse.status_code == 404


def test_suppression_corpus_id_invalide_est_422(build_client, faux_qdrant):
    reponse = build_client().delete("/corpora/Invalide%20Espace")
    assert reponse.status_code == 422


def test_suppression_corpus_default_est_403(build_client, faux_qdrant, monkeypatch):
    registre = cabler_registre_corpus_inscriptible(monkeypatch, {"default": ConfigCorpus()})
    reponse = build_client().delete("/corpora/default")
    assert reponse.status_code == 403
    assert "default" in registre


def test_suppression_corpus_default_ne_supprime_pas_sa_collection(
    build_client, faux_qdrant, monkeypatch
):
    from src.rag.corpus import nom_collection_pour_corpus

    cabler_registre_corpus_inscriptible(monkeypatch, {"default": ConfigCorpus()})
    col = nom_collection_pour_corpus("default")
    _indexer_un_point(col, "default")

    reponse = build_client().delete("/corpora/default")

    assert reponse.status_code == 403
    assert vectorstore.info_collection(nom_collection=col)["existe"]


def test_suppression_a_naffecte_pas_b(build_client, faux_qdrant, monkeypatch):
    from src.rag.corpus import nom_collection_pour_corpus

    registre = cabler_registre_corpus_inscriptible(
        monkeypatch,
        {"default": ConfigCorpus(), "corpus_a": ConfigCorpus(), "corpus_b": ConfigCorpus()},
    )
    col_a = nom_collection_pour_corpus("corpus_a")
    col_b = nom_collection_pour_corpus("corpus_b")
    _indexer_un_point(col_a, "corpus_a")
    _indexer_un_point(col_b, "corpus_b")
    retrieval.reinitialiser_catalogue()

    client = build_client()
    reponse = client.delete("/corpora/corpus_a")
    assert reponse.status_code == 204

    assert "corpus_b" in registre
    assert "corpus_a" not in registre
    assert vectorstore.info_collection(nom_collection=col_b)["existe"]
    assert not vectorstore.info_collection(nom_collection=col_a)["existe"]
    assert client.get("/corpora/corpus_b").status_code == 200
    assert client.get("/corpora/corpus_a").status_code == 404


def test_liste_corpora_apres_suppression_ne_le_contient_plus(
    build_client, faux_qdrant, monkeypatch
):
    cabler_registre_corpus_inscriptible(
        monkeypatch, {"default": ConfigCorpus(), "juridique": ConfigCorpus()}
    )
    client = build_client()
    assert "juridique" in {c["corpus_id"] for c in client.get("/corpora").json()}

    client.delete("/corpora/juridique")

    assert "juridique" not in {c["corpus_id"] for c in client.get("/corpora").json()}


def test_query_sur_corpus_supprime_est_422_erreur_propre(faux_qdrant, monkeypatch):
    """Après suppression, une requête ciblant ce corpus échoue proprement —
    au même point exact qu'un `corpus_id` jamais déclaré (`AgentService`
    résout le corpus AVANT tout appel LLM/Qdrant : aucun coût réel ici)."""
    from src.agent.service import AgentService
    from src.api import create_app
    from src.observability import NullTraceSink

    cabler_registre_corpus_inscriptible(
        monkeypatch, {"default": ConfigCorpus(), "juridique": ConfigCorpus()}
    )
    app = create_app(agent_service=AgentService(), sink=NullTraceSink())
    from fastapi.testclient import TestClient

    client = TestClient(app, raise_server_exceptions=False)

    assert client.delete("/corpora/juridique").status_code == 204

    reponse = client.post("/query", json={"query": "bonjour", "corpus_id": "juridique"})
    assert reponse.status_code == 422
    assert reponse.json()["error"]["code"] == "corpus_invalide"


def test_suppression_corpus_echec_qdrant_est_500_sans_fuite(
    build_client, faux_qdrant, monkeypatch
):
    cabler_registre_corpus_inscriptible(
        monkeypatch, {"default": ConfigCorpus(), "juridique": ConfigCorpus()}
    )

    def _leve(*_a, **_k):
        raise RuntimeError("qdrant refused at /home/jawher/data/vectordb (token=s3cr3t)")

    monkeypatch.setattr("src.rag.vectorstore.supprimer_collection", _leve)

    reponse = build_client().delete("/corpora/juridique")

    assert reponse.status_code == 500
    corps = reponse.json()
    for sensible in ("qdrant refused", "/home/jawher", "s3cr3t", "RuntimeError"):
        assert sensible not in str(corps)
