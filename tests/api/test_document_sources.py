"""
API des sources documentaires : `GET /sources`,
`POST /corpora/{corpus_id}/upload`, `POST /corpora/{corpus_id}/import-url`,
et le verrouillage de la source déclarée sur `POST /ingestion`.

Aucun Qdrant réel sauf pour les tests d'isolation A/B (`faux_qdrant`), aucun
réseau réel pour l'import URL (serveur HTTP local contrôlé, comme
`tests/api/test_url_import.py`).
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from src.config import ConfigCorpus
from src.rag import retrieval, vectorstore
from tests.rag.conftest import cabler_registre_corpus, cabler_registre_corpus_inscriptible, faux_qdrant

__all__ = ["faux_qdrant"]


# ===========================================================================
# GET /sources
# ===========================================================================


def test_sources_liste_local_et_managed(build_client):
    from src.api.dependencies import registre_sources_par_defaut

    reponse = build_client(sources=registre_sources_par_defaut()).get("/sources")
    assert reponse.status_code == 200
    ids = {s["source_id"] for s in reponse.json()}
    assert ids == {"local", "managed"}
    for s in reponse.json():
        assert s["available"] is True
        assert "name" in s and "type" in s


def test_sources_naucune_fuite_de_config(client):
    corps = str(client.get("/sources").json())
    for interdit in ("qdrant", "api_key", "password", "secret", "token", "documents_dir", "/home"):
        assert interdit not in corps.lower()


def test_sources_registre_personnalise_est_reflete(build_client):
    """Preuve du point d'extension : une source enregistrée par l'app
    apparaît, une source non enregistrée n'apparaît jamais — sans qu'aucun
    connecteur ne soit inventé côté route."""
    registre = {"local": lambda corpus_id: None, "ged-insy2s": lambda corpus_id: None}
    reponse = build_client(sources=registre).get("/sources")
    ids = {s["source_id"] for s in reponse.json()}
    assert ids == {"local", "ged-insy2s"}
    assert "managed" not in ids


# ===========================================================================
# POST /corpora/{corpus_id}/upload
# ===========================================================================


def test_upload_corpus_inconnu_est_404(build_client, faux_qdrant, monkeypatch):
    cabler_registre_corpus(monkeypatch, {"default": ConfigCorpus()})
    reponse = build_client().post(
        "/corpora/rh/upload",
        files=[("files", ("a.pdf", b"contenu", "application/pdf"))],
        data={"paths": ["a.pdf"]},
    )
    assert reponse.status_code == 404


def test_upload_corpus_id_invalide_est_422(build_client, faux_qdrant):
    reponse = build_client().post(
        "/corpora/Invalide%20Espace/upload",
        files=[("files", ("a.pdf", b"contenu", "application/pdf"))],
        data={"paths": ["a.pdf"]},
    )
    assert reponse.status_code == 422


def test_upload_corpus_non_managed_est_422(build_client, faux_qdrant, monkeypatch):
    cabler_registre_corpus(
        monkeypatch, {"default": ConfigCorpus(), "rh": ConfigCorpus(source="local")}
    )
    reponse = build_client().post(
        "/corpora/rh/upload",
        files=[("files", ("a.pdf", b"contenu", "application/pdf"))],
        data={"paths": ["a.pdf"]},
    )
    assert reponse.status_code == 422


def test_upload_pdf_txt_valides_sont_ecrits(build_client, faux_qdrant, monkeypatch):
    from src.rag.corpus import dossier_managed_pour_corpus

    cabler_registre_corpus(monkeypatch, {"default": ConfigCorpus(), "rh": ConfigCorpus(source="managed")})
    reponse = build_client().post(
        "/corpora/rh/upload",
        files=[
            ("files", ("a.pdf", b"%PDF-1.4 contenu", "application/pdf")),
            ("files", ("b.txt", b"contenu texte", "text/plain")),
        ],
        data={"paths": ["a.pdf", "sous/b.txt"]},
    )
    assert reponse.status_code == 200
    assert reponse.json()["files_received"] == 2

    racine = dossier_managed_pour_corpus("rh")
    assert (racine / "a.pdf").exists()
    assert (racine / "sous" / "b.txt").exists()


def test_upload_extension_interdite_est_422_rien_ecrit(build_client, faux_qdrant, monkeypatch):
    from src.rag.corpus import dossier_managed_pour_corpus

    cabler_registre_corpus(monkeypatch, {"default": ConfigCorpus(), "rh": ConfigCorpus(source="managed")})
    reponse = build_client().post(
        "/corpora/rh/upload",
        files=[("files", ("virus.exe", b"MZ", "application/octet-stream"))],
        data={"paths": ["virus.exe"]},
    )
    assert reponse.status_code == 422
    assert not dossier_managed_pour_corpus("rh").exists()


def test_upload_fichier_vide_est_422(build_client, faux_qdrant, monkeypatch):
    cabler_registre_corpus(monkeypatch, {"default": ConfigCorpus(), "rh": ConfigCorpus(source="managed")})
    reponse = build_client().post(
        "/corpora/rh/upload",
        files=[("files", ("a.pdf", b"", "application/pdf"))],
        data={"paths": ["a.pdf"]},
    )
    assert reponse.status_code == 422


def test_upload_traversee_repertoire_est_422_rien_ecrit(build_client, faux_qdrant, monkeypatch):
    from src.rag.corpus import dossier_managed_pour_corpus

    cabler_registre_corpus(monkeypatch, {"default": ConfigCorpus(), "rh": ConfigCorpus(source="managed")})
    reponse = build_client().post(
        "/corpora/rh/upload",
        files=[
            ("files", ("bon.pdf", b"ok", "application/pdf")),
            ("files", ("evil.pdf", b"mechant", "application/pdf")),
        ],
        data={"paths": ["bon.pdf", "../evil.pdf"]},
    )
    assert reponse.status_code == 422
    assert not dossier_managed_pour_corpus("rh").exists()


def test_upload_chemin_absolu_est_422(build_client, faux_qdrant, monkeypatch):
    cabler_registre_corpus(monkeypatch, {"default": ConfigCorpus(), "rh": ConfigCorpus(source="managed")})
    reponse = build_client().post(
        "/corpora/rh/upload",
        files=[("files", ("a.pdf", b"contenu", "application/pdf"))],
        data={"paths": ["/etc/passwd"]},
    )
    assert reponse.status_code == 422


def test_upload_nom_unicode_est_accepte(build_client, faux_qdrant, monkeypatch):
    from src.rag.corpus import dossier_managed_pour_corpus

    cabler_registre_corpus(monkeypatch, {"default": ConfigCorpus(), "rh": ConfigCorpus(source="managed")})
    reponse = build_client().post(
        "/corpora/rh/upload",
        files=[("files", ("rapport_été_2026_ééàç.pdf", b"contenu", "application/pdf"))],
        data={"paths": ["rapport_été_2026_ééàç.pdf"]},
    )
    assert reponse.status_code == 200
    assert (dossier_managed_pour_corpus("rh") / "rapport_été_2026_ééàç.pdf").exists()


def test_upload_est_additif_conserve_les_fichiers_precedents(build_client, faux_qdrant, monkeypatch):
    from src.rag.corpus import dossier_managed_pour_corpus

    cabler_registre_corpus(monkeypatch, {"default": ConfigCorpus(), "rh": ConfigCorpus(source="managed")})
    client = build_client()
    client.post(
        "/corpora/rh/upload",
        files=[("files", ("premier.pdf", b"un", "application/pdf"))],
        data={"paths": ["premier.pdf"]},
    )
    client.post(
        "/corpora/rh/upload",
        files=[("files", ("second.pdf", b"deux", "application/pdf"))],
        data={"paths": ["second.pdf"]},
    )
    racine = dossier_managed_pour_corpus("rh")
    assert (racine / "premier.pdf").exists()
    assert (racine / "second.pdf").exists()


def test_upload_treize_pdf_sont_tous_recus(build_client, faux_qdrant, monkeypatch):
    """Cas concret signalé : un dossier de 13 PDF sélectionné via « Parcourir
    un dossier » doit être reçu et écrit intégralement pour un corpus déclaré
    'managed' — aucune limite ni troncature en dessous de MAX_FICHIERS_UPLOAD
    (200)."""
    from src.rag.corpus import dossier_managed_pour_corpus

    cabler_registre_corpus(
        monkeypatch, {"default": ConfigCorpus(), "finance": ConfigCorpus(source="managed")}
    )
    fichiers = [
        ("files", (f"rapport_{i:02d}.pdf", f"%PDF-1.4 contenu {i}".encode(), "application/pdf"))
        for i in range(13)
    ]
    chemins = [f"rapport_{i:02d}.pdf" for i in range(13)]
    reponse = build_client().post(
        "/corpora/finance/upload", files=fichiers, data={"paths": chemins}
    )
    assert reponse.status_code == 200
    assert reponse.json()["files_received"] == 13

    racine = dossier_managed_pour_corpus("finance")
    assert sorted(p.name for p in racine.iterdir()) == sorted(chemins)


def test_upload_aucun_fichier_envoye_est_422(build_client, faux_qdrant, monkeypatch):
    """Sélection annulée / dossier vide côté client : aucun champ `files` du
    tout (distinct du cas « un fichier de contenu vide » déjà couvert par
    `test_upload_fichier_vide_est_422`)."""
    cabler_registre_corpus(monkeypatch, {"default": ConfigCorpus(), "rh": ConfigCorpus(source="managed")})
    reponse = build_client().post("/corpora/rh/upload", data={"paths": []})
    assert reponse.status_code == 422


def test_upload_corpus_a_ne_touche_jamais_corpus_b(build_client, faux_qdrant, monkeypatch):
    from src.rag.corpus import dossier_managed_pour_corpus

    cabler_registre_corpus(
        monkeypatch,
        {
            "default": ConfigCorpus(),
            "corpus_a": ConfigCorpus(source="managed"),
            "corpus_b": ConfigCorpus(source="managed"),
        },
    )
    client = build_client()
    client.post(
        "/corpora/corpus_b/upload",
        files=[("files", ("b.pdf", b"B", "application/pdf"))],
        data={"paths": ["b.pdf"]},
    )
    client.post(
        "/corpora/corpus_a/upload",
        files=[("files", ("a.pdf", b"A", "application/pdf"))],
        data={"paths": ["a.pdf"]},
    )
    assert sorted(p.name for p in dossier_managed_pour_corpus("corpus_a").iterdir()) == ["a.pdf"]
    assert sorted(p.name for p in dossier_managed_pour_corpus("corpus_b").iterdir()) == ["b.pdf"]


# ===========================================================================
# POST /corpora/{corpus_id}/import-url
# ===========================================================================


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/doc.pdf":
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.end_headers()
            self.wfile.write(b"%PDF-1.4 contenu")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


@pytest.fixture()
def serveur_local():
    serveur = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=serveur.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{serveur.server_port}"
    serveur.shutdown()
    thread.join(timeout=2)


@pytest.fixture()
def sans_ssrf(monkeypatch):
    monkeypatch.setattr("src.api.url_import._hote_autorise", lambda hote: None)


def test_import_url_reussi(build_client, faux_qdrant, monkeypatch, serveur_local, sans_ssrf):
    from src.rag.corpus import dossier_managed_pour_corpus

    cabler_registre_corpus(monkeypatch, {"default": ConfigCorpus(), "rh": ConfigCorpus(source="managed")})
    reponse = build_client().post(
        "/corpora/rh/import-url", json={"url": f"{serveur_local}/doc.pdf"}
    )
    assert reponse.status_code == 200
    assert reponse.json()["file_saved"] == "doc.pdf"
    assert (dossier_managed_pour_corpus("rh") / "doc.pdf").exists()


def test_import_url_corpus_non_managed_est_422(
    build_client, faux_qdrant, monkeypatch, serveur_local, sans_ssrf
):
    cabler_registre_corpus(monkeypatch, {"default": ConfigCorpus()})
    reponse = build_client().post(
        "/corpora/default/import-url", json={"url": f"{serveur_local}/doc.pdf"}
    )
    assert reponse.status_code == 422


def test_import_url_corpus_inconnu_est_404(build_client, faux_qdrant, serveur_local, sans_ssrf):
    reponse = build_client().post(
        "/corpora/rh/import-url", json={"url": f"{serveur_local}/doc.pdf"}
    )
    assert reponse.status_code == 404


def test_import_url_schema_file_est_422(build_client, faux_qdrant, monkeypatch):
    cabler_registre_corpus(monkeypatch, {"default": ConfigCorpus(), "rh": ConfigCorpus(source="managed")})
    reponse = build_client().post(
        "/corpora/rh/import-url", json={"url": "file:///etc/passwd"}
    )
    assert reponse.status_code == 422


def test_import_url_localhost_est_422(build_client, faux_qdrant, monkeypatch):
    cabler_registre_corpus(monkeypatch, {"default": ConfigCorpus(), "rh": ConfigCorpus(source="managed")})
    reponse = build_client().post(
        "/corpora/rh/import-url", json={"url": "http://127.0.0.1/doc.pdf"}
    )
    assert reponse.status_code == 422
    corps = str(reponse.json())
    for interdit in ("Traceback", "requests.exceptions", "ConnectionError"):
        assert interdit not in corps


def test_import_url_naffecte_jamais_un_autre_corpus(
    build_client, faux_qdrant, monkeypatch, serveur_local, sans_ssrf
):
    from src.rag.corpus import dossier_managed_pour_corpus

    cabler_registre_corpus(
        monkeypatch,
        {
            "default": ConfigCorpus(),
            "corpus_a": ConfigCorpus(source="managed"),
            "corpus_b": ConfigCorpus(source="managed"),
        },
    )
    build_client().post("/corpora/corpus_a/import-url", json={"url": f"{serveur_local}/doc.pdf"})
    assert (dossier_managed_pour_corpus("corpus_a") / "doc.pdf").exists()
    assert not dossier_managed_pour_corpus("corpus_b").exists()


# ===========================================================================
# POST /ingestion — verrouillage de la source déclarée du corpus
# ===========================================================================


def test_ingestion_source_conforme_a_la_declaration_est_acceptee(
    build_client, ingestion_service, faux_qdrant, monkeypatch
):
    cabler_registre_corpus(monkeypatch, {"default": ConfigCorpus(), "rh": ConfigCorpus(source="managed")})
    sources = {"local": lambda corpus_id: None, "managed": lambda corpus_id: None}
    reponse = build_client(ingestion_service=ingestion_service, sources=sources).post(
        "/ingestion", json={"source": "managed", "corpus_id": "rh"}
    )
    assert reponse.status_code == 200
    assert ingestion_service.appels[0]["corpus_id"] == "rh"


def test_ingestion_source_omise_utilise_la_source_declaree(
    build_client, ingestion_service, faux_qdrant, monkeypatch
):
    cabler_registre_corpus(monkeypatch, {"default": ConfigCorpus(), "rh": ConfigCorpus(source="managed")})
    sources = {"local": lambda corpus_id: None, "managed": lambda corpus_id: None}
    reponse = build_client(ingestion_service=ingestion_service, sources=sources).post(
        "/ingestion", json={"corpus_id": "rh"}
    )
    assert reponse.status_code == 200


def test_ingestion_source_incoherente_avec_la_declaration_est_422(
    build_client, ingestion_service, faux_qdrant, monkeypatch
):
    """Le client ne peut jamais faire lire au registre de fichiers d'un
    corpus déclaré 'managed' le contenu du dossier local partagé — ce qui
    ferait passer pour supprimés des documents jamais absents de sa vraie
    source (voir `src.sources.base`)."""
    cabler_registre_corpus(monkeypatch, {"default": ConfigCorpus(), "rh": ConfigCorpus(source="managed")})
    reponse = build_client(ingestion_service=ingestion_service).post(
        "/ingestion", json={"source": "local", "corpus_id": "rh"}
    )
    assert reponse.status_code == 422
    assert ingestion_service.appels == []


# ===========================================================================
# Isolation A/B — upload + déclenchement d'ingestion, sans Qdrant/LLM réels
# (l'ingestion RÉELLE bout en bout — Qdrant, embeddings, LLM — est couverte
# par le script d'E2E dédié, section 12 du chantier, pas par cette suite
# rapide hors ligne)
# ===========================================================================


def test_upload_puis_appel_ingestion_isole_entre_deux_corpus(
    build_client, ingestion_service, faux_qdrant, monkeypatch
):
    """Corpus A et B tous deux en source 'managed' : chaque upload matérialise
    dans SON stockage géré ; chaque `POST /ingestion` reçoit une source dont
    la racine correspond exactement à SON corpus — jamais à l'autre."""
    from src.api.dependencies import registre_sources_par_defaut
    from src.rag.corpus import dossier_managed_pour_corpus

    cabler_registre_corpus(
        monkeypatch,
        {
            "default": ConfigCorpus(),
            "corpus_a": ConfigCorpus(source="managed"),
            "corpus_b": ConfigCorpus(source="managed"),
        },
    )
    client = build_client(
        ingestion_service=ingestion_service, sources=registre_sources_par_defaut()
    )

    client.post(
        "/corpora/corpus_a/upload",
        files=[("files", ("a.txt", b"Contenu ALPHA du corpus A.", "text/plain"))],
        data={"paths": ["a.txt"]},
    )
    client.post(
        "/corpora/corpus_b/upload",
        files=[("files", ("b.txt", b"Contenu BETA du corpus B.", "text/plain"))],
        data={"paths": ["b.txt"]},
    )
    assert (dossier_managed_pour_corpus("corpus_a") / "a.txt").exists()
    assert (dossier_managed_pour_corpus("corpus_b") / "b.txt").exists()
    assert not (dossier_managed_pour_corpus("corpus_a") / "b.txt").exists()
    assert not (dossier_managed_pour_corpus("corpus_b") / "a.txt").exists()

    reponse_a = client.post("/ingestion", json={"corpus_id": "corpus_a"})
    reponse_b = client.post("/ingestion", json={"corpus_id": "corpus_b"})
    assert reponse_a.status_code == 200 and reponse_b.status_code == 200

    source_a = ingestion_service.appels[0]["source"]
    source_b = ingestion_service.appels[1]["source"]
    assert Path(source_a.racine) == dossier_managed_pour_corpus("corpus_a")
    assert Path(source_b.racine) == dossier_managed_pour_corpus("corpus_b")
    assert source_a.racine != source_b.racine
