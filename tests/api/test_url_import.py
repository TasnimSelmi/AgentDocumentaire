"""
`src.api.url_import` — téléchargement d'un document par URL avec protections
SSRF, sans HTTP FastAPI. Un serveur HTTP local (`threading.HTTPServer`) sert
de destination CONTRÔLÉE : aucun test ne dépend d'Internet.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.api.url_import import ImportUrlInvalide, nom_fichier_depuis_url, telecharger

EXT = {".pdf", ".txt"}
CONTENU_PDF = b"%PDF-1.4 contenu de test"


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - nom imposé par BaseHTTPRequestHandler
        if self.path == "/doc.pdf":
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.end_headers()
            self.wfile.write(CONTENU_PDF)
        elif self.path == "/redirect-once.pdf":
            self.send_response(302)
            self.send_header("Location", "/doc.pdf")
            self.end_headers()
        elif self.path == "/redirect-loop.pdf":
            self.send_response(302)
            self.send_header("Location", "/redirect-loop.pdf")
            self.end_headers()
        elif self.path == "/big.pdf":
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.end_headers()
            self.wfile.write(b"X" * 5000)
        elif self.path == "/vide.pdf":
            self.send_response(200)
            self.end_headers()
        elif self.path == "/lent.pdf":
            import time

            time.sleep(2)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(CONTENU_PDF)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):  # silence
        pass


@pytest.fixture()
def serveur_local():
    serveur = HTTPServer(("127.0.0.1", 0), _Handler)
    port = serveur.server_port
    thread = threading.Thread(target=serveur.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    serveur.shutdown()
    thread.join(timeout=2)


@pytest.fixture()
def sans_ssrf(monkeypatch):
    """Désactive la vérification d'hôte pour tester la logique de
    téléchargement contre le serveur de test local (loopback, bloqué par
    construction — voir les tests SSRF dédiés, sans ce fixture)."""
    monkeypatch.setattr("src.api.url_import._hote_autorise", lambda hote: None)


# ===========================================================================
# Nom de fichier
# ===========================================================================


def test_nom_fichier_derive_du_chemin():
    assert nom_fichier_depuis_url("http://x.test/dir/doc.pdf", extensions_autorisees=EXT) == "doc.pdf"


def test_nom_fichier_sans_extension_autorisee_est_refuse():
    with pytest.raises(ImportUrlInvalide, match="Format non supporté"):
        nom_fichier_depuis_url("http://x.test/doc.exe", extensions_autorisees=EXT)


def test_nom_fichier_sans_nom_est_refuse():
    with pytest.raises(ImportUrlInvalide):
        nom_fichier_depuis_url("http://x.test/", extensions_autorisees=EXT)


# ===========================================================================
# Téléchargement — succès (SSRF désactivé, destination = serveur de test)
# ===========================================================================


def test_telechargement_reussi(serveur_local, sans_ssrf):
    nom, contenu = telecharger(f"{serveur_local}/doc.pdf", extensions_autorisees=EXT)
    assert nom == "doc.pdf"
    assert contenu == CONTENU_PDF


def test_redirection_valide_est_suivie(serveur_local, sans_ssrf):
    nom, contenu = telecharger(f"{serveur_local}/redirect-once.pdf", extensions_autorisees=EXT)
    assert contenu == CONTENU_PDF
    assert nom == "redirect-once.pdf"  # dérivé de l'URL demandée, pas de la cible


def test_trop_de_redirections_est_refuse(serveur_local, sans_ssrf):
    with pytest.raises(ImportUrlInvalide, match="redirections"):
        telecharger(f"{serveur_local}/redirect-loop.pdf", extensions_autorisees=EXT)


def test_404_est_refuse(serveur_local, sans_ssrf):
    with pytest.raises(ImportUrlInvalide, match="404"):
        telecharger(f"{serveur_local}/absent.pdf", extensions_autorisees=EXT)


def test_taille_depassee_est_refusee(serveur_local, sans_ssrf):
    with pytest.raises(ImportUrlInvalide, match="taille maximale"):
        telecharger(f"{serveur_local}/big.pdf", extensions_autorisees=EXT, taille_max_octets=1000)


def test_contenu_vide_est_refuse(serveur_local, sans_ssrf):
    with pytest.raises(ImportUrlInvalide, match="vide"):
        telecharger(f"{serveur_local}/vide.pdf", extensions_autorisees=EXT)


def test_timeout_est_requalifie_proprement(serveur_local, sans_ssrf):
    import httpx

    with pytest.raises(ImportUrlInvalide, match="Téléchargement impossible"):
        telecharger(
            f"{serveur_local}/lent.pdf",
            extensions_autorisees=EXT,
            client=httpx.Client(
                follow_redirects=False, timeout=httpx.Timeout(0.05, connect=0.05)
            ),
        )


# ===========================================================================
# SSRF — SANS désactiver la vérification d'hôte
# ===========================================================================


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x.pdf",
        "http://localhost/x.pdf",
        "http://[::1]/x.pdf",
        "http://169.254.169.254/x.pdf",  # metadata endpoint (AWS/GCP-style)
        "http://10.0.0.5/x.pdf",
        "http://192.168.1.5/x.pdf",
    ],
)
def test_destinations_privees_sont_bloquees(url):
    with pytest.raises(ImportUrlInvalide, match="interdite"):
        telecharger(url, extensions_autorisees=EXT)


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "ftp://x.test/doc.pdf", "javascript:alert(1)"],
)
def test_schemas_non_http_sont_refuses(url):
    with pytest.raises(ImportUrlInvalide):
        telecharger(url, extensions_autorisees=EXT)


def test_redirection_vers_destination_interdite_est_bloquee(serveur_local, monkeypatch):
    """SSRF actif (pas de `sans_ssrf`) : le premier saut vers le serveur de
    test passe la résolution DNS réelle de 127.0.0.1... qui EST privée. On
    prouve donc ici le cas symétrique : autoriser le premier hôte mais pas la
    cible de redirection, en substituant une politique qui bloque
    sélectivement `169.254.169.254` (le nom d'hôte de la redirection)."""
    import ipaddress

    import src.api.url_import as url_import

    reel = url_import._hote_autorise

    def _politique(hote: str) -> None:
        if hote == "127.0.0.1":
            return  # autorise le serveur de test lui-même
        reel(hote)  # règle réelle pour tout le reste (bloque 169.254.169.254)

    monkeypatch.setattr(url_import, "_hote_autorise", _politique)

    class _RedirectionInterdite(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/secret")
            self.end_headers()

        def log_message(self, *a):
            pass

    serveur = HTTPServer(("127.0.0.1", 0), _RedirectionInterdite)
    thread = threading.Thread(target=serveur.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(ImportUrlInvalide, match="interdite"):
            telecharger(
                f"http://127.0.0.1:{serveur.server_port}/x.pdf", extensions_autorisees=EXT
            )
    finally:
        serveur.shutdown()
        thread.join(timeout=2)
