"""
Tests de l'OCR par lots (`src.rag.loaders._ocr_pdf`) — audit long-documents.

Avant le correctif : un PDF scanné de plus de `pages_max` pages n'était
OCRisé que sur ses `pages_max` premières pages, le reste étant abandonné
silencieusement (seul un `logger.warning`, jamais remonté à l'appelant).

Après : `pages_max` borne la taille d'UN lot de rendu, plus jamais le
document entier. Un PDF de 300 pages avec `pages_max=20` est traité en 15
lots successifs, fusionnés dans l'ordre — aucune page perdue.

Aucun vrai Tesseract/poppler nécessaire : `pdf2image.convert_from_path`,
`pdf2image.pdfinfo_from_path` et `pytesseract.image_to_string` sont
doublés, déterministes.
"""

from __future__ import annotations

from pathlib import Path

import pdf2image
import pytesseract

from src.config import ConfigOCR, Settings
from src.rag.loaders import _ocr_pdf


def _config_ocr(pages_max: int = 20) -> ConfigOCR:
    return ConfigOCR(active=True, langues="fra", dpi=100, pages_max=pages_max)


def _settings() -> Settings:
    return Settings(llm_provider="ollama")


class _ImageFactice:
    """Représente UNE page rendue ; porte juste son numéro pour l'OCR factice."""

    def __init__(self, numero_reel: int) -> None:
        self.numero_reel = numero_reel


def _cabler_pdf_scanne(monkeypatch, *, nb_pages: int, pages_max: int) -> list[int]:
    """
    Câble un PDF scanné factice de `nb_pages` pages : `pdfinfo_from_path`
    renvoie le vrai nombre de pages, `convert_from_path` ne rend QUE la
    plage demandée (`first_page`/`last_page`), l'OCR renvoie un texte
    identifiant le numéro RÉEL de la page.

    Retourne la liste des appels à `convert_from_path` sous forme de
    `(first_page, last_page)`, pour vérifier le découpage en lots.
    """
    appels_convert: list[tuple[int | None, int | None]] = []

    def _fake_pdfinfo(chemin, **kwargs):
        return {"Pages": nb_pages}

    def _fake_convert(chemin, dpi=200, first_page=None, last_page=None, **kwargs):
        appels_convert.append((first_page, last_page))
        debut = first_page if first_page is not None else 1
        fin = last_page if last_page is not None else nb_pages
        return [_ImageFactice(n) for n in range(debut, fin + 1)]

    def _fake_ocr(image, lang=None):
        return f"texte de la page {image.numero_reel}"

    monkeypatch.setattr(pdf2image, "pdfinfo_from_path", _fake_pdfinfo)
    monkeypatch.setattr(pdf2image, "convert_from_path", _fake_convert)
    monkeypatch.setattr(pytesseract, "image_to_string", _fake_ocr)

    return appels_convert


# ===========================================================================
# Invariant central : AUCUNE perte silencieuse de page
# ===========================================================================


def test_pdf_scanne_plus_long_que_pages_max_est_couvert_integralement(monkeypatch):
    """PDF scanné de 300 pages, pages_max=20 : les 300 pages sont OCRisées,
    pas seulement les 20 premières (comportement AVANT le correctif)."""
    appels = _cabler_pdf_scanne(monkeypatch, nb_pages=300, pages_max=20)

    pages = _ocr_pdf(Path("scan_300_pages.pdf"), _config_ocr(pages_max=20), _settings())

    assert len(pages) == 300
    assert [p.numero for p in pages] == list(range(1, 301))
    # La dernière page (celle qui aurait été perdue avant le correctif) est bien présente.
    assert pages[-1].texte == "texte de la page 300"
    assert pages[0].texte == "texte de la page 1"


def test_ocr_traite_par_lots_bornes_par_pages_max(monkeypatch):
    """300 pages, pages_max=20 -> exactement 15 lots de rendu, chacun <= 20 pages."""
    appels = _cabler_pdf_scanne(monkeypatch, nb_pages=300, pages_max=20)

    _ocr_pdf(Path("scan_300_pages.pdf"), _config_ocr(pages_max=20), _settings())

    assert len(appels) == 15
    assert appels[0] == (1, 20)
    assert appels[1] == (21, 40)
    assert appels[-1] == (281, 300)
    for debut, fin in appels:
        assert fin - debut + 1 <= 20


def test_document_extreme_1000_pages_couvert_integralement(monkeypatch):
    """Cas extrême de l'audit : 1000 pages, aucune page perdue, 50 lots."""
    appels = _cabler_pdf_scanne(monkeypatch, nb_pages=1000, pages_max=20)

    pages = _ocr_pdf(Path("scan_1000_pages.pdf"), _config_ocr(pages_max=20), _settings())

    assert len(pages) == 1000
    assert len(appels) == 50
    assert pages[-1].numero == 1000
    assert pages[-1].texte == "texte de la page 1000"


def test_ordre_documentaire_respecte_a_travers_les_lots(monkeypatch):
    _cabler_pdf_scanne(monkeypatch, nb_pages=45, pages_max=20)

    pages = _ocr_pdf(Path("scan.pdf"), _config_ocr(pages_max=20), _settings())

    assert [p.numero for p in pages] == list(range(1, 46))


def test_document_plus_court_que_pages_max_un_seul_lot(monkeypatch):
    """Un document de 5 pages avec pages_max=20 : comportement inchangé, un seul lot."""
    appels = _cabler_pdf_scanne(monkeypatch, nb_pages=5, pages_max=20)

    pages = _ocr_pdf(Path("court.pdf"), _config_ocr(pages_max=20), _settings())

    assert len(pages) == 5
    assert len(appels) == 1
    assert appels[0] == (1, 5)


def test_nombre_de_pages_indisponible_replie_sur_un_rendu_complet(monkeypatch):
    """Si `pdfinfo_from_path` échoue, repli sur un rendu complet en un seul
    lot — jamais de troncature silencieuse non plus dans ce cas de repli."""

    def _pdfinfo_en_echec(chemin, **kwargs):
        raise RuntimeError("poppler indisponible")

    images = [_ImageFactice(n) for n in range(1, 8)]

    def _fake_convert(chemin, dpi=200, first_page=None, last_page=None, **kwargs):
        assert first_page is None and last_page is None
        return images

    monkeypatch.setattr(pdf2image, "pdfinfo_from_path", _pdfinfo_en_echec)
    monkeypatch.setattr(pdf2image, "convert_from_path", _fake_convert)
    monkeypatch.setattr(pytesseract, "image_to_string", lambda image, lang=None: f"page {image.numero_reel}")

    pages = _ocr_pdf(Path("scan.pdf"), _config_ocr(pages_max=3), _settings())

    assert len(pages) == 7
    assert [p.numero for p in pages] == list(range(1, 8))
