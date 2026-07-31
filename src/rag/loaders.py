from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.config import ConfigOCR, Settings, get_config_technique, get_settings

logger = logging.getLogger(__name__)


# ===========================================================================
# Structures de retour
# ===========================================================================

@dataclass
class Page:
    """Une page (ou une feuille de tableur, ou un bloc pour les formats sans pagination)."""
    numero: int
    texte: str


@dataclass
class DocumentCharge:
    """Résultat de l'extraction d'un fichier."""
    chemin: Path
    pages: list[Page]
    ocr_utilise: bool = False
    avertissements: list[str] = field(default_factory=list)

    @property
    def texte_complet(self) -> str:
        return "\n\n".join(p.texte for p in self.pages)

    @property
    def nb_caracteres(self) -> int:
        return sum(len(p.texte) for p in self.pages)

    @property
    def est_vide(self) -> bool:
        return self.nb_caracteres == 0


class ErreurChargement(Exception):
    """Levée quand un fichier ne peut être lu par aucun loader disponible."""


# ===========================================================================
# Loaders par format
# ===========================================================================

def _charger_texte(chemin: Path) -> list[Page]:
    """.txt / .md : lecture directe, un seul bloc."""
    contenu = chemin.read_text(encoding="utf-8", errors="replace")
    return [Page(numero=1, texte=contenu.strip())]


def _charger_pdf(chemin: Path) -> list[Page]:
    """
    PDF : extraction de la couche texte via pdfplumber.
    pdfplumber gère mieux la mise en page multi-colonne que pypdf.
    L'OCR éventuel est décidé plus haut, pas ici.
    """
    import pdfplumber

    pages: list[Page] = []
    with pdfplumber.open(chemin) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            texte = page.extract_text() or ""
            pages.append(Page(numero=i, texte=texte.strip()))
    return pages


def _charger_docx(chemin: Path) -> list[Page]:
    """
    DOCX : paragraphes + tableaux. Word n'a pas de pagination fiable
    hors rendu, donc tout est regroupé en un bloc.
    """
    from docx import Document

    doc = Document(str(chemin))
    morceaux: list[str] = [p.text for p in doc.paragraphs if p.text.strip()]

    for table in doc.tables:
        for ligne in table.rows:
            cellules = [c.text.strip() for c in ligne.cells]
            if any(cellules):
                morceaux.append(" | ".join(cellules))

    return [Page(numero=1, texte="\n".join(morceaux).strip())]


def _charger_xlsx(chemin: Path) -> list[Page]:
    """
    XLSX : une Page par feuille, chaque ligne en Markdown ' | '.
    Le format tabulaire est préservé pour rester lisible par le LLM.
    """
    from openpyxl import load_workbook

    wb = load_workbook(str(chemin), read_only=True, data_only=True)
    pages: list[Page] = []

    for idx, feuille in enumerate(wb.worksheets, start=1):
        lignes: list[str] = []
        for ligne in feuille.iter_rows(values_only=True):
            cellules = [str(c) if c is not None else "" for c in ligne]
            if any(cellules):
                lignes.append(" | ".join(cellules))
        if lignes:
            entete = f"# Feuille : {feuille.title}"
            pages.append(Page(numero=idx, texte=entete + "\n" + "\n".join(lignes)))

    wb.close()
    return pages or [Page(numero=1, texte="")]


def _charger_csv(chemin: Path) -> list[Page]:
    """CSV : lecture via pandas, rendu Markdown."""
    import pandas as pd

    try:
        df = pd.read_csv(chemin, dtype=str, keep_default_na=False)
    except UnicodeDecodeError:
        df = pd.read_csv(chemin, dtype=str, keep_default_na=False, encoding="latin-1")

    lignes = [" | ".join(df.columns)]
    for _, row in df.iterrows():
        lignes.append(" | ".join(str(v) for v in row.values))
    return [Page(numero=1, texte="\n".join(lignes))]


def _charger_pptx(chemin: Path) -> list[Page]:
    """PPTX : une Page par diapositive."""
    from pptx import Presentation

    prs = Presentation(str(chemin))
    pages: list[Page] = []
    for i, slide in enumerate(prs.slides, start=1):
        morceaux = [
            shape.text.strip()
            for shape in slide.shapes
            if shape.has_text_frame and shape.text.strip()
        ]
        pages.append(Page(numero=i, texte="\n".join(morceaux)))
    return pages


def _charger_html(chemin: Path) -> list[Page]:
    """HTML : texte visible seulement, scripts et styles retirés."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(chemin.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for balise in soup(["script", "style", "noscript"]):
        balise.decompose()
    texte = soup.get_text(separator="\n")
    lignes = [l.strip() for l in texte.splitlines() if l.strip()]
    return [Page(numero=1, texte="\n".join(lignes))]


# Table extension -> loader. Ajouter un format = ajouter une ligne.
_LOADERS = {
    ".txt": _charger_texte,
    ".md": _charger_texte,
    ".pdf": _charger_pdf,
    ".docx": _charger_docx,
    ".xlsx": _charger_xlsx,
    ".csv": _charger_csv,
    ".pptx": _charger_pptx,
    ".html": _charger_html,
    ".htm": _charger_html,
}


# ===========================================================================
# OCR (PDF scannés)
# ===========================================================================

def _configurer_tesseract(settings: Settings) -> None:
    """Renseigne le chemin du binaire si fourni (nécessaire sous Windows)."""
    if settings.tesseract_cmd:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd


def _ocr_pdf(chemin: Path, cfg_ocr: ConfigOCR, settings: Settings) -> list[Page]:
    """
    Rendu de chaque page en image puis OCR Tesseract.
    Coûteux : plafonné par cfg_ocr.pages_max.
    """
    import pytesseract
    from pdf2image import convert_from_path

    _configurer_tesseract(settings)

    images = convert_from_path(str(chemin), dpi=cfg_ocr.dpi)
    if len(images) > cfg_ocr.pages_max:
        logger.warning(
            "%s : %d pages, OCR limité à %d.",
            chemin.name, len(images), cfg_ocr.pages_max,
        )
        images = images[: cfg_ocr.pages_max]

    pages: list[Page] = []
    for i, image in enumerate(images, start=1):
        texte = pytesseract.image_to_string(image, lang=cfg_ocr.langues)
        pages.append(Page(numero=i, texte=texte.strip()))
    return pages


# ===========================================================================
# Point d'entrée
# ===========================================================================

def charger_document(chemin: Path) -> DocumentCharge:
    """
    Charge un fichier et renvoie son texte, page par page.

    Bascule sur l'OCR si un PDF rend un texte anormalement court
    (couche texte absente = scan). Lève ErreurChargement si le format
    n'est pas géré ou si l'extraction échoue totalement.
    """
    settings = get_settings()
    tech = get_config_technique()
    ext = chemin.suffix.lower()

    loader = _LOADERS.get(ext)
    if loader is None:
        raise ErreurChargement(f"Format non géré : {ext} ({chemin.name})")

    try:
        pages = loader(chemin)
    except Exception as exc:  # noqa: BLE001 — on requalifie en erreur métier
        raise ErreurChargement(f"Échec d'extraction sur {chemin.name} : {exc}") from exc

    doc = DocumentCharge(chemin=chemin, pages=pages)

    # Détection de scan : uniquement pour les PDF, si OCR activé.
    scan_probable = (
        ext == ".pdf"
        and tech.ocr.active
        and settings.ocr_enabled
        and doc.nb_caracteres < tech.ingestion.seuil_texte_vide
    )

    if scan_probable:
        logger.info("%s : texte court (%d car.), OCR…", chemin.name, doc.nb_caracteres)
        try:
            pages_ocr = _ocr_pdf(chemin, tech.ocr, settings)
            doc = DocumentCharge(chemin=chemin, pages=pages_ocr, ocr_utilise=True)
            if doc.est_vide:
                doc.avertissements.append("OCR effectué mais aucun texte récupéré.")
        except Exception as exc:  # noqa: BLE001
            logger.error("OCR échoué sur %s : %s", chemin.name, exc)
            doc.avertissements.append(f"OCR indisponible ou échoué : {exc}")

    if doc.est_vide:
        doc.avertissements.append("Aucun texte extrait — document ignoré à l'indexation.")

    return doc


def formats_supportes() -> list[str]:
    return sorted(_LOADERS.keys())