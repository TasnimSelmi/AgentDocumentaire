"""
`LocalDocumentSource` — adaptateur MVP du contrat `DocumentSource`.

Tests entièrement hors ligne : aucun accès Ollama ni Qdrant. Ils vérifient
que la source locale reproduit le comportement historique du pipeline
(découverte identique, dossier rendu tel quel, pass-through) et signale
proprement les erreurs d'accès.
"""

from __future__ import annotations

import pytest

from src.sources import ErreurSource, LocalDocumentSource


def _ecrire(chemin, texte="contenu"):
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(texte, encoding="utf-8")


# ---------------------------------------------------------------------------
# inventaire()
# ---------------------------------------------------------------------------

def test_inventaire_liste_les_documents_supportes(tmp_path):
    _ecrire(tmp_path / "a.txt")
    _ecrire(tmp_path / "sous_dossier" / "b.md")
    _ecrire(tmp_path / "c.pdf")

    inv = LocalDocumentSource(tmp_path).inventaire()

    assert inv == ["a.txt", "c.pdf", "sous_dossier/b.md"]


def test_inventaire_ignore_les_formats_non_supportes(tmp_path):
    _ecrire(tmp_path / "garde.txt")
    _ecrire(tmp_path / "ignore.xyz")
    _ecrire(tmp_path / "image.png")

    assert LocalDocumentSource(tmp_path).inventaire() == ["garde.txt"]


def test_inventaire_source_vide(tmp_path):
    assert LocalDocumentSource(tmp_path).inventaire() == []


def test_inventaire_identifiants_stables_et_relatifs(tmp_path):
    _ecrire(tmp_path / "x.txt")
    _ecrire(tmp_path / "n1" / "n2" / "y.md")

    source = LocalDocumentSource(tmp_path)

    assert source.inventaire() == source.inventaire()  # déterministe
    for source_id in source.inventaire():
        assert not source_id.startswith("/")   # jamais absolu
        assert "\\" not in source_id           # POSIX, indépendant de l'OS


def test_inventaire_restriction_extensions(tmp_path):
    _ecrire(tmp_path / "a.txt")
    _ecrire(tmp_path / "b.md")

    source = LocalDocumentSource(tmp_path, extensions=[".md"])

    assert source.inventaire() == ["b.md"]


# ---------------------------------------------------------------------------
# materialiser() — pass-through
# ---------------------------------------------------------------------------

def test_materialiser_rend_le_dossier_tel_quel(tmp_path):
    _ecrire(tmp_path / "a.txt")
    source = LocalDocumentSource(tmp_path)

    with source.materialiser() as repertoire:
        assert repertoire == tmp_path.resolve()
        # Aucune copie : c'est bien le dossier d'origine.
        assert (repertoire / "a.txt").read_text(encoding="utf-8") == "contenu"


def test_materialiser_ne_cree_ni_ne_supprime_rien(tmp_path):
    _ecrire(tmp_path / "a.txt")
    avant = {p.name for p in tmp_path.iterdir()}

    with LocalDocumentSource(tmp_path).materialiser():
        pass

    assert {p.name for p in tmp_path.iterdir()} == avant


# ---------------------------------------------------------------------------
# Erreurs d'accès — jamais de snapshot partiel exposé
# ---------------------------------------------------------------------------

def test_repertoire_absent_leve_erreur_source(tmp_path):
    source = LocalDocumentSource(tmp_path / "inexistant")

    with pytest.raises(ErreurSource):
        source.inventaire()
    with pytest.raises(ErreurSource):
        with source.materialiser():
            pass


def test_chemin_vers_un_fichier_leve_erreur_source(tmp_path):
    fichier = tmp_path / "pas_un_dossier.txt"
    _ecrire(fichier)
    source = LocalDocumentSource(fichier)

    with pytest.raises(ErreurSource):
        source.inventaire()
    with pytest.raises(ErreurSource):
        with source.materialiser():
            pass


def test_conformite_au_protocole():
    from src.sources.base import DocumentSource

    assert isinstance(LocalDocumentSource("."), DocumentSource)
