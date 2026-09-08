"""
`src.api.uploads` — validation pure d'un lot de fichiers uploadés, sans
FastAPI ni HTTP : chemins relatifs, formats, tailles, comptages.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.api.uploads import (
    FichierValide,
    UploadInvalide,
    ecrire_lot,
    valider_chemin_relatif,
    valider_lot,
)

EXT = {".pdf", ".txt", ".docx"}


# ===========================================================================
# valider_chemin_relatif
# ===========================================================================


@pytest.mark.parametrize(
    "chemin",
    ["a.pdf", "sous/dossier/a.pdf", "a b c.pdf", "éàü.pdf", "sous\\dossier\\a.pdf"],
)
def test_chemin_relatif_valide_est_accepte(chemin):
    resultat = valider_chemin_relatif(chemin)
    assert isinstance(resultat, Path)
    assert not resultat.is_absolute()


@pytest.mark.parametrize(
    "chemin",
    [
        "",
        "   ",
        "/etc/passwd",
        "../evil.pdf",
        "a/../../evil.pdf",
        "..",
        "C:/Users/x/a.pdf",
        "C:\\Users\\x\\a.pdf",
    ],
)
def test_chemin_relatif_invalide_est_refuse(chemin):
    with pytest.raises(UploadInvalide):
        valider_chemin_relatif(chemin)


# ===========================================================================
# valider_lot
# ===========================================================================


def test_lot_valide_est_accepte():
    valides = valider_lot(
        [("a.pdf", b"contenu"), ("sous/b.txt", b"autre contenu")],
        extensions_autorisees=EXT,
        taille_max_fichier_octets=1000,
    )
    assert len(valides) == 2
    assert all(isinstance(v, FichierValide) for v in valides)


def test_lot_vide_est_refuse():
    with pytest.raises(UploadInvalide):
        valider_lot([], extensions_autorisees=EXT, taille_max_fichier_octets=1000)


def test_extension_interdite_est_refusee():
    with pytest.raises(UploadInvalide, match="Format non supporté"):
        valider_lot(
            [("virus.exe", b"x")], extensions_autorisees=EXT, taille_max_fichier_octets=1000
        )


def test_fichier_vide_est_refuse():
    with pytest.raises(UploadInvalide, match="vide"):
        valider_lot([("a.pdf", b"")], extensions_autorisees=EXT, taille_max_fichier_octets=1000)


def test_fichier_trop_volumineux_est_refuse():
    with pytest.raises(UploadInvalide, match="volumineux"):
        valider_lot(
            [("a.pdf", b"x" * 2000)], extensions_autorisees=EXT, taille_max_fichier_octets=1000
        )


def test_taille_totale_trop_importante_est_refusee():
    with pytest.raises(UploadInvalide, match="totale"):
        valider_lot(
            [("a.pdf", b"x" * 600), ("b.pdf", b"x" * 600)],
            extensions_autorisees=EXT,
            taille_max_fichier_octets=1000,
            taille_totale_max_octets=1000,
        )


def test_trop_de_fichiers_est_refuse():
    lot = [(f"{i}.pdf", b"x") for i in range(5)]
    with pytest.raises(UploadInvalide, match="Trop de fichiers"):
        valider_lot(lot, extensions_autorisees=EXT, taille_max_fichier_octets=1000, max_fichiers=3)


def test_chemin_traversee_dans_le_lot_est_refuse_avant_toute_ecriture(tmp_path):
    with pytest.raises(UploadInvalide):
        valider_lot(
            [("bon.pdf", b"ok"), ("../evil.pdf", b"mechant")],
            extensions_autorisees=EXT,
            taille_max_fichier_octets=1000,
        )


def test_nom_duplique_le_second_ecrase_le_premier_dans_le_lot():
    """Deux entrées avec le même chemin relatif : comportement défini (le
    second l'emporte à l'écriture), jamais une exception silencieuse."""
    valides = valider_lot(
        [("a.pdf", b"un"), ("a.pdf", b"deux")],
        extensions_autorisees=EXT,
        taille_max_fichier_octets=1000,
    )
    assert len(valides) == 2


# ===========================================================================
# ecrire_lot — écriture réelle, additive, jamais hors racine
# ===========================================================================


def test_ecrire_lot_preserve_larborescence_relative(tmp_path):
    racine = tmp_path / "corpus_x" / "documents"
    valides = valider_lot(
        [("a.pdf", b"A"), ("sous/dossier/b.pdf", b"B")],
        extensions_autorisees=EXT,
        taille_max_fichier_octets=1000,
    )
    ecrire_lot(valides, racine=racine)

    assert (racine / "a.pdf").read_bytes() == b"A"
    assert (racine / "sous" / "dossier" / "b.pdf").read_bytes() == b"B"


def test_ecrire_lot_est_additif_ne_supprime_rien(tmp_path):
    racine = tmp_path / "corpus_x" / "documents"
    racine.mkdir(parents=True)
    (racine / "ancien.pdf").write_bytes(b"ANCIEN")

    valides = valider_lot(
        [("nouveau.pdf", b"NOUVEAU")], extensions_autorisees=EXT, taille_max_fichier_octets=1000
    )
    ecrire_lot(valides, racine=racine)

    assert (racine / "ancien.pdf").read_bytes() == b"ANCIEN"
    assert (racine / "nouveau.pdf").read_bytes() == b"NOUVEAU"


def test_ecrire_lot_dun_corpus_ne_touche_jamais_un_autre(tmp_path):
    racine_a = tmp_path / "corpus_a" / "documents"
    racine_b = tmp_path / "corpus_b" / "documents"
    racine_b.mkdir(parents=True)
    (racine_b / "b.pdf").write_bytes(b"B")

    valides = valider_lot(
        [("a.pdf", b"A")], extensions_autorisees=EXT, taille_max_fichier_octets=1000
    )
    ecrire_lot(valides, racine=racine_a)

    assert sorted(p.name for p in racine_b.iterdir()) == ["b.pdf"]
    assert sorted(p.name for p in racine_a.iterdir()) == ["a.pdf"]
