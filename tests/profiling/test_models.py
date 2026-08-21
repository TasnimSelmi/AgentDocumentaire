"""Tests du modèle DomainProfile."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.profiling.models import DomainProfile, normaliser_nom_profil


def _profil(**surcharges) -> DomainProfile:
    donnees = {
        "profile_name": "domaine_test",
        "domain": "Domaine de test",
        "description": "Description générique du domaine de test.",
        "keywords": ["concept a", "concept b", "concept c"],
    }
    donnees.update(surcharges)
    return DomainProfile(**donnees)


def test_profil_valide():
    profil = _profil()
    assert profil.profile_name == "domaine_test"
    assert profil.output_language == "fr"
    assert len(profil.keywords) == 3


@pytest.mark.parametrize(
    ("saisi", "attendu"),
    [
        ("Finance", "finance"),
        ("  RESSOURCES-HUMAINES  ", "ressources-humaines"),
        ("juridique tunisien", "juridique_tunisien"),
        ("Juridique_Tunisién", "juridique_tunisien"),
    ],
)
def test_normalisation_du_nom(saisi, attendu):
    assert _profil(profile_name=saisi).profile_name == attendu


@pytest.mark.parametrize("nom", ["", "   ", "..", ".", "../finance", "finance/test",
                                 "finance\\test", "C:/finance", "finance!", "-finance"])
def test_noms_refuses(nom):
    with pytest.raises(ValidationError):
        _profil(profile_name=nom)


def test_traversee_repertoire_refusee_directement():
    with pytest.raises(ValueError, match="chemin"):
        normaliser_nom_profil("../../etc/passwd")


def test_mots_cles_vides_supprimes():
    profil = _profil(keywords=["audit", "  ", "", "bilan", "résultat", "   "])
    assert profil.keywords == ["audit", "bilan", "résultat"]


def test_mots_cles_dedupliques_insensible_casse():
    profil = _profil(keywords=["Audit", "audit", "AUDIT", "bilan", "Bilan", "résultat"])
    assert profil.keywords == ["Audit", "bilan", "résultat"]


def test_trop_peu_de_mots_cles():
    with pytest.raises(ValidationError):
        _profil(keywords=["a", "b"])


def test_trop_de_mots_cles():
    with pytest.raises(ValidationError):
        _profil(keywords=[f"concept_{i}" for i in range(31)])


def test_accents_conserves_dans_les_textes():
    profil = _profil(
        domain="Finance et comptabilité",
        description="États financiers, audit et contrôle interne.",
        keywords=["états financiers", "comptabilité", "contrôle interne"],
    )
    assert profil.domain == "Finance et comptabilité"
    assert "États" in profil.description
    assert "états financiers" in profil.keywords


def test_mots_cles_arabes():
    profil = _profil(
        domain="التأمين",
        description="مجال التأمين والتعويضات.",
        keywords=["تأمين", "عقد", "تعويض"],
        output_language="ar",
    )
    assert profil.keywords == ["تأمين", "عقد", "تعويض"]
    assert profil.output_language == "ar"


@pytest.mark.parametrize(("saisi", "attendu"), [("FR", "fr"), ("En", "en"), ("fr_TN", "fr-tn")])
def test_langue_normalisee(saisi, attendu):
    assert _profil(output_language=saisi).output_language == attendu


@pytest.mark.parametrize("langue", ["", "français", "f", "12"])
def test_langue_invalide(langue):
    with pytest.raises(ValidationError):
        _profil(output_language=langue)


@pytest.mark.parametrize("champ", ["domain", "description"])
def test_champs_obligatoires_non_vides(champ):
    with pytest.raises(ValidationError):
        _profil(**{champ: "   "})


def test_champ_obligatoire_absent():
    with pytest.raises(ValidationError):
        DomainProfile(profile_name="x", domain="Y", keywords=["a", "b", "c"])


def test_champ_inconnu_refuse():
    with pytest.raises(ValidationError):
        _profil(champs_extraction=["montant"])


def test_bloc_contexte_domaine_contient_le_vocabulaire():
    bloc = _profil().bloc_contexte_domaine()
    assert "Domaine de test" in bloc
    assert "concept a" in bloc


def test_ancien_nom_de_methode_absent():
    """`bloc_prompt` appartient au profil technique, pas au profil de domaine."""
    assert not hasattr(_profil(), "bloc_prompt")
