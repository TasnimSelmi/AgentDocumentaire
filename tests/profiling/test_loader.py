"""Tests du loader de profil de domaine actif."""

from __future__ import annotations

import pytest

from src.profiling import loader
from src.profiling.exceptions import DomainProfileNotFoundError
from src.profiling.loader import load_active_domain_profile
from src.profiling.models import DomainProfile
from src.profiling.storage import save_domain_profile


@pytest.fixture()
def dossier_profils(tmp_path):
    profil = DomainProfile(
        profile_name="sante",
        domain="Santé",
        description="Domaine médical et hospitalier.",
        keywords=["patient", "diagnostic", "traitement"],
    )
    save_domain_profile(profil, dossier=tmp_path)
    return tmp_path


def _profil_actif(monkeypatch, valeur):
    monkeypatch.setattr(loader, "nom_profil_actif", lambda: valeur)


def test_profil_explicite(monkeypatch, dossier_profils):
    _profil_actif(monkeypatch, None)
    profil = load_active_domain_profile("sante", dossier=dossier_profils)
    assert profil is not None
    assert profil.profile_name == "sante"


def test_profil_actif_configure(monkeypatch, dossier_profils):
    _profil_actif(monkeypatch, "sante")
    profil = load_active_domain_profile(dossier=dossier_profils)
    assert profil is not None
    assert profil.domain == "Santé"


def test_aucun_profil_actif(monkeypatch, dossier_profils):
    _profil_actif(monkeypatch, None)
    assert load_active_domain_profile(dossier=dossier_profils) is None


def test_profil_actif_vide(monkeypatch, dossier_profils):
    _profil_actif(monkeypatch, "   ")
    assert load_active_domain_profile(dossier=dossier_profils) is None


def test_profil_actif_introuvable(monkeypatch, dossier_profils):
    _profil_actif(monkeypatch, "juridique")
    with pytest.raises(DomainProfileNotFoundError):
        load_active_domain_profile(dossier=dossier_profils)


def test_langue_par_defaut_vient_de_la_configuration(monkeypatch):
    class FauxSettings:
        domain_profile_output_language = "en"

    import src.config

    monkeypatch.setattr(src.config, "get_settings", lambda: FauxSettings())
    assert loader.langue_sortie_par_defaut() == "en"


def test_langue_par_defaut_repli_si_configuration_vide(monkeypatch):
    class FauxSettings:
        domain_profile_output_language = "   "

    import src.config

    monkeypatch.setattr(src.config, "get_settings", lambda: FauxSettings())
    assert loader.langue_sortie_par_defaut() == "fr"


def test_le_loader_ne_choisit_pas_un_profil_au_hasard(monkeypatch, dossier_profils):
    """Un profil existe sur disque, mais aucun n'est actif : rien n'est chargé."""
    _profil_actif(monkeypatch, None)
    assert load_active_domain_profile(dossier=dossier_profils) is None
