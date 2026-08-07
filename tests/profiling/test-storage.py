"""Tests de la persistance des profils de domaine."""

from __future__ import annotations

import pytest
import yaml

from src.profiling.exceptions import (
    DomainProfileAlreadyExistsError,
    DomainProfileNotFoundError,
    DomainProfileStorageError,
)
from src.profiling.models import DomainProfile
from src.profiling.storage import (
    delete_domain_profile,
    domain_profile_exists,
    list_domain_profiles,
    load_domain_profile,
    save_domain_profile,
)


@pytest.fixture()
def profil() -> DomainProfile:
    return DomainProfile(
        profile_name="finance",
        domain="Finance et comptabilité",
        description="États financiers, audit et normes comptables.",
        keywords=["comptabilité", "états financiers", "audit", "IFRS"],
    )


def test_sauvegarde_cree_le_dossier(tmp_path, profil):
    cible = tmp_path / "profils" / "domaines"
    chemin = save_domain_profile(profil, dossier=cible)

    assert chemin.is_file()
    assert chemin.name == "finance.yaml"
    assert chemin.parent == cible


def test_yaml_lisible_et_unicode(tmp_path, profil):
    chemin = save_domain_profile(profil, dossier=tmp_path)
    contenu = chemin.read_text(encoding="utf-8")

    assert "comptabilité" in contenu  # accents non échappés
    donnees = yaml.safe_load(contenu)
    assert list(donnees)[0] == "profile_name"


def test_aller_retour_identique(tmp_path, profil):
    save_domain_profile(profil, dossier=tmp_path)
    recharge = load_domain_profile("finance", dossier=tmp_path)
    assert recharge == profil


def test_aller_retour_arabe(tmp_path):
    profil = DomainProfile(
        profile_name="assurance",
        domain="التأمين",
        description="مجال التأمين والتعويضات.",
        keywords=["تأمين", "عقد", "تعويض"],
        output_language="ar",
    )
    save_domain_profile(profil, dossier=tmp_path)
    assert load_domain_profile("assurance", dossier=tmp_path) == profil


def test_ecrasement_refuse_par_defaut(tmp_path, profil):
    save_domain_profile(profil, dossier=tmp_path)
    with pytest.raises(DomainProfileAlreadyExistsError):
        save_domain_profile(profil, dossier=tmp_path)


def test_ecrasement_autorise(tmp_path, profil):
    save_domain_profile(profil, dossier=tmp_path)
    modifie = profil.model_copy(update={"domain": "Finance publique"})
    save_domain_profile(modifie, overwrite=True, dossier=tmp_path)

    assert load_domain_profile("finance", dossier=tmp_path).domain == "Finance publique"


def test_profil_absent(tmp_path):
    with pytest.raises(DomainProfileNotFoundError):
        load_domain_profile("inexistant", dossier=tmp_path)


def test_yaml_invalide(tmp_path):
    (tmp_path / "casse.yaml").write_text("profile_name: [", encoding="utf-8")
    with pytest.raises(DomainProfileStorageError):
        load_domain_profile("casse", dossier=tmp_path)


def test_racine_yaml_non_dictionnaire(tmp_path):
    (tmp_path / "liste.yaml").write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(DomainProfileStorageError):
        load_domain_profile("liste", dossier=tmp_path)


def test_incoherence_nom_fichier(tmp_path, profil):
    donnees = profil.model_dump()
    (tmp_path / "autre.yaml").write_text(
        yaml.safe_dump(donnees, allow_unicode=True), encoding="utf-8"
    )
    with pytest.raises(DomainProfileStorageError, match="Incohérence"):
        load_domain_profile("autre", dossier=tmp_path)


def test_suppression(tmp_path, profil):
    save_domain_profile(profil, dossier=tmp_path)
    assert delete_domain_profile("finance", dossier=tmp_path) is True
    assert delete_domain_profile("finance", dossier=tmp_path) is False
    assert domain_profile_exists("finance", dossier=tmp_path) is False


def test_listing_trie_et_tolerant(tmp_path, profil):
    save_domain_profile(profil, dossier=tmp_path)
    save_domain_profile(
        profil.model_copy(update={"profile_name": "aviation"}), dossier=tmp_path
    )
    (tmp_path / "notes.txt").write_text("sans rapport", encoding="utf-8")
    (tmp_path / "sous_dossier").mkdir()
    (tmp_path / "abime.yaml").write_text("profile_name: [", encoding="utf-8")

    noms = [p.profile_name for p in list_domain_profiles(dossier=tmp_path)]
    assert noms == ["aviation", "finance"]


def test_listing_dossier_absent(tmp_path):
    assert list_domain_profiles(dossier=tmp_path / "vide") == []


@pytest.mark.parametrize("nom", ["../evasion", "sous/evasion", "..", "evasion\\x"])
def test_sortie_du_dossier_impossible(tmp_path, nom):
    with pytest.raises(DomainProfileStorageError):
        load_domain_profile(nom, dossier=tmp_path)
    with pytest.raises(DomainProfileStorageError):
        delete_domain_profile(nom, dossier=tmp_path)


def test_aucun_fichier_temporaire_restant(tmp_path, profil):
    save_domain_profile(profil, dossier=tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["finance.yaml"]
