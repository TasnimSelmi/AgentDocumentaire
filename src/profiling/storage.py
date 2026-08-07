"""
Persistance des profils de domaine, au format YAML.

Un profil = un fichier `<profile_name>.yaml` dans le dossier configuré par
`DOMAIN_PROFILES_DIR`. L'écriture est atomique, l'encodage est UTF-8 et les
accents comme l'arabe sont conservés tels quels.

Le dossier cible est un paramètre optionnel de chaque fonction : les tests
peuvent ainsi écrire dans un `tmp_path` sans jamais toucher aux vrais profils.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from src.profiling.exceptions import (
    DomainProfileAlreadyExistsError,
    DomainProfileNotFoundError,
    DomainProfileStorageError,
)
from src.profiling.models import DomainProfile, normaliser_nom_profil

logger = logging.getLogger(__name__)

EXTENSIONS_YAML = (".yaml", ".yml")

# Ordre d'écriture des clés : lisibilité avant tri alphabétique.
ORDRE_CLES = ("profile_name", "domain", "description", "keywords", "output_language")


def dossier_profils(dossier: Path | None = None) -> Path:
    """
    Résout le dossier des profils de domaine.

    Sans argument, la valeur vient de la configuration du projet
    (`DOMAIN_PROFILES_DIR`). L'import de `src.config` est local afin que ce
    module reste importable dans un test isolé.
    """
    if dossier is not None:
        return Path(dossier)

    from src.config import get_settings

    return Path(get_settings().domain_profiles_dir)


def _chemin_profil(profile_name: str, dossier: Path | None = None) -> Path:
    """
    Construit le chemin d'un profil et vérifie qu'il reste dans le dossier.

    Raises:
        DomainProfileStorageError: si le nom est invalide ou sort du dossier.
    """
    try:
        nom = normaliser_nom_profil(profile_name)
    except ValueError as exc:
        raise DomainProfileStorageError(str(exc)) from exc

    base = dossier_profils(dossier)
    chemin = (base / f"{nom}.yaml").resolve()

    # Double garde : même si la validation du nom changeait, aucun chemin
    # hors du dossier de profils ne peut être lu, écrit ou supprimé.
    if chemin.parent != base.resolve():
        raise DomainProfileStorageError(
            f"Chemin de profil hors du dossier autorisé : {chemin}"
        )
    return chemin


def domain_profile_exists(profile_name: str, dossier: Path | None = None) -> bool:
    """Indique si un profil portant ce nom est déjà enregistré."""
    return _chemin_profil(profile_name, dossier).is_file()


def save_domain_profile(
    profile: DomainProfile,
    *,
    overwrite: bool = False,
    dossier: Path | None = None,
) -> Path:
    """
    Écrit un profil de domaine sur disque.

    Args:
        profile: profil validé à enregistrer.
        overwrite: autorise le remplacement d'un profil existant.
        dossier: dossier cible ; par défaut celui de la configuration.

    Returns:
        Le chemin du fichier écrit.

    Raises:
        DomainProfileAlreadyExistsError: si le profil existe et `overwrite` est faux.
        DomainProfileStorageError: en cas d'échec d'écriture.
    """
    chemin = _chemin_profil(profile.profile_name, dossier)

    if chemin.exists() and not overwrite:
        raise DomainProfileAlreadyExistsError(
            f"Le profil '{profile.profile_name}' existe déjà : {chemin}. "
            "Utilise overwrite=True pour le remplacer."
        )

    donnees = profile.model_dump()
    ordonnees: dict[str, Any] = {cle: donnees[cle] for cle in ORDRE_CLES if cle in donnees}

    try:
        chemin.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DomainProfileStorageError(
            f"Impossible de créer le dossier des profils : {chemin.parent} ({exc})"
        ) from exc

    fichier_temporaire: Path | None = None
    try:
        # Écriture atomique : fichier temporaire dans le même dossier, puis
        # remplacement. Un plantage ne laisse jamais un profil à moitié écrit.
        descripteur, nom_temporaire = tempfile.mkstemp(
            dir=chemin.parent, prefix=f".{chemin.stem}.", suffix=".tmp"
        )
        fichier_temporaire = Path(nom_temporaire)
        with os.fdopen(descripteur, "w", encoding="utf-8") as flux:
            yaml.safe_dump(
                ordonnees,
                flux,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
                width=88,
            )
        os.replace(fichier_temporaire, chemin)
        fichier_temporaire = None
    except (OSError, yaml.YAMLError) as exc:
        raise DomainProfileStorageError(
            f"Échec de l'écriture du profil '{profile.profile_name}' : {exc}"
        ) from exc
    finally:
        if fichier_temporaire is not None and fichier_temporaire.exists():
            fichier_temporaire.unlink(missing_ok=True)

    logger.info("Profil de domaine enregistré : %s", chemin)
    return chemin


def _profil_depuis_fichier(chemin: Path, nom_attendu: str | None = None) -> DomainProfile:
    """Lit et valide un fichier de profil."""
    try:
        contenu = chemin.read_text(encoding="utf-8")
        donnees = yaml.safe_load(contenu)
    except (OSError, yaml.YAMLError) as exc:
        raise DomainProfileStorageError(
            f"Fichier de profil illisible : {chemin} ({exc})"
        ) from exc

    if not isinstance(donnees, dict):
        raise DomainProfileStorageError(
            f"Fichier de profil invalide : {chemin}. "
            "La racine du YAML doit être un dictionnaire."
        )

    try:
        profil = DomainProfile.model_validate(donnees)
    except Exception as exc:  # ValidationError et erreurs de validateur
        raise DomainProfileStorageError(
            f"Profil invalide dans {chemin} : {exc}"
        ) from exc

    if nom_attendu is not None and profil.profile_name != nom_attendu:
        raise DomainProfileStorageError(
            f"Incohérence dans {chemin} : le fichier déclare "
            f"profile_name='{profil.profile_name}' alors que le nom demandé "
            f"est '{nom_attendu}'."
        )
    return profil


def load_domain_profile(
    profile_name: str,
    dossier: Path | None = None,
) -> DomainProfile:
    """
    Charge un profil de domaine par son nom.

    Raises:
        DomainProfileNotFoundError: si le fichier n'existe pas.
        DomainProfileStorageError: si le fichier est illisible ou invalide.
    """
    chemin = _chemin_profil(profile_name, dossier)
    if not chemin.is_file():
        raise DomainProfileNotFoundError(
            f"Profil de domaine introuvable : '{profile_name}' ({chemin})."
        )
    return _profil_depuis_fichier(chemin, nom_attendu=chemin.stem)


def list_domain_profiles(dossier: Path | None = None) -> list[DomainProfile]:
    """
    Liste les profils enregistrés, triés par `profile_name`.

    Les fichiers qui ne sont pas des YAML sont ignorés sans bruit. Un YAML
    présent dans le dossier mais illisible ou non conforme est journalisé en
    avertissement et écarté : un fichier abîmé ne doit ni disparaître
    silencieusement, ni empêcher de lister les autres profils.
    """
    base = dossier_profils(dossier)
    if not base.is_dir():
        return []

    profils: list[DomainProfile] = []
    for chemin in sorted(base.iterdir()):
        if not chemin.is_file() or chemin.suffix.lower() not in EXTENSIONS_YAML:
            continue
        try:
            profils.append(_profil_depuis_fichier(chemin, nom_attendu=chemin.stem))
        except DomainProfileStorageError as exc:
            logger.warning("Profil ignoré (%s) : %s", chemin.name, exc)

    return sorted(profils, key=lambda p: p.profile_name)


def delete_domain_profile(profile_name: str, dossier: Path | None = None) -> bool:
    """
    Supprime un profil de domaine.

    Returns:
        True si un fichier a été supprimé, False si aucun profil de ce nom
        n'existait. Aucune exception n'est levée pour un profil absent.

    Raises:
        DomainProfileStorageError: si le nom est invalide, s'il désigne un
            chemin hors du dossier des profils, ou si la suppression échoue.
    """
    chemin = _chemin_profil(profile_name, dossier)
    if not chemin.is_file():
        return False
    try:
        chemin.unlink()
    except OSError as exc:
        raise DomainProfileStorageError(
            f"Échec de la suppression du profil '{profile_name}' : {exc}"
        ) from exc

    logger.info("Profil de domaine supprimé : %s", chemin)
    return True


__all__ = [
    "save_domain_profile",
    "load_domain_profile",
    "list_domain_profiles",
    "delete_domain_profile",
    "domain_profile_exists",
    "dossier_profils",
]
