"""
Point d'entrée simple destiné au reste du RAG.

Le loader se contente de lire un profil déjà validé sur disque. Il n'appelle
aucun LLM, ne crée aucun profil, ne déclenche aucune ingestion et ne choisit
jamais un profil arbitrairement : sans profil actif configuré, il renvoie
`None` et le RAG continue de fonctionner exactement comme avant.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.profiling.models import DomainProfile
from src.profiling.storage import load_domain_profile

logger = logging.getLogger(__name__)


def nom_profil_actif() -> str | None:
    """Lit le nom du profil de domaine actif dans la configuration."""
    from src.config import get_settings

    valeur = get_settings().active_domain_profile
    if valeur is None:
        return None
    valeur = valeur.strip()
    return valeur or None


def langue_sortie_par_defaut() -> str:
    """
    Langue de sortie par défaut des profils de domaine.

    La valeur vient de `DOMAIN_PROFILE_OUTPUT_LANGUAGE`. Elle est lue à
    l'appel, jamais à l'import, afin que `--help` ne déclenche pas la
    lecture du `.env`.
    """
    from src.config import get_settings

    valeur = (get_settings().domain_profile_output_language or "").strip()
    return valeur or "fr"


def load_active_domain_profile(
    profile_name: str | None = None,
    *,
    dossier: Path | None = None,
) -> DomainProfile | None:
    """
    Charge le profil de domaine à utiliser pour répondre.

    Args:
        profile_name: profil explicite. Si absent, le profil actif défini
            par `ACTIVE_DOMAIN_PROFILE` est utilisé.
        dossier: dossier des profils ; par défaut celui de la configuration.

    Returns:
        Le profil demandé, ou `None` si aucun profil actif n'est configuré.

    Raises:
        DomainProfileNotFoundError: si un profil est explicitement demandé ou
            configuré mais introuvable. Une configuration erronée doit être
            visible, pas silencieusement ignorée.
        DomainProfileStorageError: si le fichier est illisible ou invalide.
    """
    nom = profile_name.strip() if isinstance(profile_name, str) else None
    if not nom:
        actif = nom_profil_actif()
        nom = actif.strip() if isinstance(actif, str) else None

    if not nom:
        logger.debug("Aucun profil de domaine actif : génération sans contexte métier.")
        return None

    return load_domain_profile(nom, dossier)


__all__ = ["load_active_domain_profile", "nom_profil_actif", "langue_sortie_par_defaut"]
