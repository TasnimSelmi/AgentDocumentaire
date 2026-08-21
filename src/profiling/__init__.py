"""
Profils de domaine.

Un profil de domaine décrit le champ métier du corpus (intitulé, description,
vocabulaire), sans jamais analyser un document. Il est saisi par un
administrateur, éventuellement pré-rempli par le LLM, puis validé et
enregistré en YAML.

Usage :

    from src.profiling import suggest_domain_profile, save_domain_profile

    profil = suggest_domain_profile(domain="Finance et comptabilité")
    chemin = save_domain_profile(profil)
"""

from src.profiling.exceptions import (
    DomainProfileAlreadyExistsError,
    DomainProfileError,
    DomainProfileGenerationError,
    DomainProfileNotFoundError,
    DomainProfileStorageError,
)
from src.profiling.loader import (
    langue_sortie_par_defaut,
    load_active_domain_profile,
)
from src.profiling.models import DomainProfile
from src.profiling.prompts import build_domain_profile_prompt
from src.profiling.service import suggest_domain_profile
from src.profiling.storage import (
    delete_domain_profile,
    domain_profile_exists,
    list_domain_profiles,
    load_domain_profile,
    save_domain_profile,
)

__all__ = [
    "DomainProfile",
    "suggest_domain_profile",
    "build_domain_profile_prompt",
    "save_domain_profile",
    "load_domain_profile",
    "list_domain_profiles",
    "delete_domain_profile",
    "domain_profile_exists",
    "load_active_domain_profile",
    "langue_sortie_par_defaut",
    "DomainProfileError",
    "DomainProfileGenerationError",
    "DomainProfileStorageError",
    "DomainProfileNotFoundError",
    "DomainProfileAlreadyExistsError",
]
