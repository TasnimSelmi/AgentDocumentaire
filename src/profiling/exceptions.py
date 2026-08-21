"""
Exceptions du module de profils de domaine.

Une seule racine (`DomainProfileError`) permet à l'appelant d'attraper
l'ensemble des erreurs du module sans connaître le détail des causes.
"""

from __future__ import annotations


class DomainProfileError(Exception):
    """Erreur de base liée aux profils de domaine."""


class DomainProfileGenerationError(DomainProfileError):
    """Le LLM n'a pas produit un profil de domaine valide."""


class DomainProfileStorageError(DomainProfileError):
    """Erreur pendant la lecture ou l'écriture d'un profil."""


class DomainProfileNotFoundError(DomainProfileError):
    """Le profil demandé est introuvable."""


class DomainProfileAlreadyExistsError(DomainProfileError):
    """Un profil portant ce nom existe déjà."""


__all__ = [
    "DomainProfileError",
    "DomainProfileGenerationError",
    "DomainProfileStorageError",
    "DomainProfileNotFoundError",
    "DomainProfileAlreadyExistsError",
]
