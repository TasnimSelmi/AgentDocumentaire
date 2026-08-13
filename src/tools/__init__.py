"""
Registre des outils documentaires disponibles pour l'agent.
"""

from __future__ import annotations

from src.tools.base import (
    ContexteOutil,
    DefinitionOutil,
    RegistreOutils,
    ResultatOutil,
    SourceOutil,
    fabriques_enregistrees,
)

# Ces imports déclenchent volontairement les décorateurs @outil.
from src.tools import classify as _classify  # noqa: F401
from src.tools import extract as _extract  # noqa: F401
from src.tools import search as _search  # noqa: F401
from src.tools import summarize as _summarize  # noqa: F401


def construire_registre(
    contexte: ContexteOutil | None = None,
) -> RegistreOutils:
    """
    Reconstruit tous les outils pour le profil actuellement actif.

    C'est important car search et classify dépendent du profil technique.
    """
    registre = RegistreOutils(contexte=contexte)

    for fabrique in fabriques_enregistrees():
        registre.enregistrer(fabrique())

    return registre


__all__ = [
    "ContexteOutil",
    "DefinitionOutil",
    "RegistreOutils",
    "ResultatOutil",
    "SourceOutil",
    "construire_registre",
]