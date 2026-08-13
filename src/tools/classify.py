"""
Outil de classification documentaire.

Les catégories autorisées proviennent du profil technique actif.
L'outil travaille sur les passages déjà disponibles et ne relance
aucune recherche.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.config import get_profil
from src.tools._llm import (
    extraire_json_objet,
    invoquer_llm,
    rendre_sources,
    sources_depuis_ids,
)
from src.tools.base import (
    ContexteOutil,
    DefinitionOutil,
    ResultatOutil,
    outil,
)


class ArgumentsClassify(BaseModel):

    instruction: str = Field(
        default="Classifier le contenu documentaire disponible.",
        description=(
            "Précision éventuelle sur l'objet à classifier."
        ),
    )


def _executer_classify(
    *,
    contexte: ContexteOutil | None = None,
    instruction: str = "Classifier le contenu documentaire disponible.",
    **_: Any,
) -> ResultatOutil:

    if contexte is None or not contexte.a_des_sources:
        return ResultatOutil.echec(
            "classify",
            (
                "Aucun contenu documentaire n'est disponible. "
                "Utilise d'abord search."
            ),
        )

    profil = get_profil()
    categories = profil.classification.noms()

    if not categories:
        return ResultatOutil.echec(
            "classify",
            "Le profil actif ne définit aucune catégorie.",
        )

    texte_sources, correspondance = rendre_sources(
        contexte.sources
    )

    domaine = contexte.bloc_domaine()

    systeme = f"""
Tu es un outil de classification documentaire.

{domaine}

CATÉGORIES AUTORISÉES
{", ".join(categories)}

RÈGLES
- Choisis exclusivement une catégorie de la liste.
- Utilise seulement le contenu documentaire fourni.
- N'invente aucune nouvelle catégorie.
- La confiance doit être comprise entre 0 et 1.
- Retourne uniquement du JSON valide.

Format :
{{
  "categorie": "categorie exacte",
  "confiance": 0.0,
  "raison": "justification courte",
  "source_ids": ["P1"]
}}
""".strip()

    utilisateur = f"""
OBJECTIF
{instruction}

CONTENU
{texte_sources}
""".strip()

    texte = invoquer_llm(
        contexte,
        systeme=systeme,
        utilisateur=utilisateur,
    )

    objet = extraire_json_objet(texte)

    categorie = str(
        objet.get("categorie", "")
    ).strip()

    if categorie not in categories:
        return ResultatOutil.echec(
            "classify",
            (
                f"Catégorie retournée invalide : {categorie!r}. "
                f"Catégories autorisées : {', '.join(categories)}."
            ),
        )

    try:
        confiance = float(objet.get("confiance", 0.0))
    except (TypeError, ValueError):
        confiance = 0.0

    confiance = max(0.0, min(1.0, confiance))

    sources = sources_depuis_ids(
        objet.get("source_ids"),
        correspondance,
    )

    if not sources:
        sources = list(correspondance.values())

    return ResultatOutil(
        outil="classify",
        succes=True,
        message=f"Contenu classifié comme « {categorie} ».",
        donnees={
            "categorie": categorie,
            "confiance": confiance,
            "raison": str(
                objet.get("raison", "")
            ).strip(),
        },
        sources=sources,
    )


@outil
def definir_classify() -> DefinitionOutil:

    profil = get_profil()
    categories = profil.classification.noms()

    description = (
        "Classe le contenu documentaire déjà disponible dans l'une des "
        "catégories définies par le profil actif. "
        "Cet outil ne recherche aucun nouveau passage. "
        f"Catégories possibles : {', '.join(categories) or 'aucune'}."
    )

    return DefinitionOutil(
        nom="classify",
        description=description,
        schema_arguments=ArgumentsClassify,
        fonction=_executer_classify,
        lecture_seule=True,
        actif=True,
    )