"""
Outil de synthèse documentaire.

Il synthétise uniquement les passages déjà présents dans le contexte
agentique. Aucun retrieval n'est effectué ici.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

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


class ArgumentsSummarize(BaseModel):

    instruction: str = Field(
        ...,
        min_length=3,
        description=(
            "Ce que la synthèse doit couvrir ou mettre en évidence."
        ),
    )

    longueur: Literal["courte", "moyenne", "detaillee"] = Field(
        default="moyenne",
        description="Niveau de détail souhaité.",
    )


def _executer_summarize(
    *,
    contexte: ContexteOutil | None = None,
    instruction: str,
    longueur: str = "moyenne",
    **_: Any,
) -> ResultatOutil:

    if contexte is None or not contexte.a_des_sources:
        return ResultatOutil.echec(
            "summarize",
            (
                "Aucun passage documentaire n'est disponible. "
                "Utilise d'abord search."
            ),
        )

    texte_sources, correspondance = rendre_sources(
        contexte.sources
    )

    domaine = contexte.bloc_domaine()

    systeme = f"""
Tu es un outil de synthèse documentaire.

{domaine}

RÈGLES
- Synthétise exclusivement les passages fournis.
- Ne complète jamais avec tes connaissances générales.
- Signale les contradictions importantes.
- Ne présente pas une information absente comme certaine.
- Les identifiants P1, P2... correspondent aux passages.
- Retourne uniquement du JSON valide.

Format :
{{
  "resume": "texte de synthèse",
  "points_cles": ["point 1", "point 2"],
  "source_ids": ["P1", "P2"]
}}
""".strip()

    utilisateur = f"""
OBJECTIF
{instruction}

NIVEAU DE DÉTAIL
{longueur}

PASSAGES
{texte_sources}
""".strip()

    texte = invoquer_llm(
        contexte,
        systeme=systeme,
        utilisateur=utilisateur,
    )

    objet = extraire_json_objet(texte)

    resume = str(objet.get("resume", "") or "").strip()
    points = objet.get("points_cles", [])

    if not isinstance(points, list):
        points = []

    sources = sources_depuis_ids(
        objet.get("source_ids"),
        correspondance,
    )

    avertissements: list[str] = []

    if resume and not sources:
        sources = list(correspondance.values())
        avertissements.append(
            "Les passages utilisés n'ont pas été identifiés précisément."
        )

    if not resume:
        return ResultatOutil.echec(
            "summarize",
            "Le LLM n'a produit aucune synthèse exploitable.",
        )

    return ResultatOutil(
        outil="summarize",
        succes=True,
        message="Synthèse documentaire produite.",
        donnees={
            "resume": resume,
            "points_cles": points,
        },
        sources=sources,
        avertissements=avertissements,
    )


@outil
def definir_summarize() -> DefinitionOutil:
    return DefinitionOutil(
        nom="summarize",
        description=(
            "Synthétise les passages documentaires déjà disponibles. "
            "Utilise cet outil lorsqu'il faut résumer, condenser ou dégager "
            "les points essentiels d'informations déjà retrouvées. "
            "Il ne lance aucune nouvelle recherche."
        ),
        schema_arguments=ArgumentsSummarize,
        fonction=_executer_summarize,
        lecture_seule=True,
        actif=True,
    )