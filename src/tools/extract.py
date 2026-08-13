"""
Outil d'extraction structurée.

``extract`` travaille exclusivement sur les passages déjà retrouvés par
les appels précédents de ``search``.

Il ne lance jamais de retrieval et n'accède jamais directement à Qdrant.
"""

from __future__ import annotations

from typing import Any

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


class ArgumentsExtract(BaseModel):
    """Arguments visibles par l'agent."""

    instruction: str = Field(
        ...,
        min_length=3,
        description=(
            "Information précise à extraire des passages déjà trouvés."
        ),
    )

    champs: list[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description=(
            "Noms des informations à extraire. "
            "Exemple générique : ['date', 'montant', 'organisation']."
        ),
    )


def _executer_extract(
    *,
    contexte: ContexteOutil | None = None,
    instruction: str,
    champs: list[str],
    **_: Any,
) -> ResultatOutil:

    if contexte is None or not contexte.a_des_sources:
        return ResultatOutil.echec(
            "extract",
            (
                "Aucun passage documentaire n'est disponible. "
                "Utilise d'abord l'outil search."
            ),
        )

    champs = [
        str(champ).strip()
        for champ in champs
        if str(champ).strip()
    ]

    if not champs:
        return ResultatOutil.echec(
            "extract",
            "Aucun champ à extraire n'a été fourni.",
        )

    texte_sources, correspondance = rendre_sources(
        contexte.sources
    )

    domaine = contexte.bloc_domaine()

    systeme = f"""
Tu es un outil d'extraction documentaire stricte.

{domaine}

RÈGLES
- Utilise exclusivement les passages fournis.
- N'invente aucune valeur.
- Si une valeur n'est pas présente, indique trouve=false et valeur=null.
- Ne déduis pas une valeur qui n'est pas explicitement justifiée.
- Les identifiants P1, P2... désignent les passages disponibles.
- Retourne uniquement un objet JSON valide, sans Markdown.

Format obligatoire :
{{
  "valeurs": [
    {{
      "champ": "nom_du_champ",
      "trouve": true,
      "valeur": "valeur extraite",
      "source_ids": ["P1"]
    }}
  ]
}}
""".strip()

    utilisateur = f"""
INSTRUCTION
{instruction}

CHAMPS À EXTRAIRE
{", ".join(champs)}

PASSAGES
{texte_sources}
""".strip()

    texte = invoquer_llm(
        contexte,
        systeme=systeme,
        utilisateur=utilisateur,
    )

    objet = extraire_json_objet(texte)

    valeurs = objet.get("valeurs", [])
    if not isinstance(valeurs, list):
        valeurs = []

    ids_utilises: list[str] = []

    for valeur in valeurs:
        if not isinstance(valeur, dict):
            continue

        ids = valeur.get("source_ids", [])
        if isinstance(ids, list):
            ids_utilises.extend(str(i) for i in ids)

    sources = sources_depuis_ids(
        ids_utilises,
        correspondance,
    )

    # Si le LLM a extrait quelque chose mais a oublié les IDs,
    # on conserve les sources disponibles avec un avertissement.
    avertissements: list[str] = []

    trouve = any(
        isinstance(valeur, dict)
        and bool(valeur.get("trouve"))
        for valeur in valeurs
    )

    if trouve and not sources:
        sources = list(correspondance.values())
        avertissements.append(
            "Le LLM n'a pas indiqué précisément les passages utilisés."
        )

    return ResultatOutil(
        outil="extract",
        succes=True,
        message=(
            "Extraction terminée."
            if trouve
            else "Les informations demandées n'ont pas été trouvées "
                 "dans les passages disponibles."
        ),
        donnees={
            "instruction": instruction,
            "valeurs": valeurs,
        },
        sources=sources,
        avertissements=avertissements,
    )


@outil
def definir_extract() -> DefinitionOutil:
    return DefinitionOutil(
        nom="extract",
        description=(
            "Extrait des valeurs ou informations précises depuis les passages "
            "documentaires déjà obtenus. Utilise cet outil après search "
            "lorsqu'une réponse nécessite des données structurées ou des "
            "valeurs exactes. Cet outil ne recherche pas de nouveaux passages."
        ),
        schema_arguments=ArgumentsExtract,
        fonction=_executer_extract,
        lecture_seule=True,
        actif=True,
    )