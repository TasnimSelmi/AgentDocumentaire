"""
Fonctions LLM communes au RAG et aux outils agentiques.

Ce module ne fait aucun retrieval et ne contient aucune logique agentique.
Il centralise uniquement les opérations génériques liées au modèle :
    - nettoyage des réponses ;
    - invocation LangChain ;
    - contexte du DomainProfile ;
    - extraction défensive de JSON.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.profiling import DomainProfile
from src.tools.base import nettoyer_reflexion


def texte_message(reponse: Any) -> str:
    """
    Extrait le texte d'une réponse LangChain quel que soit le provider.

    Les éventuels blocs <think> sont retirés avant réutilisation.
    """
    contenu = getattr(reponse, "content", reponse)

    if isinstance(contenu, str):
        return nettoyer_reflexion(contenu)

    if isinstance(contenu, list):
        morceaux: list[str] = []

        for element in contenu:
            if isinstance(element, str):
                morceaux.append(element)

            elif (
                isinstance(element, dict)
                and isinstance(element.get("text"), str)
            ):
                morceaux.append(element["text"])

        return nettoyer_reflexion("\n".join(morceaux))

    return nettoyer_reflexion(str(contenu))


def bloc_profil_domaine(
    profil_domaine: DomainProfile | None,
) -> str:
    """
    Construit le contexte métier fourni au LLM.

    Le DomainProfile sert uniquement à comprendre le vocabulaire et le
    domaine. Il n'est jamais considéré comme une source factuelle.
    """
    if profil_domaine is None:
        return ""

    return f"""
{profil_domaine.bloc_contexte_domaine()}

RÈGLES D'UTILISATION DU PROFIL DE DOMAINE
- Utilise ce profil uniquement pour comprendre le domaine et son vocabulaire.
- Ce profil n'est jamais une source factuelle.
- Les informations factuelles doivent provenir des documents fournis.
- En cas de conflit, les documents sont prioritaires.
""".strip()


def invoquer_llm(
    llm: Any,
    *,
    systeme: str,
    utilisateur: str,
) -> str:
    """Exécute un appel LLM simple avec messages système et utilisateur."""
    if llm is None:
        raise RuntimeError("Aucun LLM n'a été fourni.")

    messages = [
        SystemMessage(content=systeme),
        HumanMessage(content=utilisateur),
    ]

    reponse = llm.invoke(messages)

    texte = texte_message(reponse)

    if not texte:
        raise RuntimeError("Le LLM a retourné une réponse vide.")

    return texte


def extraire_json_objet(texte: str) -> dict[str, Any]:
    """
    Extrait un objet JSON d'une réponse LLM.

    Tolère notamment :
        ```json
        {...}
        ```
    """
    texte = nettoyer_reflexion(texte).strip()

    texte = re.sub(
        r"^```(?:json)?\s*",
        "",
        texte,
        flags=re.IGNORECASE,
    )
    texte = re.sub(
        r"\s*```$",
        "",
        texte,
    )

    try:
        valeur = json.loads(texte)

    except json.JSONDecodeError:
        debut = texte.find("{")
        fin = texte.rfind("}")

        if debut == -1 or fin == -1 or fin <= debut:
            raise ValueError(
                "Le LLM n'a pas retourné d'objet JSON exploitable."
            )

        try:
            valeur = json.loads(
                texte[debut : fin + 1]
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Le JSON retourné par le LLM est invalide."
            ) from exc

    if not isinstance(valeur, dict):
        raise ValueError(
            "La réponse structurée du LLM doit être un objet JSON."
        )

    return valeur