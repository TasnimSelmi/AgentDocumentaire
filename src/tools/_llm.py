"""
Utilitaires LLM communs aux outils documentaires.

Ce module ne fait aucun retrieval. Il fournit uniquement :
- l'appel au LLM déjà injecté dans le contexte agentique ;
- la construction d'un contexte documentaire compact ;
- le parsing défensif des réponses JSON ;
- la sélection des sources effectivement citées par un outil.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.tools.base import (
    ContexteOutil,
    SourceOutil,
    nettoyer_reflexion,
)


LIMITE_CONTEXTE_OUTIL = 16_000
LIMITE_EXTRAIT_SOURCE = 4_000


def texte_message(reponse: Any) -> str:
    """Extrait proprement le texte d'une réponse LangChain."""
    contenu = getattr(reponse, "content", reponse)

    if isinstance(contenu, str):
        return nettoyer_reflexion(contenu)

    if isinstance(contenu, list):
        morceaux: list[str] = []

        for element in contenu:
            if isinstance(element, str):
                morceaux.append(element)
            elif isinstance(element, dict):
                texte = element.get("text")
                if texte:
                    morceaux.append(str(texte))

        return nettoyer_reflexion("\n".join(morceaux))

    return nettoyer_reflexion(str(contenu))


def invoquer_llm(
    contexte: ContexteOutil | None,
    *,
    systeme: str,
    utilisateur: str,
) -> str:
    """
    Appelle le LLM attaché à la requête agentique.

    Les outils ne construisent jamais leur propre client Ollama/OpenAI :
    le même LLM est injecté par la couche agent.
    """
    if contexte is None or contexte.llm is None:
        raise RuntimeError(
            "Aucun LLM n'est disponible dans le contexte de l'outil."
        )

    messages = [
        SystemMessage(content=systeme),
        HumanMessage(content=utilisateur),
    ]

    reponse = contexte.llm.invoke(messages)
    return texte_message(reponse)


def rendre_sources(
    sources: list[SourceOutil],
    *,
    limite: int = LIMITE_CONTEXTE_OUTIL,
) -> tuple[str, dict[str, SourceOutil]]:
    """
    Transforme les passages conservés en contexte numéroté.

    Les identifiants P1, P2... sont locaux à l'appel de l'outil.
    """
    morceaux: list[str] = []
    correspondance: dict[str, SourceOutil] = {}
    total = 0

    for index, source in enumerate(sources, start=1):
        identifiant = f"P{index}"

        extrait = " ".join(source.extrait.split())
        if len(extrait) > LIMITE_EXTRAIT_SOURCE:
            extrait = extrait[: LIMITE_EXTRAIT_SOURCE - 3] + "..."

        bloc = (
            f"[{identifiant}]\n"
            f"Document : {source.localisation}\n"
            f"Extrait : {extrait}"
        )

        if total + len(bloc) > limite:
            break

        morceaux.append(bloc)
        correspondance[identifiant] = source
        total += len(bloc)

    return "\n\n".join(morceaux), correspondance


def extraire_json_objet(texte: str) -> dict[str, Any]:
    """
    Extrait un objet JSON depuis une réponse LLM.

    Accepte notamment les réponses entourées de ```json ... ```.
    """
    texte = nettoyer_reflexion(texte).strip()

    texte = re.sub(
        r"^```(?:json)?\s*",
        "",
        texte,
        flags=re.IGNORECASE,
    )
    texte = re.sub(r"\s*```$", "", texte)

    try:
        valeur = json.loads(texte)
    except json.JSONDecodeError:
        debut = texte.find("{")
        fin = texte.rfind("}")

        if debut == -1 or fin == -1 or fin <= debut:
            raise ValueError(
                "Le LLM n'a pas retourné d'objet JSON exploitable."
            )

        valeur = json.loads(texte[debut : fin + 1])

    if not isinstance(valeur, dict):
        raise ValueError(
            "La réponse structurée du LLM doit être un objet JSON."
        )

    return valeur


def sources_depuis_ids(
    ids: Any,
    correspondance: dict[str, SourceOutil],
) -> list[SourceOutil]:
    """Résout les identifiants P1/P2 retournés par le LLM."""
    if not isinstance(ids, list):
        return []

    resultat: list[SourceOutil] = []
    vues: set[str] = set()

    for identifiant in ids:
        cle = str(identifiant).strip().upper()
        source = correspondance.get(cle)

        if source is None or cle in vues:
            continue

        resultat.append(source)
        vues.add(cle)

    return resultat