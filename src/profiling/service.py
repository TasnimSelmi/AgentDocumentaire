"""
Suggestion d'un profil de domaine par le LLM.

Le service ne lit aucun document et ne touche ni à Qdrant, ni aux embeddings,
ni au retrieval. Il transforme une saisie administrateur (« Finance et
comptabilité ») en un objet `DomainProfile` validé.

La suggestion et la persistance restent séparées : cette fonction ne sauvegarde
jamais le profil.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol, runtime_checkable

from pydantic import ValidationError

from src.profiling.exceptions import DomainProfileGenerationError
from src.profiling.models import DomainProfile
from src.profiling.prompts import build_domain_profile_prompt

logger = logging.getLogger(__name__)

# Longueur de réponse brute reprise dans les messages d'erreur.
EXTRAIT_ERREUR = 500

_BLOC_MARKDOWN = re.compile(r"```(?:json|JSON)?\s*(?P<contenu>.*?)```", re.DOTALL)
_BLOC_RAISONNEMENT = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@runtime_checkable
class ClientLLM(Protocol):
    """
    Interface minimale attendue d'un client LLM.

    `ChatOllama` (LangChain) la satisfait déjà. Un faux client suffit donc
    pour tester le service sans appel réseau.
    """

    def invoke(self, entree: Any) -> Any:  # pragma: no cover - protocole
        ...


def _client_par_defaut() -> ClientLLM:
    """
    Construit le client LLM du projet.

    L'import est volontairement local : importer `src.profiling` ne doit
    charger ni LangChain ni la configuration LLM, et surtout ne déclencher
    aucun appel réseau.
    """
    from src.llm.factory import construire_llm

    return construire_llm()


def extraire_texte_reponse(reponse: Any) -> str:
    """
    Extrait le texte d'une réponse LLM, qu'elle soit une chaîne, un message
    LangChain ou une liste de blocs de contenu.
    """
    contenu = getattr(reponse, "content", reponse)
    if isinstance(contenu, str):
        return contenu.strip()
    if isinstance(contenu, list):
        morceaux: list[str] = []
        for element in contenu:
            if isinstance(element, str):
                morceaux.append(element)
            elif isinstance(element, dict) and isinstance(element.get("text"), str):
                morceaux.append(element["text"])
        return "\n".join(morceaux).strip()
    return str(contenu).strip()


def extraire_json(texte: str) -> dict[str, Any]:
    """
    Isole l'objet JSON d'une réponse LLM.

    Tolère un bloc Markdown ```json, un bloc de raisonnement <think> et du
    texte autour de l'objet. Ne tolère aucun JSON absent ou invalide.

    Raises:
        DomainProfileGenerationError: si aucun objet JSON exploitable n'est trouvé.
    """
    nettoye = _BLOC_RAISONNEMENT.sub("", texte).strip()

    bloc = _BLOC_MARKDOWN.search(nettoye)
    if bloc:
        nettoye = bloc.group("contenu").strip()

    if not nettoye:
        raise DomainProfileGenerationError("Le LLM a retourné une réponse vide.")

    candidats = [nettoye]
    debut = nettoye.find("{")
    fin = nettoye.rfind("}")
    if debut != -1 and fin > debut:
        candidats.append(nettoye[debut : fin + 1])

    for candidat in candidats:
        try:
            donnees = json.loads(candidat)
        except json.JSONDecodeError:
            continue
        if isinstance(donnees, dict):
            return donnees
        raise DomainProfileGenerationError(
            "Le LLM a retourné un JSON qui n'est pas un objet : "
            f"{type(donnees).__name__}."
        )

    raise DomainProfileGenerationError(
        "Le LLM n'a pas retourné de JSON exploitable. "
        f"Réponse reçue : {nettoye[:EXTRAIT_ERREUR]!r}"
    )


def suggest_domain_profile(
    domain: str,
    output_language: str = "fr",
    *,
    llm: ClientLLM | None = None,
) -> DomainProfile:
    """
    Demande au LLM une proposition de profil pour le domaine indiqué.

    Args:
        domain: domaine métier saisi par l'administrateur.
        output_language: code de langue attendu pour le profil.
        llm: client LLM à utiliser. Par défaut, celui du projet
            (`src.llm.factory.construire_llm`). Permet d'injecter un faux
            client dans les tests.

    Returns:
        Un `DomainProfile` validé. Le profil n'est pas sauvegardé.

    Raises:
        ValueError: si le domaine est vide.
        DomainProfileGenerationError: si le LLM échoue ou produit une
            réponse non conforme au modèle.
    """
    domaine = (domain or "").strip()
    if not domaine:
        raise ValueError("Le domaine est obligatoire.")

    prompt = build_domain_profile_prompt(domaine, output_language)
    client = llm if llm is not None else _client_par_defaut()

    logger.info("Suggestion d'un profil de domaine pour %r.", domaine)

    try:
        reponse = client.invoke(prompt)
    except Exception as exc:  # noqa: BLE001 — normalisation de l'erreur provider
        raise DomainProfileGenerationError(
            f"Échec de l'appel au LLM : {exc}"
        ) from exc

    texte = extraire_texte_reponse(reponse)
    donnees = extraire_json(texte)

    # Le LLM omet parfois la langue : on réutilise celle demandée plutôt que
    # de laisser la valeur par défaut du modèle contredire la demande.
    donnees.setdefault("output_language", output_language)
    # Le domaine saisi par l'administrateur fait foi s'il manque à la réponse.
    donnees.setdefault("domain", domaine)

    try:
        profil = DomainProfile.model_validate(donnees)
    except ValidationError as exc:
        raise DomainProfileGenerationError(
            "Le LLM a produit un profil non conforme au modèle attendu. "
            f"Détails : {exc.errors(include_url=False)}"
        ) from exc

    logger.info("Profil suggéré : %s (%d mots-clés).", profil.profile_name, len(profil.keywords))
    return profil


__all__ = ["suggest_domain_profile", "ClientLLM", "extraire_json", "extraire_texte_reponse"]
