"""
Fabrique centralisée du modèle de langage local.

Le projet utilise exclusivement Ollama :
- aucune clé API payante ;
- aucune dépendance à OpenAI ou Anthropic ;
- données et inférences conservées dans l'environnement local.
"""

from __future__ import annotations

from langchain_ollama import ChatOllama

from src.config import get_settings


class ErreurConfigurationLLM(RuntimeError):
    """Configuration du fournisseur LLM invalide."""


def construire_llm() -> ChatOllama:
    """
    Construit le client LangChain connecté au serveur Ollama local.

    La configuration est lue depuis le fichier .env.
    """
    settings = get_settings()
    provider = settings.llm_provider.strip().lower()

    if provider != "ollama":
        raise ErreurConfigurationLLM(
            "Ce projet est configuré pour utiliser uniquement Ollama. "
            f"Valeur reçue pour LLM_PROVIDER : {provider!r}. "
            "Utilise LLM_PROVIDER=ollama dans le fichier .env."
        )

    modele = settings.llm_model.strip()

    if not modele:
        raise ErreurConfigurationLLM(
            "LLM_MODEL est vide. Indique le nom d'un modèle Ollama "
            "dans le fichier .env."
        )

    base_url = (
        settings.llm_base_url.strip()
        if settings.llm_base_url
        else "http://localhost:11434"
    )

    return ChatOllama(
        model=modele,
        base_url=base_url,
        temperature=settings.llm_temperature,
        num_predict=settings.llm_max_tokens,
        num_ctx=settings.llm_num_ctx,
    )