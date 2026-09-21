"""
Tests de `src.llm.factory.construire_llm` — timeout HTTP effectif.

`src/llm/factory.py` n'est listé nulle part dans `docs/DO_NOT_TOUCH.md` :
modifiable sans cycle d'évaluation complet.

Contexte : `ConfigAgent.timeout_secondes` (`config/default.yaml`) existait
déjà mais n'était jamais transmis à Ollama — un appel bloqué ne levait
jamais d'exception, quelle que soit sa valeur. La correction transmet cette
même valeur (source de vérité unique, aucune constante parallèle) via
`ChatOllama(client_kwargs={"timeout": ...})`, qui la propage à
`ollama.Client`/`AsyncClient` puis à `httpx.Client`/`AsyncClient`
(vérifié : `ollama.Client.__init__` fait
`super().__init__(httpx.Client, host, **kwargs)`).

Les deux derniers tests appellent un VRAI serveur Ollama local (aucune
doublure) : ils vérifient le mécanisme réel, pas seulement la présence du
paramètre. Ignorés proprement si Ollama n'est pas joignable dans
l'environnement d'exécution.
"""

from __future__ import annotations

import time

import httpx
import pytest

from src.config import get_config_technique, get_settings
from src.llm.factory import construire_llm


def _ollama_joignable() -> bool:
    settings = get_settings()
    base_url = settings.llm_base_url or "http://localhost:11434"
    try:
        httpx.get(f"{base_url}/api/tags", timeout=2.0)
        return True
    except httpx.HTTPError:
        return False


def test_construire_llm_transmet_le_timeout_configure():
    """`client_kwargs["timeout"]` == `ConfigAgent.timeout_secondes`, la
    configuration réelle du projet — jamais une valeur raccourcie pour le
    test."""
    llm = construire_llm()

    timeout_projet = get_config_technique().agent.timeout_secondes
    assert llm.client_kwargs == {"timeout": timeout_projet}
    assert timeout_projet == 120, (
        "valeur figée de config/default.yaml (agent.timeout_secondes) — si "
        "ce test casse, la configuration du projet a changé, pas ce test"
    )


@pytest.mark.skipif(not _ollama_joignable(), reason="Ollama non joignable dans cet environnement")
def test_timeout_reel_declenche_une_exception_controlable():
    """Mécanisme réel : un timeout délibérément trop court pour un vrai
    appel Ollama doit lever une exception `httpx` contrôlable, pas bloquer
    indéfiniment ni planter silencieusement. N'utilise PAS `construire_llm()`
    (qui applique la vraie valeur projet, 120s) : ce test exerce directement
    le mécanisme `client_kwargs` avec un timeout volontairement minuscule,
    séparé de la valeur de configuration source de vérité."""
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage

    settings = get_settings()
    llm = ChatOllama(
        model=settings.llm_model,
        base_url=settings.llm_base_url or "http://localhost:11434",
        num_predict=64,
        client_kwargs={"timeout": 0.001},
    )

    with pytest.raises(Exception) as exc_info:
        llm.invoke([HumanMessage(content="Réponds juste 'ok'.")])

    # httpx.TimeoutException ou une exception l'enveloppant (ResponseError
    # côté client ollama) — dans les deux cas le nom porte "Timeout".
    assert "timeout" in str(type(exc_info.value)).lower() or "timeout" in str(
        exc_info.value
    ).lower()


@pytest.mark.skipif(not _ollama_joignable(), reason="Ollama non joignable dans cet environnement")
def test_comportement_normal_inchange_avec_le_vrai_timeout():
    """Le comportement normal (appel qui répond bien avant le timeout
    configuré) n'est pas affecté par l'ajout du paramètre."""
    from langchain_core.messages import HumanMessage

    llm = construire_llm()
    debut = time.monotonic()

    reponse = llm.invoke([HumanMessage(content="Réponds uniquement le mot 'ok', rien d'autre.")])

    duree = time.monotonic() - debut
    assert reponse.content is not None
    timeout_projet = get_config_technique().agent.timeout_secondes
    assert duree < timeout_projet, "un appel normal ne doit pas approcher le timeout configuré"
