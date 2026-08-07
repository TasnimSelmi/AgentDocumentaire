"""Tests du service de suggestion. Aucun appel réseau n'est effectué."""

from __future__ import annotations

import json

import pytest

from src.profiling.exceptions import DomainProfileGenerationError
from src.profiling.prompts import build_domain_profile_prompt
from src.profiling.service import suggest_domain_profile

PROFIL_JSON = {
    "profile_name": "finance",
    "domain": "Finance et comptabilité",
    "description": "États financiers, audit et normes comptables.",
    "keywords": ["comptabilité", "états financiers", "audit"],
    "output_language": "fr",
}


class FauxLLM:
    """Client LLM factice : renvoie une réponse fixée, mémorise le prompt."""

    def __init__(self, reponse: object):
        self.reponse = reponse
        self.prompts: list[str] = []

    def invoke(self, entree):
        self.prompts.append(entree)
        if isinstance(self.reponse, Exception):
            raise self.reponse
        return self.reponse


class Message:
    """Imite un message LangChain (attribut `content`)."""

    def __init__(self, content):
        self.content = content


def test_reponse_json_valide():
    llm = FauxLLM(json.dumps(PROFIL_JSON, ensure_ascii=False))
    profil = suggest_domain_profile("Finance et comptabilité", llm=llm)

    assert profil.profile_name == "finance"
    assert profil.keywords == ["comptabilité", "états financiers", "audit"]
    assert "Finance et comptabilité" in llm.prompts[0]


def test_reponse_dans_un_bloc_markdown():
    brut = "Voici le profil :\n```json\n" + json.dumps(PROFIL_JSON) + "\n```\nVoilà."
    profil = suggest_domain_profile("Finance", llm=FauxLLM(brut))
    assert profil.profile_name == "finance"


def test_reponse_avec_bloc_de_raisonnement():
    brut = "<think>je réfléchis</think>\n" + json.dumps(PROFIL_JSON)
    assert suggest_domain_profile("Finance", llm=FauxLLM(brut)).profile_name == "finance"


def test_reponse_message_langchain():
    llm = FauxLLM(Message(json.dumps(PROFIL_JSON)))
    assert suggest_domain_profile("Finance", llm=llm).profile_name == "finance"


def test_json_invalide():
    with pytest.raises(DomainProfileGenerationError):
        suggest_domain_profile("Finance", llm=FauxLLM("désolé, je ne peux pas."))


def test_reponse_vide():
    with pytest.raises(DomainProfileGenerationError):
        suggest_domain_profile("Finance", llm=FauxLLM("   "))


def test_champ_obligatoire_absent():
    incomplet = {k: v for k, v in PROFIL_JSON.items() if k != "description"}
    with pytest.raises(DomainProfileGenerationError, match="non conforme"):
        suggest_domain_profile("Finance", llm=FauxLLM(json.dumps(incomplet)))


def test_mauvais_type_pour_keywords():
    invalide = dict(PROFIL_JSON, keywords="comptabilité, audit")
    with pytest.raises(DomainProfileGenerationError):
        suggest_domain_profile("Finance", llm=FauxLLM(json.dumps(invalide)))


def test_json_non_objet():
    with pytest.raises(DomainProfileGenerationError):
        suggest_domain_profile("Finance", llm=FauxLLM("[1, 2, 3]"))


def test_domaine_vide():
    with pytest.raises(ValueError):
        suggest_domain_profile("   ", llm=FauxLLM(json.dumps(PROFIL_JSON)))


def test_exception_du_fournisseur():
    llm = FauxLLM(RuntimeError("ollama injoignable"))
    with pytest.raises(DomainProfileGenerationError, match="Échec de l'appel au LLM"):
        suggest_domain_profile("Finance", llm=llm)


def test_langue_demandee_reprise_si_absente():
    sans_langue = {k: v for k, v in PROFIL_JSON.items() if k != "output_language"}
    profil = suggest_domain_profile(
        "Finance", output_language="en", llm=FauxLLM(json.dumps(sans_langue))
    )
    assert profil.output_language == "en"


def test_prompt_generique():
    prompt = build_domain_profile_prompt("Tourisme et voyages")
    assert "Tourisme et voyages" in prompt
    for interdit in ("finance", "comptabilité", "IFRS", "bilan"):
        assert interdit.lower() not in prompt.lower()


def test_prompt_domaine_vide():
    with pytest.raises(ValueError):
        build_domain_profile_prompt("  ")
