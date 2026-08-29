"""
Routing des branches COMPARE / SYNTHESIZE dans le graphe (P1.5).

Vérifie `noeud_detecter_intention` + `router_intention` : le signal
multi-document déterministe (P1.4) fait basculer SEARCH/SUMMARIZE vers
COMPARE/SYNTHESIZE, sans jamais toucher les autres intentions ni les
anti-faux-positifs. Puis exécution de bout en bout des nouveaux nœuds
(corpus fictif, LLM scripté — aucun Ollama/Qdrant).
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from src.agent import multidoc_pipeline, nodes
from src.agent.graph_state import EtatGraphe
from src.agent.session import construire_session
from tests.agent._multidoc_fakes import LLMScripte, cabler_corpus, passage


class _LLMRoutage:
    """Répond SEARCH aux désambiguïsateurs ; sinon délègue au LLM scripté
    multi-doc (MAP/REDUCE)."""

    def invoke(self, messages):
        systeme = messages[0].content
        if "deux intentions possibles" in systeme:
            return AIMessage(content='{"intention": "SEARCH"}')
        return LLMScripte().invoke(messages)


def _session(query: str):
    return construire_session(
        query,
        llm=_LLMRoutage(),
        charger_profil_domaine=False,
    )


def _intention(query: str) -> str:
    etat = EtatGraphe(session=_session(query))
    maj = nodes.noeud_detecter_intention(etat)
    etat2 = EtatGraphe(session=maj["session"], intention=maj["intention"])
    return nodes.router_intention(etat2)


# --------------------------------------------------------------------------
# Routing — bascule COMPARE / SYNTHESIZE
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "Compare rapport_alpha.pdf et rapport_beta.pdf.",
        "Quelles différences entre ces deux documents ?",
        "Compare the conclusions of report_a.pdf and report_b.pdf.",
        "En quoi contrat_2025.pdf diffère-t-il de contrat_2024.pdf ?",
        "Montre les écarts entre rapport_alpha.pdf et rapport_beta.pdf.",
        "Which of report_a.pdf and report_b.pdf sets stricter penalties?",
    ],
)
def test_rt052_057_routent_vers_compare(query: str) -> None:
    assert _intention(query) == "compare"


@pytest.mark.parametrize(
    "query",
    [
        "Fais une synthèse de rapport_alpha.pdf et rapport_beta.pdf.",
        "Combine the key findings from these two reports.",
        "Consolide les recommandations de rapport_alpha.pdf, rapport_beta.pdf et rapport_gamma.pdf.",
        "Produce a single unified summary of report_a.pdf and report_b.pdf.",
    ],
)
def test_rt058_061_routent_vers_synthesize(query: str) -> None:
    assert _intention(query) == "synthesize"


# --------------------------------------------------------------------------
# Routing — anti-faux-positifs (inchangés)
# --------------------------------------------------------------------------


def test_rt017_mono_document_reste_search() -> None:
    assert _intention("Compare les deux méthodes décrites dans rapport_alpha.pdf.") == "rechercher"


def test_rt018_mono_document_reste_search() -> None:
    assert (
        _intention(
            "Quels sont les points communs entre les deux approches présentées "
            "dans ce document ?"
        )
        == "rechercher"
    )


def test_rt023_multidoc_sans_operation_reste_search() -> None:
    assert (
        _intention("Dans ces documents, quelle est la date limite de dépôt des dossiers ?")
        == "rechercher"
    )


def test_rt030_synthetiser_un_doc_reste_summarize() -> None:
    assert _intention("Peux-tu synthétiser rapport_alpha.pdf ?") == "summarize"


@pytest.mark.parametrize(
    "query,attendu",
    [
        ("Résume le document rapport_alpha.pdf.", "summarize"),
        ("Classe ce document.", "classify"),
        ("Extrais le nom du client et la date de contrat_2025.pdf.", "extract"),
        ("Quel est le montant total indiqué dans facture_2025.pdf ?", "rechercher"),
    ],
)
def test_routing_existant_inchange(query: str, attendu: str) -> None:
    assert _intention(query) == attendu


# --------------------------------------------------------------------------
# Exécution de bout en bout des nouveaux nœuds
# --------------------------------------------------------------------------


def _cabler_2docs(monkeypatch) -> None:
    cabler_corpus(
        monkeypatch,
        multidoc_pipeline,
        fiches={"rapport_alpha.pdf": "A", "rapport_beta.pdf": "B"},
        passages_par_doc={
            "A": [passage("A", 1, "Le montant est de 100.", page=1)],
            "B": [passage("B", 1, "Le montant est de 250.", page=1)],
        },
    )


def test_noeud_compare_produit_un_resultatoutil(monkeypatch) -> None:
    _cabler_2docs(monkeypatch)
    session = _session("Compare rapport_alpha.pdf et rapport_beta.pdf.")
    maj = nodes.noeud_detecter_intention(EtatGraphe(session=session))
    etat = EtatGraphe(session=maj["session"], intention=maj["intention"], multidoc_signal=maj["multidoc_signal"])

    sortie = nodes.noeud_compare(etat)
    resultat = sortie["reponse"]
    assert resultat.outil == "compare"
    assert resultat.succes
    assert set(sortie["documents_resolus"]) == {"rapport_alpha.pdf", "rapport_beta.pdf"}
    assert {s.doc_id for s in resultat.sources} <= {"A", "B"}
    assert sortie["resultat_compare"] is not None


def test_noeud_synthesize_produit_un_resultatoutil(monkeypatch) -> None:
    _cabler_2docs(monkeypatch)
    session = _session("Fais une synthèse de rapport_alpha.pdf et rapport_beta.pdf.")
    maj = nodes.noeud_detecter_intention(EtatGraphe(session=session))
    etat = EtatGraphe(session=maj["session"], intention=maj["intention"], multidoc_signal=maj["multidoc_signal"])

    sortie = nodes.noeud_synthesize(etat)
    resultat = sortie["reponse"]
    assert resultat.outil == "synthesize"
    assert resultat.succes
    assert sortie["resultat_synthesize"] is not None


def test_noeud_compare_abstention_si_reference_non_nommee(monkeypatch) -> None:
    _cabler_2docs(monkeypatch)
    # « ces deux documents » : multi-doc + compare, mais aucun fichier nommé.
    session = _session("Quelles différences entre ces deux documents ?")
    maj = nodes.noeud_detecter_intention(EtatGraphe(session=session))
    assert maj["intention"] == "compare"  # routing OK
    etat = EtatGraphe(session=maj["session"], intention=maj["intention"], multidoc_signal=maj["multidoc_signal"])

    sortie = nodes.noeud_compare(etat)
    resultat = sortie["reponse"]
    assert not resultat.succes  # abstention déterministe, PAS de repli search
    assert resultat.donnees.get("motif") == "references_insuffisantes"


# --------------------------------------------------------------------------
# §2.7 — précédence multi-document sur les zones grises CLASSIFY / EXTRACT
# --------------------------------------------------------------------------


class _LLMDesambigInterdit:
    """Échoue si un désambiguïsateur de zone grise est appelé ; délègue le
    reste au LLM scripté multi-doc."""

    def __init__(self) -> None:
        self.appels_desambig = 0

    def invoke(self, messages):
        systeme = messages[0].content
        if "Tu distingues deux intentions possibles" in systeme:
            self.appels_desambig += 1
            raise AssertionError(
                "désambiguïsateur de zone grise appelé alors que le signal "
                "multi-document est explicite"
            )
        return LLMScripte().invoke(messages)


@pytest.mark.parametrize(
    "query,attendu",
    [
        # « classification » -> _AMBIGU_CLASSIFY ; 2 fichiers + « Compare »
        (
            "Compare la classification de risque de report_a.pdf et report_b.pdf.",
            "compare",
        ),
        # « catégorie » -> _AMBIGU_CLASSIFY ; 2 fichiers + « Consolide » (synthèse)
        (
            "Consolide la catégorie de report_a.pdf et report_b.pdf.",
            "synthesize",
        ),
        # énumération -> _AMBIGU_SEARCH_EXTRACT ; 2 fichiers + « Compare »
        (
            "Compare le fournisseur, la date et le montant de report_a.pdf et report_b.pdf.",
            "compare",
        ),
    ],
)
def test_precedence_multidoc_sur_zone_grise(query: str, attendu: str) -> None:
    llm = _LLMDesambigInterdit()
    session = construire_session(query, llm=llm, charger_profil_domaine=False)
    maj = nodes.noeud_detecter_intention(EtatGraphe(session=session))

    assert maj["intention"] == attendu
    assert llm.appels_desambig == 0  # aucun appel LLM de zone grise
    # trace cohérente
    trace = session.etat.trace[-1]
    assert trace.donnees["intention"] == attendu
    assert trace.donnees["multidoc"] is True
    assert trace.donnees["operation_hint"] == attendu
    assert trace.donnees["multidoc_explicite"] is True
    assert trace.donnees["desambiguisation_llm"] is False

    etat2 = EtatGraphe(session=maj["session"], intention=maj["intention"])
    assert nodes.router_intention(etat2) == attendu


def test_zone_grise_normale_toujours_desambiguisee_si_pas_multidoc() -> None:
    """Une requête à un seul document + vocabulaire ambigu passe TOUJOURS par
    le désambiguïsateur (la précédence n'entre en jeu qu'avec >= 2 fichiers)."""
    llm = _LLMRoutage()  # répond {"intention": "SEARCH"} au désambiguïsateur
    session = construire_session(
        "Quelle est la classification de risque de report_a.pdf ?",
        llm=llm,
        charger_profil_domaine=False,
    )
    maj = nodes.noeud_detecter_intention(EtatGraphe(session=session))
    trace = session.etat.trace[-1]
    assert trace.donnees["desambiguisation_llm"] is True
    assert trace.donnees["multidoc_explicite"] is False
    assert maj["intention"] == "search"
