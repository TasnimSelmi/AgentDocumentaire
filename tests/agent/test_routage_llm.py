"""
Classifieur d'intention LLM de repli (P1.10) — `src.agent.nodes`.

Aucun Ollama, aucun Qdrant : LLM factice qui compte ses appels. Vérifie que
le repli n'intervient que sur une requête déjà routée SEARCH, que ses
garde-fous déterministes tiennent (promotion limitée, documents désignés
exigés pour COMPARE / SYNTHESIZE) et que tout échec retombe sur SEARCH.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from evaluation import evaluate_routing as er
from src.agent import nodes
from src.agent.graph_state import EtatGraphe
from src.agent.multidoc import PORTEE_INTER, PORTEE_INTRA, signal_operation_imposee
from src.agent.session import construire_session


class _LLMClassifieur:
    """Répond `intention` au classifieur de repli, SEARCH aux
    désambiguïsateurs de zone grise ; compte les appels au classifieur."""

    def __init__(self, intention: str = "SEARCH", *, brut: str | None = None) -> None:
        self.intention = intention
        self.brut = brut
        self.appels_classifieur = 0

    def invoke(self, messages, think: bool | None = None):
        systeme = messages[0].content
        if "Tu identifies l'intention" in systeme:
            self.appels_classifieur += 1
            if self.brut is not None:
                return AIMessage(content=self.brut)
            return AIMessage(content=json.dumps({"intention": self.intention, "portee": None}))
        return AIMessage(content='{"intention": "SEARCH"}')


class _LLMEnPanne:
    def invoke(self, messages, think: bool | None = None):
        raise RuntimeError("LLM indisponible")


def _detecter(query: str, llm) -> dict:
    session = construire_session(query, llm=llm, charger_profil_domaine=False)
    return nodes.noeud_detecter_intention(EtatGraphe(session=session))


# --------------------------------------------------------------------------
# Classifieur brut : repli SEARCH sur tout échec
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "llm",
    [None, _LLMEnPanne(), _LLMClassifieur(brut="pas du JSON"), _LLMClassifieur(brut="{}")],
)
def test_classifieur_repli_search_sur_echec(llm) -> None:
    assert nodes._classifier_intention_llm(llm, "Compare A et B.") == "search"


# --------------------------------------------------------------------------
# Garde-fous déterministes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sortie", ["EXTRACT", "CLASSIFY", "SEARCH", "INCONNU"])
def test_promotion_limitee_a_summarize_compare_synthesize(sortie: str) -> None:
    intention, signal, brut = nodes._repli_classification_llm(
        _LLMClassifieur(sortie), "Quel est le montant total indiqué dans facture_2025.pdf ?"
    )
    assert intention == "search"
    assert signal is None
    assert brut == sortie.lower()


def test_summarize_promu_sans_signal() -> None:
    intention, signal, _ = nodes._repli_classification_llm(
        _LLMClassifieur("SUMMARIZE"), "Ce doc parle de quoi en gros ?"
    )
    assert intention == "summarize"
    assert signal is None


def test_compare_sans_document_designe_reste_search() -> None:
    intention, signal, brut = nodes._repli_classification_llm(
        _LLMClassifieur("COMPARE"), "Quelle est la différence entre le contrat CDD et le CDI ?"
    )
    assert (intention, signal, brut) == ("search", None, "compare")


def test_compare_deux_fichiers_nommes_portee_inter() -> None:
    intention, signal, _ = nodes._repli_classification_llm(
        _LLMClassifieur("COMPARE"), "c quoi la diff entre rapport_alpha.pdf et rapport_beta.pdf"
    )
    assert intention == "compare"
    assert signal.portee == PORTEE_INTER
    assert signal.is_multidoc is True
    assert signal.documents_cibles == ("rapport_alpha.pdf", "rapport_beta.pdf")


def test_synthese_documents_resolus_portee_inter() -> None:
    intention, signal, _ = nodes._repli_classification_llm(
        _LLMClassifieur("SYNTHESIZE"),
        "fusionne les constats des deux baromètres en un seul texte",
        resolveur=lambda _q: ("barometre_2024.pdf", "barometre_2025.pdf"),
    )
    assert intention == "synthesize"
    assert signal.portee == PORTEE_INTER
    assert signal.documents_cibles == ("barometre_2024.pdf", "barometre_2025.pdf")


def test_compare_un_document_resolu_portee_intra() -> None:
    intention, signal, _ = nodes._repli_classification_llm(
        _LLMClassifieur("COMPARE"),
        "Qu'est-ce qui change entre le diagnostic et les recommandations de l'avis ?",
        resolveur=lambda _q: ("avis.pdf",),
    )
    assert intention == "compare"
    assert signal.portee == PORTEE_INTRA
    assert signal.documents_cibles == ("avis.pdf",)


def test_compare_deixis_demonstrative_portee_intra() -> None:
    signal = signal_operation_imposee(
        "Le début et la fin de ce rapport disent-ils la même chose ?", "compare"
    )
    assert signal is not None
    assert signal.portee == PORTEE_INTRA
    assert signal.documents_cibles == ()


def test_signal_operation_imposee_refuse_une_operation_hors_multidoc() -> None:
    assert signal_operation_imposee("Résume rapport_alpha.pdf", "summarize") is None


def test_resolveur_defaillant_ne_casse_pas_le_repli() -> None:
    def _resolveur(_q: str):
        raise RuntimeError("catalogue indisponible")

    intention, signal, _ = nodes._repli_classification_llm(
        _LLMClassifieur("COMPARE"), "compare le rapport X et le rapport Y", resolveur=_resolveur
    )
    assert (intention, signal) == ("search", None)


# --------------------------------------------------------------------------
# Branchement dans `noeud_detecter_intention`
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,attendu",
    [
        ("Résume le document rapport_alpha.pdf.", "summarize"),
        ("Classe ce document.", "classify"),
        ("Compare rapport_alpha.pdf et rapport_beta.pdf.", "compare"),
    ],
)
def test_intention_reconnue_par_les_regles_jamais_soumise_au_llm(query: str, attendu: str) -> None:
    llm = _LLMClassifieur("SYNTHESIZE")
    maj = _detecter(query, llm)
    assert maj["intention"] == attendu
    assert llm.appels_classifieur == 0


def test_requete_search_soumise_une_fois_au_classifieur() -> None:
    llm = _LLMClassifieur("SEARCH")
    maj = _detecter("Combien d'entreprises utilisent le cloud ?", llm)
    assert maj["intention"] == "search"
    assert llm.appels_classifieur == 1
    trace = maj["session"].etat.trace[-1]
    assert trace.donnees["classification_llm"] == "search"


def test_repli_summarize_route_vers_summarize() -> None:
    maj = _detecter("Ce doc parle de quoi en gros ?", _LLMClassifieur("SUMMARIZE"))
    etat = EtatGraphe(session=maj["session"], intention=maj["intention"])
    assert nodes.router_intention(etat) == "summarize"


def test_repli_compare_remplace_le_signal_multidoc(monkeypatch) -> None:
    corpus: list[str] = []

    def _fabrique(corpus_id: str = "default"):
        corpus.append(corpus_id)
        return lambda _q: ("barometre_2024.pdf", "barometre_2025.pdf")

    monkeypatch.setattr(nodes, "resolveur_catalogue", _fabrique)
    maj = _detecter(
        "c quoi la diff entre le barometre 2024 et celui de 2025 ?", _LLMClassifieur("COMPARE")
    )

    assert maj["intention"] == "compare"
    signal = maj["multidoc_signal"]
    assert signal.portee == PORTEE_INTER
    assert signal.documents_cibles == ("barometre_2024.pdf", "barometre_2025.pdf")
    assert corpus and set(corpus) == {"default"}


def test_repli_compare_sans_document_reste_search() -> None:
    maj = _detecter(
        "Quelle est la différence entre le contrat CDD et le CDI ?", _LLMClassifieur("COMPARE")
    )
    assert maj["intention"] == "search"


def test_llm_en_panne_reste_search() -> None:
    maj = _detecter("Combien d'entreprises utilisent le cloud ?", _LLMEnPanne())
    assert maj["intention"] == "search"


# --------------------------------------------------------------------------
# Banc de routage : même repli en production, jamais en déterministe
# --------------------------------------------------------------------------


def test_banc_production_applique_le_repli() -> None:
    routee, _, deferred = er.router_cas(
        "Ce doc parle de quoi en gros ?", mode=er.MODE_PRODUCTION, llm=_LLMClassifieur("SUMMARIZE")
    )
    assert routee == "SUMMARIZE"
    assert deferred == "repli_llm"


def test_banc_deterministe_nappelle_jamais_le_classifieur() -> None:
    routee, _, deferred = er.router_cas("Ce doc parle de quoi en gros ?", mode=er.MODE_DETERMINISTE)
    assert routee == "SEARCH"
    assert deferred is None
