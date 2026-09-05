"""
Contrat de sortie unique `AgentResponse` (P1.3).

`normaliser_reponse_agent` est 100 % déterministe : ces tests construisent
directement les structures internes réelles (`ReponseRAG`, `ResultatOutil`)
et vérifient la transposition. Deux tests de bout en bout (`executer_agent`)
utilisent le corpus/LLM fictif — aucun Ollama, aucun Qdrant.
"""

from __future__ import annotations

import json

import pytest

from src.agent import graph as graph_mod
from src.agent.graph import executer_agent
from src.agent.response import (
    CODE_REFUS_SEARCH,
    STATUT_ERREUR,
    STATUT_REFUS,
    STATUT_SUCCES,
    AgentResponse,
    normaliser_reponse_agent,
)
from src.rag.generation import ReponseRAG, SourceCitee
from src.tools.base import ResultatOutil, SourceOutil
from tests.agent._multidoc_fakes import LLMScripte, cabler_corpus, passage


def _source_citee(cid: str, doc: str, page: int) -> SourceCitee:
    return SourceCitee(
        citation=cid, source=doc, nom_fichier=doc, page=page,
        categorie="", score=0.0, extrait=f"extrait {cid}", document=doc,
    )


def _source_outil(doc: str, page: int) -> SourceOutil:
    return SourceOutil(
        doc_id=doc.split(".")[0], source=doc, nom_fichier=doc, page=page,
        categorie="", score=0.0, extrait=f"extrait {doc}",
    )


def _serialisable(ar: AgentResponse) -> dict:
    d = ar.vers_dict()
    recharge = json.loads(json.dumps(d, ensure_ascii=False))  # round-trip
    assert recharge == d
    return d


# =========================================================================
# A — SEARCH succès
# =========================================================================


def test_search_succes() -> None:
    rr = ReponseRAG(
        question="Q ?",
        reponse="La bataille a eu lieu le 20 septembre 1792. [S1]",
        profil="generic",
        contexte_suffisant=True,
        citations_valides=True,
        citations_reparees=False,
        profil_domaine="histoire",
        sources=[_source_citee("S1", "doc_219.txt", 1)],
        avertissements=[],
        duree_secondes=12.3,
    )
    ar = normaliser_reponse_agent(
        rr, capability="search", preuves_pertinentes=True, preuves_suffisantes=True
    )
    assert ar.status == STATUT_SUCCES
    assert ar.capability == "search"
    assert ar.answer == "La bataille a eu lieu le 20 septembre 1792. [S1]"
    assert len(ar.sources) == 1
    assert ar.sources[0]["document"] == "doc_219.txt"
    assert ar.sources[0]["page"] == 1
    assert ar.citations == ["S1"]
    assert ar.error is None
    assert ar.metadata["profil"] == "generic"
    assert ar.metadata["nombre_sources"] == 1
    _serialisable(ar)


def test_search_refus_evidence_insuffisante() -> None:
    rr = ReponseRAG(
        question="Q ?",
        reponse="Aucun passage suffisamment pertinent n'a été retrouvé.",
        profil="generic",
        contexte_suffisant=False,
        citations_valides=True,
        citations_reparees=False,
        sources=[],
        avertissements=["score maximal 0.11 < seuil 0.15"],
    )
    ar = normaliser_reponse_agent(
        rr, capability="search", preuves_pertinentes=False, preuves_suffisantes=None
    )
    assert ar.status == STATUT_REFUS
    assert ar.capability == "search"
    assert ar.error == {
        "code": CODE_REFUS_SEARCH,
        "message": "Aucun passage suffisamment pertinent n'a été retrouvé.",
    }
    assert ar.warnings == ["score maximal 0.11 < seuil 0.15"]
    _serialisable(ar)


# =========================================================================
# B — SUMMARIZE succès
# =========================================================================


def test_summarize_succes() -> None:
    ro = ResultatOutil(
        outil="summarize",
        succes=True,
        message="Résumé produit à partir du document complet (8 passage(s), 2 lot(s)).",
        donnees={
            "resume": "Le rapport traite de X et Y. [S1][S3]",
            "objectif": "Q ?",
            "format": "court",
            "nombre_sources_disponibles": 8,
            "nombre_lots": 2,
            "citations_valides": ["S1", "S3"],
        },
        sources=[_source_outil("doc.pdf", 2), _source_outil("doc.pdf", 5)],
        avertissements=[],
        duree_secondes=30.0,
    )
    ar = normaliser_reponse_agent(ro, capability="summarize")
    assert ar.status == STATUT_SUCCES
    assert ar.capability == "summarize"
    assert ar.answer == "Le rapport traite de X et Y. [S1][S3]"
    assert "resume" not in ar.data  # gros texte retiré de data (dans answer)
    assert ar.data["nombre_lots"] == 2
    assert ar.citations == ["S1", "S3"]
    assert len(ar.sources) == 2
    _serialisable(ar)


# =========================================================================
# C — CLASSIFY succès
# =========================================================================


def test_classify_succes() -> None:
    ro = ResultatOutil(
        outil="classify",
        succes=True,
        message="Document classé : contrat.",
        donnees={
            "document": "doc.pdf",
            "categorie": "contrat",
            "confiance": 0.92,
            "justification": "clauses de résiliation...",
            "categories_autorisees": ["contrat", "rapport", "facture"],
            "citations": ["S1", "S4"],
        },
        sources=[_source_outil("doc.pdf", 1)],
    )
    ar = normaliser_reponse_agent(ro, capability="classify")
    assert ar.status == STATUT_SUCCES
    assert ar.capability == "classify"
    assert ar.data["categorie"] == "contrat"
    assert ar.data["confiance"] == 0.92
    assert ar.data["categories_autorisees"] == ["contrat", "rapport", "facture"]
    assert ar.citations == ["S1", "S4"]
    assert ar.answer == "Document classé : contrat."
    _serialisable(ar)


# =========================================================================
# D — EXTRACT succès (aucune perte de la structure d'extractions)
# =========================================================================


def test_extract_succes_data_intacte() -> None:
    extractions = {
        "date_signature": {"trouve": True, "valeur": "2025-01-10", "citations": ["S2"]},
        "montant_total": {"trouve": True, "valeur": "12 000 €", "citations": ["S5"]},
        "penalite": {"trouve": False, "valeur": None, "citations": []},
    }
    ro = ResultatOutil(
        outil="extract",
        succes=True,
        message="2 champ(s) trouvé(s) sur 3.",
        donnees={
            "document": "doc.pdf",
            "extractions": extractions,
            "champs_demandes": ["date_signature", "montant_total", "penalite"],
            "nombre_demandes": 3,
            "nombre_trouves": 2,
            "nombre_lots": 1,
        },
        sources=[_source_outil("doc.pdf", 2), _source_outil("doc.pdf", 5)],
    )
    ar = normaliser_reponse_agent(ro, capability="extract")
    assert ar.status == STATUT_SUCCES
    assert ar.capability == "extract"
    assert ar.data["extractions"] == extractions  # aucune perte
    assert ar.data["nombre_trouves"] == 2
    assert ar.answer == "2 champ(s) trouvé(s) sur 3."
    _serialisable(ar)


# =========================================================================
# E / F — COMPARE et SYNTHESIZE succès (structure + citations multi-doc)
# =========================================================================


def test_compare_succes_structure_et_citations() -> None:
    comparaison = {
        "question": "Compare A et B",
        "documents": ["a.pdf", "b.pdf"],
        "points_communs": ["Même sujet. [D1S1][D2S1]"],
        "differences": ["A dit 3 % [D1S1] ; B dit 5 % [D2S1]"],
        "positions_par_document": {},
        "contradictions": ["Divergence de taux : [D1S1] contre [D2S1]"],
        "conclusion": "Les deux documents divergent sur le taux. [D1S1][D2S1]",
        "documents_sans_evidence": [],
        "documents_en_echec": [],
    }
    ro = ResultatOutil(
        outil="compare",
        succes=True,
        message="Comparaison de 2 documents (2 avec des éléments pertinents).",
        donnees={
            "comparaison": comparaison,
            "par_document": {
                "a.pdf": {"citations": ["D1S1"], "sans_evidence": False, "echec": None,
                          "nombre_lots": 1, "lots_en_echec": 0},
                "b.pdf": {"citations": ["D2S1"], "sans_evidence": False, "echec": None,
                          "nombre_lots": 1, "lots_en_echec": 0},
            },
            "citations_utilisees": ["D1S1", "D2S1"],
        },
        sources=[_source_outil("a.pdf", 2), _source_outil("b.pdf", 4)],
        avertissements=[],
    )
    ar = normaliser_reponse_agent(ro, capability="compare", documents_resolus=["a.pdf", "b.pdf"])
    assert ar.status == STATUT_SUCCES
    assert ar.capability == "compare"
    assert ar.data["comparaison"] == comparaison  # structure intacte
    assert ar.data["par_document"]["a.pdf"]["nombre_lots"] == 1
    assert ar.citations == ["D1S1", "D2S1"]
    assert ar.answer == "Les deux documents divergent sur le taux. [D1S1][D2S1]"
    assert {s["document"] for s in ar.sources} == {"a.pdf", "b.pdf"}
    assert ar.metadata["documents_resolus"] == ["a.pdf", "b.pdf"]
    _serialisable(ar)


def test_synthesize_succes_structure_et_divergences() -> None:
    synthese = {
        "question": "Synthèse A et B",
        "documents": ["a.pdf", "b.pdf"],
        "themes_communs": ["Thème partagé. [D1S1][D2S1]"],
        "elements_complementaires": ["Apport propre à B. [D2S2]"],
        "divergences": ["D1 affirme X [D1S1] alors que D2 affirme Y [D2S1]"],
        "synthese_transversale": "Synthèse articulée. [D1S1][D2S1]",
        "documents_sans_evidence": [],
        "documents_en_echec": [],
    }
    ro = ResultatOutil(
        outil="synthesize",
        succes=True,
        message="Synthèse transversale de 2 documents (2 avec des éléments pertinents).",
        donnees={
            "synthese": synthese,
            "par_document": {"a.pdf": {}, "b.pdf": {}},
            "citations_utilisees": ["D1S1", "D2S1", "D2S2"],
        },
        sources=[_source_outil("a.pdf", 1), _source_outil("b.pdf", 1)],
    )
    ar = normaliser_reponse_agent(ro, capability="synthesize", documents_resolus=["a.pdf", "b.pdf"])
    assert ar.status == STATUT_SUCCES
    assert ar.capability == "synthesize"
    assert ar.data["synthese"]["divergences"] == synthese["divergences"]
    assert ar.answer == "Synthèse articulée. [D1S1][D2S1]"
    assert ar.citations == ["D1S1", "D2S1", "D2S2"]
    _serialisable(ar)


# =========================================================================
# G — Refus fonctionnels : jamais d'exception, motif conservé
# =========================================================================


@pytest.mark.parametrize(
    "outil,motif,message",
    [
        ("compare", "budget_reduce_depasse", "Les analyses par document dépassent... (X > Y)."),
        ("compare", "document_introuvable", "Document(s) introuvable(s) : rapport_x.pdf."),
        ("synthesize", "references_insuffisantes", "Au moins 2 documents nommés."),
        ("extract", None, "Aucun champ d'extraction valide n'a été fourni."),
    ],
)
def test_refus_fonctionnel(outil: str, motif: str | None, message: str) -> None:
    donnees = {"motif": motif} if motif else {}
    ro = ResultatOutil(outil=outil, succes=False, message=message, donnees=donnees)
    ar = normaliser_reponse_agent(ro, capability=outil)
    assert ar.status == STATUT_REFUS
    assert ar.capability == outil
    assert ar.error == {"code": motif, "message": message}
    assert ar.answer == message
    _serialisable(ar)


# =========================================================================
# H — Warning conservé (ne devient pas une erreur)
# =========================================================================


def test_warning_conserve_sur_un_succes() -> None:
    ro = ResultatOutil(
        outil="compare",
        succes=True,
        message="Comparaison de 3 documents (2 avec des éléments pertinents).",
        donnees={
            "comparaison": {"points_communs": ["x [D1S1]"], "differences": [],
                            "positions_par_document": {}, "contradictions": [],
                            "conclusion": None, "documents": ["a", "b", "c"],
                            "question": "q", "documents_sans_evidence": ["c.pdf"],
                            "documents_en_echec": []},
            "par_document": {}, "citations_utilisees": ["D1S1"],
        },
        sources=[_source_outil("a.pdf", 1)],
        avertissements=["Aucun élément pertinent trouvé dans : c.pdf."],
    )
    ar = normaliser_reponse_agent(ro, capability="compare")
    assert ar.status == STATUT_SUCCES
    assert ar.warnings == ["Aucun élément pertinent trouvé dans : c.pdf."]
    assert ar.error is None
    _serialisable(ar)


# =========================================================================
# I / J — Sérialisation & aucune perte de provenance
# =========================================================================


def test_serialisation_tous_types_natifs() -> None:
    rr = ReponseRAG(
        question="Q", reponse="R [S1]", profil="p", contexte_suffisant=True,
        citations_valides=True, citations_reparees=True,
        sources=[_source_citee("S1", "d.pdf", 3)],
        avertissements=["w"], citations_hors_perimetre=["S9"],
    )
    d = normaliser_reponse_agent(
        rr, capability="search", preuves_pertinentes=True, preuves_suffisantes=True
    ).vers_dict()
    texte = json.dumps(d, ensure_ascii=False)
    assert json.loads(texte) == d
    # aucun objet exotique : uniquement dict/list/str/int/float/bool/None
    def _natif(v):
        if isinstance(v, dict):
            return all(_natif(x) for x in v.values())
        if isinstance(v, list):
            return all(_natif(x) for x in v)
        return v is None or isinstance(v, (str, int, float, bool))
    assert _natif(d)


def test_aucune_perte_de_provenance() -> None:
    sources_internes = [_source_citee("S1", "a.pdf", 2), _source_citee("S2", "b.pdf", 7)]
    rr = ReponseRAG(
        question="Q", reponse="R [S1][S2]", profil="p", contexte_suffisant=True,
        citations_valides=True, citations_reparees=False, sources=sources_internes,
    )
    ar = normaliser_reponse_agent(
        rr, capability="search", preuves_pertinentes=True, preuves_suffisantes=True
    )
    assert len(ar.sources) == len(sources_internes)
    for interne, externe in zip(sources_internes, ar.sources):
        assert externe["citation"] == interne.citation
        assert externe["document"] == interne.document
        assert externe["page"] == interne.page
        assert externe["extrait"] == interne.extrait


# =========================================================================
# Type inattendu -> erreur technique
# =========================================================================


def test_resultat_interne_none_ou_inconnu_est_une_erreur() -> None:
    for mauvais in (None, 42, "texte", {"foo": "bar"}):
        ar = normaliser_reponse_agent(mauvais, capability="search")
        assert ar.status == STATUT_ERREUR
        assert ar.error["code"] == "resultat_interne_inattendu"


# =========================================================================
# Bout en bout : executer_agent (routage -> nœud -> normalisation)
# =========================================================================


def test_executer_agent_compare_bout_en_bout(monkeypatch) -> None:
    from src.agent import multidoc_pipeline
    from langchain_core.messages import AIMessage

    cabler_corpus(
        monkeypatch,
        multidoc_pipeline,
        fiches={"rapport_alpha.pdf": "A", "rapport_beta.pdf": "B"},
        passages_par_doc={
            "A": [passage("A", 1, "Le taux est de 3 %.", page=1)],
            "B": [passage("B", 1, "Le taux est de 5 %.", page=1)],
        },
    )

    class _LLM:
        def invoke(self, messages, think: bool | None = None):
            if "deux intentions possibles" in messages[0].content:
                return AIMessage(content='{"intention": "SEARCH"}')
            return LLMScripte().invoke(messages, think=think)

    ar = executer_agent(
        "Compare rapport_alpha.pdf et rapport_beta.pdf.",
        llm=_LLM(),
        charger_profil_domaine=False,
    )
    assert isinstance(ar, AgentResponse)
    assert ar.status == STATUT_SUCCES
    assert ar.capability == "compare"
    assert "comparaison" in ar.data
    assert {s["document"] for s in ar.sources} <= {"rapport_alpha.pdf", "rapport_beta.pdf", "A.pdf", "B.pdf"}
    assert ar.metadata["documents_resolus"]
    _serialisable(ar)


def test_executer_agent_exception_technique_donne_status_error(monkeypatch) -> None:
    def _explose(*_a, **_k):
        raise RuntimeError("panne interne du graphe")

    monkeypatch.setattr(graph_mod._GRAPHE, "invoke", _explose)
    ar = executer_agent("Compare a.pdf et b.pdf.", charger_profil_domaine=False)
    assert ar.status == STATUT_ERREUR
    assert ar.capability == ""
    assert ar.error == {"code": "RuntimeError", "message": "panne interne du graphe"}
    _serialisable(ar)
