"""
Tests de la capacité COMPARE (P1.5) — `src.tools.compare`.

Aucun Ollama, aucun Qdrant : corpus fictif câblé sur
`src.agent.multidoc_pipeline`, LLM scripté qui ne cite que ce qu'il voit.
Le REDUCE ne reçoit JAMAIS le corpus, seulement les sorties MAP validées.
"""

from __future__ import annotations

import pytest

from src.agent import multidoc_pipeline
from src.tools.compare import ResultatCompare, comparer
from tests.agent._multidoc_fakes import (
    SANS_EVIDENCE,
    LLMExplose,
    LLMScripte,
    cabler_corpus,
    make_llm_map_echoue,
    passage,
)


def _corpus_2(monkeypatch, *, b_sans_evidence: bool = False) -> None:
    cabler_corpus(
        monkeypatch,
        multidoc_pipeline,
        fiches={"rapport_a.pdf": "A", "rapport_b.pdf": "B"},
        passages_par_doc={
            "A": [passage("A", 1, "Le taux retenu est de 3 %.", page=2)],
            "B": [
                passage(
                    "B",
                    1,
                    SANS_EVIDENCE if b_sans_evidence else "Le taux retenu est de 5 %.",
                    page=4,
                )
            ],
        },
    )


def _corpus_3(monkeypatch) -> None:
    cabler_corpus(
        monkeypatch,
        multidoc_pipeline,
        fiches={
            "rapport_a.pdf": "A",
            "rapport_b.pdf": "B",
            "rapport_c.pdf": "C",
        },
        passages_par_doc={
            "A": [passage("A", 1, "Position A.")],
            "B": [passage("B", 1, "Position B.")],
            "C": [passage("C", 1, "Position C.")],
        },
    )


# --------------------------------------------------------------------------
# Cas nominaux
# --------------------------------------------------------------------------


def test_compare_deux_documents_fr(monkeypatch) -> None:
    _corpus_2(monkeypatch)
    llm = LLMScripte()
    r = comparer(
        "Compare le taux entre rapport_a.pdf et rapport_b.pdf.",
        ["rapport_a.pdf", "rapport_b.pdf"],
        llm=llm,
    )
    assert r.succes
    comp = ResultatCompare(**r.donnees["comparaison"])
    assert comp.documents == ["rapport_a.pdf", "rapport_b.pdf"]
    assert comp.points_communs and comp.differences
    assert r.sources  # provenance conservée
    # aucun appel REDUCE ne contient de passage brut du corpus
    systeme_reduce = [s for s, u in llm.appels if "COMPARAISON" in s]
    assert systeme_reduce, "le REDUCE a bien été appelé"
    for _s, u in [(s, u) for s, u in llm.appels if "COMPARAISON" in s]:
        assert "Le taux retenu est de 3 %" not in u  # que les analyses MAP


def test_compare_trois_documents(monkeypatch) -> None:
    _corpus_3(monkeypatch)
    r = comparer(
        "Compare rapport_a.pdf, rapport_b.pdf et rapport_c.pdf.",
        ["rapport_a.pdf", "rapport_b.pdf", "rapport_c.pdf"],
        llm=LLMScripte(),
    )
    assert r.succes
    assert len(r.donnees["comparaison"]["documents"]) == 3


def test_compare_en_anglais(monkeypatch) -> None:
    cabler_corpus(
        monkeypatch,
        multidoc_pipeline,
        fiches={"report_a.pdf": "A", "report_b.pdf": "B"},
        passages_par_doc={
            "A": [passage("A", 1, "Penalty clause: 2%.")],
            "B": [passage("B", 1, "Penalty clause: 10%.")],
        },
    )
    r = comparer(
        "Compare the penalty clauses of report_a.pdf and report_b.pdf.",
        ["report_a.pdf", "report_b.pdf"],
        llm=LLMScripte(),
    )
    assert r.succes


# --------------------------------------------------------------------------
# Provenance / cloisonnement
# --------------------------------------------------------------------------


def test_citations_conservees_et_dans_le_perimetre(monkeypatch) -> None:
    _corpus_2(monkeypatch)
    r = comparer(
        "Compare rapport_a.pdf et rapport_b.pdf.",
        ["rapport_a.pdf", "rapport_b.pdf"],
        llm=LLMScripte(),
    )
    assert r.succes
    doc_ids = {s.doc_id for s in r.sources}
    assert doc_ids <= {"A", "B"}
    assert all(c.startswith(("D1", "D2")) for c in r.donnees["citations_utilisees"])


def test_aucun_passage_d_un_troisieme_document_non_demande(monkeypatch) -> None:
    # Le corpus contient A, B ET C, mais la requête ne vise que A et B.
    _corpus_3(monkeypatch)
    r = comparer(
        "Compare rapport_a.pdf et rapport_b.pdf.",
        ["rapport_a.pdf", "rapport_b.pdf"],
        llm=LLMScripte(),
    )
    assert r.succes
    assert {s.doc_id for s in r.sources} <= {"A", "B"}
    assert "D3" not in " ".join(r.donnees["citations_utilisees"])


def test_citation_hors_scope_du_llm_est_filtree(monkeypatch) -> None:
    _corpus_2(monkeypatch)
    r = comparer(
        "Compare rapport_a.pdf et rapport_b.pdf.",
        ["rapport_a.pdf", "rapport_b.pdf"],
        llm=LLMScripte(citation_hors_scope="D9S9"),
    )
    assert r.succes
    assert "D9S9" not in " ".join(r.donnees["citations_utilisees"])
    assert all("D9S9" not in d for d in r.donnees["comparaison"]["differences"])


def test_contradiction_conservee(monkeypatch) -> None:
    _corpus_2(monkeypatch)
    r = comparer(
        "Compare le taux de rapport_a.pdf et rapport_b.pdf.",
        ["rapport_a.pdf", "rapport_b.pdf"],
        llm=LLMScripte(),
    )
    assert r.succes
    comp = r.donnees["comparaison"]
    assert comp["contradictions"], "la divergence entre documents doit être conservée"


# --------------------------------------------------------------------------
# Abstention / erreurs
# --------------------------------------------------------------------------


def test_document_absent_refus_sur(monkeypatch) -> None:
    _corpus_2(monkeypatch)
    r = comparer(
        "Compare rapport_a.pdf et rapport_absent.pdf.",
        ["rapport_a.pdf", "rapport_absent.pdf"],
        llm=LLMScripte(),
    )
    assert not r.succes
    assert "introuvable" in r.message.lower()
    assert r.donnees.get("motif") == "document_introuvable"


def test_reference_unique_refus(monkeypatch) -> None:
    _corpus_2(monkeypatch)
    r = comparer("Compare rapport_a.pdf.", ["rapport_a.pdf"], llm=LLMScripte())
    assert not r.succes
    assert r.donnees.get("motif") == "references_insuffisantes"


def test_deux_references_meme_document_refus(monkeypatch) -> None:
    _corpus_2(monkeypatch)
    r = comparer(
        "Compare rapport_a.pdf et rapport_a.pdf.",
        ["rapport_a.pdf", "rapport_a.pdf"],
        llm=LLMScripte(),
    )
    assert not r.succes  # dédupliqué -> < 2 références


def test_au_dela_de_la_limite_refus(monkeypatch) -> None:
    cabler_corpus(
        monkeypatch,
        multidoc_pipeline,
        fiches={f"r{i}.pdf": f"D{i}" for i in range(6)},
        passages_par_doc={f"D{i}": [passage(f"D{i}", 1, "x")] for i in range(6)},
    )
    refs = [f"r{i}.pdf" for i in range(5)]
    r = comparer("Compare 5 docs.", refs, llm=LLMScripte())
    assert not r.succes
    assert r.donnees.get("motif") == "au_dela_limite"
    assert "4" in r.message


def test_un_document_sans_evidence_pertinente_partiel(monkeypatch) -> None:
    """Audit long-documents, section D : 1 document exploitable sur 2 ->
    réponse PARTIELLE sourcée, jamais un refus global (ancien comportement)."""
    _corpus_2(monkeypatch, b_sans_evidence=True)
    r = comparer(
        "Compare rapport_a.pdf et rapport_b.pdf.",
        ["rapport_a.pdf", "rapport_b.pdf"],
        llm=LLMScripte(),
    )
    assert r.succes
    assert r.donnees.get("statut") == "partiel"
    assert "rapport_b.pdf" in r.donnees["comparaison"]["documents_sans_evidence"]
    # Preuve réellement disponible (document A) conservée et sourcée.
    assert r.sources
    assert {s.doc_id for s in r.sources} == {"A"}


def test_aucun_document_exploitable_refus_dur(monkeypatch) -> None:
    """Zéro preuve exploitable : le refus dur reste (seul cas restant)."""
    cabler_corpus(
        monkeypatch,
        multidoc_pipeline,
        fiches={"rapport_a.pdf": "A", "rapport_b.pdf": "B"},
        passages_par_doc={
            "A": [passage("A", 1, SANS_EVIDENCE, page=1)],
            "B": [passage("B", 1, SANS_EVIDENCE, page=1)],
        },
    )
    r = comparer(
        "Compare rapport_a.pdf et rapport_b.pdf.",
        ["rapport_a.pdf", "rapport_b.pdf"],
        llm=LLMScripte(),
    )
    assert not r.succes
    assert "aucun document ne fournit" in r.message.lower()
    assert not r.sources


def test_map_llm_echoue_sur_un_document_partiel(monkeypatch) -> None:
    """1 document en échec technique, l'autre exploitable -> PARTIEL, pas un
    refus global (l'échec technique bloquant ne concerne que ZÉRO preuve)."""
    _corpus_2(monkeypatch)
    r = comparer(
        "Compare rapport_a.pdf et rapport_b.pdf.",
        ["rapport_a.pdf", "rapport_b.pdf"],
        llm=make_llm_map_echoue("rapport_b.pdf"),
    )
    assert r.succes
    assert r.donnees.get("statut") == "partiel"
    assert "rapport_b.pdf" in r.donnees["comparaison"]["documents_en_echec"]


def test_reduce_llm_echoue_refus(monkeypatch) -> None:
    _corpus_2(monkeypatch)

    class _LLM:
        def invoke(self, messages, think: bool | None = None):
            if "COMPARAISON" in messages[0].content:
                raise RuntimeError("REDUCE KO")
            return LLMScripte().invoke(messages, think=think)

    r = comparer(
        "Compare rapport_a.pdf et rapport_b.pdf.",
        ["rapport_a.pdf", "rapport_b.pdf"],
        llm=_LLM(),
    )
    assert not r.succes
    assert "impossible" in r.message.lower()


def test_llm_absent_refus(monkeypatch) -> None:
    _corpus_2(monkeypatch)
    r = comparer("Compare rapport_a.pdf et rapport_b.pdf.", ["rapport_a.pdf", "rapport_b.pdf"], llm=None)
    assert not r.succes


# --------------------------------------------------------------------------
# Couverture INTÉGRALE des documents volumineux (multi-lots)
# --------------------------------------------------------------------------


def test_compare_documents_volumineux_couverts_en_entier(monkeypatch) -> None:
    monkeypatch.setattr(multidoc_pipeline, "LIMITE_CARACTERES_LOT", 200)
    cabler_corpus(
        monkeypatch,
        multidoc_pipeline,
        fiches={"rapport_a.pdf": "A", "rapport_b.pdf": "B"},
        passages_par_doc={
            "A": [passage("A", i, f"élément A {i} " * 5, page=i) for i in range(1, 9)],
            "B": [passage("B", i, f"élément B {i} " * 5, page=i) for i in range(1, 7)],
        },
    )
    r = comparer(
        "Compare rapport_a.pdf et rapport_b.pdf.",
        ["rapport_a.pdf", "rapport_b.pdf"],
        llm=LLMScripte(),
    )
    assert r.succes
    par_doc = r.donnees["par_document"]
    assert par_doc["rapport_a.pdf"]["nombre_lots"] > 1
    assert par_doc["rapport_b.pdf"]["nombre_lots"] > 1
    # provenance : pages issues de lots au-delà du premier
    pages = {s.page for s in r.sources}
    assert pages - {1}  # au moins une page > 1 citée
    assert {s.doc_id for s in r.sources} <= {"A", "B"}
