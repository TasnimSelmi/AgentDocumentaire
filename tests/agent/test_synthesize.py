"""
Tests de la capacité SYNTHESIZE (P1.5) — `src.agent.synthesize`.

Aucun Ollama, aucun Qdrant. Le REDUCE ne reçoit que les sorties MAP
validées. Les divergences entre documents sont conservées explicitement.
"""

from __future__ import annotations

import pytest

from src.agent import multidoc_pipeline
from src.tools.synthesize import ResultatSynthese, synthetiser_documents
from tests.agent._multidoc_fakes import (
    SANS_EVIDENCE,
    LLMScripte,
    cabler_corpus,
    passage,
)


def _corpus_2(monkeypatch, *, b_sans_evidence: bool = False) -> None:
    cabler_corpus(
        monkeypatch,
        multidoc_pipeline,
        fiches={"note_a.pdf": "A", "note_b.pdf": "B"},
        passages_par_doc={
            "A": [passage("A", 1, "Recommandation : investir dans la formation.", page=1)],
            "B": [
                passage(
                    "B",
                    1,
                    SANS_EVIDENCE if b_sans_evidence else "Recommandation : réduire les coûts.",
                    page=1,
                )
            ],
        },
    )


def _corpus_3(monkeypatch) -> None:
    cabler_corpus(
        monkeypatch,
        multidoc_pipeline,
        fiches={"note_a.pdf": "A", "note_b.pdf": "B", "note_c.pdf": "C"},
        passages_par_doc={
            "A": [passage("A", 1, "Point A.")],
            "B": [passage("B", 1, "Point B.")],
            "C": [passage("C", 1, "Point C.")],
        },
    )


def test_synthese_deux_documents(monkeypatch) -> None:
    _corpus_2(monkeypatch)
    llm = LLMScripte()
    r = synthetiser_documents(
        "Fais une synthèse de note_a.pdf et note_b.pdf.",
        ["note_a.pdf", "note_b.pdf"],
        llm=llm,
    )
    assert r.succes
    syn = ResultatSynthese(**r.donnees["synthese"])
    assert syn.themes_communs and syn.synthese_transversale
    assert r.sources
    # REDUCE ne voit pas le corpus brut
    for _s, u in [(s, u) for s, u in llm.appels if "SYNTHÈSE TRANSVERSALE" in s]:
        assert "investir dans la formation" not in u


def test_synthese_trois_documents(monkeypatch) -> None:
    _corpus_3(monkeypatch)
    r = synthetiser_documents(
        "Consolide note_a.pdf, note_b.pdf et note_c.pdf.",
        ["note_a.pdf", "note_b.pdf", "note_c.pdf"],
        llm=LLMScripte(),
    )
    assert r.succes
    assert len(r.donnees["synthese"]["documents"]) == 3


def test_divergences_conservees(monkeypatch) -> None:
    _corpus_2(monkeypatch)
    r = synthetiser_documents(
        "Synthèse des recommandations de note_a.pdf et note_b.pdf.",
        ["note_a.pdf", "note_b.pdf"],
        llm=LLMScripte(),
    )
    assert r.succes
    assert r.donnees["synthese"]["divergences"], "les désaccords ne doivent pas être lissés"


def test_provenance_dans_le_perimetre(monkeypatch) -> None:
    _corpus_3(monkeypatch)  # A, B, C indexés
    r = synthetiser_documents(
        "Synthèse de note_a.pdf et note_b.pdf.",
        ["note_a.pdf", "note_b.pdf"],
        llm=LLMScripte(),
    )
    assert r.succes
    assert {s.doc_id for s in r.sources} <= {"A", "B"}
    assert "D3" not in " ".join(r.donnees["citations_utilisees"])


def test_citation_hors_scope_filtree(monkeypatch) -> None:
    _corpus_2(monkeypatch)
    r = synthetiser_documents(
        "Synthèse de note_a.pdf et note_b.pdf.",
        ["note_a.pdf", "note_b.pdf"],
        llm=LLMScripte(citation_hors_scope="D7S1"),
    )
    assert r.succes
    texte = " ".join(
        [
            *r.donnees["synthese"]["themes_communs"],
            *r.donnees["synthese"]["elements_complementaires"],
            *r.donnees["synthese"]["divergences"],
            r.donnees["synthese"]["synthese_transversale"] or "",
        ]
    )
    assert "D7S1" not in texte
    assert "D7S1" not in " ".join(r.donnees["citations_utilisees"])


def test_document_absent_refus(monkeypatch) -> None:
    _corpus_2(monkeypatch)
    r = synthetiser_documents(
        "Synthèse de note_a.pdf et note_absente.pdf.",
        ["note_a.pdf", "note_absente.pdf"],
        llm=LLMScripte(),
    )
    assert not r.succes
    assert r.donnees.get("motif") == "document_introuvable"


def test_au_dela_limite_refus(monkeypatch) -> None:
    cabler_corpus(
        monkeypatch,
        multidoc_pipeline,
        fiches={f"n{i}.pdf": f"D{i}" for i in range(6)},
        passages_par_doc={f"D{i}": [passage(f"D{i}", 1, "x")] for i in range(6)},
    )
    r = synthetiser_documents(
        "Synthèse de 5 notes.", [f"n{i}.pdf" for i in range(5)], llm=LLMScripte()
    )
    assert not r.succes
    assert r.donnees.get("motif") == "au_dela_limite"


def test_un_document_sans_evidence_refus(monkeypatch) -> None:
    _corpus_2(monkeypatch, b_sans_evidence=True)
    r = synthetiser_documents(
        "Synthèse de note_a.pdf et note_b.pdf.",
        ["note_a.pdf", "note_b.pdf"],
        llm=LLMScripte(),
    )
    assert not r.succes
    assert "note_b.pdf" in " ".join(r.donnees.get("documents_sans_evidence", []))


def test_synthese_documents_volumineux_couverts_en_entier(monkeypatch) -> None:
    monkeypatch.setattr(multidoc_pipeline, "LIMITE_CARACTERES_LOT", 180)
    cabler_corpus(
        monkeypatch,
        multidoc_pipeline,
        fiches={"note_a.pdf": "A", "note_b.pdf": "B"},
        passages_par_doc={
            "A": [passage("A", i, f"reco A {i} " * 6, page=i) for i in range(1, 8)],
            "B": [passage("B", i, f"reco B {i} " * 6, page=i) for i in range(1, 6)],
        },
    )
    r = synthetiser_documents(
        "Synthèse de note_a.pdf et note_b.pdf.",
        ["note_a.pdf", "note_b.pdf"],
        llm=LLMScripte(),
    )
    assert r.succes
    assert r.donnees["par_document"]["note_a.pdf"]["nombre_lots"] > 1
    assert {s.page for s in r.sources} - {1}
    assert {s.doc_id for s in r.sources} <= {"A", "B"}
