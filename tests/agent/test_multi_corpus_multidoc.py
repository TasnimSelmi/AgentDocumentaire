"""
Isolation multi-corpus — COMPARE / SYNTHESIZE (§12/§13 de l'audit
multi-corpus).

Aucun Qdrant réel : `catalogue` / `charger_document` de
`src.agent.multidoc_pipeline` sont doublés par corpus logique. Deux corpus,
"corpus_a" / "corpus_b", contiennent chacun deux documents de MÊME NOM
("rapport_a.pdf", "rapport_b.pdf") mais un contenu et un secret distincts —
le scénario exact de l'audit. Si rapport_a/rapport_b existaient dans
plusieurs corpus, le resolver ne doit voir QUE ceux du corpus actif.
"""

from __future__ import annotations

from src.agent import multidoc_pipeline
from src.rag.retrieval import DocumentInconnu, PerimetreDocumentaire
from src.tools.compare import comparer
from src.tools.synthesize import synthetiser_documents
from tests.agent._multidoc_fakes import LLMScripte, passage

ALPHA = "ALPHA_SECRET_123"
BETA = "BETA_SECRET_456"

# corpus_a : rapport_a.pdf -> doc_id "A1", rapport_b.pdf -> doc_id "A2" (contenu ALPHA)
# corpus_b : rapport_a.pdf -> doc_id "B1", rapport_b.pdf -> doc_id "B2" (contenu BETA)
_FICHES_PAR_CORPUS = {
    "corpus_a": {"rapport_a.pdf": "A1", "rapport_b.pdf": "A2"},
    "corpus_b": {"rapport_a.pdf": "B1", "rapport_b.pdf": "B2"},
}
_PASSAGES_PAR_CORPUS = {
    "corpus_a": {
        "A1": [passage("A1", 1, f"Objectif principal. {ALPHA}", page=1)],
        "A2": [passage("A2", 1, "Second document du corpus A.", page=1)],
    },
    "corpus_b": {
        "B1": [passage("B1", 1, f"Objectif principal. {BETA}", page=1)],
        "B2": [passage("B2", 1, "Second document du corpus B.", page=1)],
    },
}


class _FauxCatalogueCorpus:
    def __init__(self, fiches: dict[str, str]) -> None:
        self._fiches = fiches

    def par_identifiant(self, identifiant: str):
        from src.rag.retrieval import FicheDocument

        doc_id = self._fiches.get(str(identifiant).strip())
        if doc_id is None:
            return None
        return FicheDocument(document_id=doc_id, champ_id="nom_fichier", nom_fichier=identifiant, titre="")


def _cabler_multi_corpus(monkeypatch) -> None:
    def _catalogue(profil=None, corpus_id=None):
        fiches = _FICHES_PAR_CORPUS.get(corpus_id, {})
        return _FauxCatalogueCorpus(fiches)

    def _charger_document(doc_id, corpus_id=None):
        passages = _PASSAGES_PAR_CORPUS.get(corpus_id, {})
        if doc_id in passages:
            return passages[doc_id]
        raise DocumentInconnu(doc_id)

    monkeypatch.setattr(multidoc_pipeline, "catalogue", _catalogue)
    monkeypatch.setattr(multidoc_pipeline, "charger_document", _charger_document)


def test_resoudre_cibles_corpus_a_ne_voit_que_ses_propres_documents(monkeypatch):
    _cabler_multi_corpus(monkeypatch)

    resolution_a = multidoc_pipeline.resoudre_cibles(
        ["rapport_a.pdf", "rapport_b.pdf"], corpus_id="corpus_a"
    )
    resolution_b = multidoc_pipeline.resoudre_cibles(
        ["rapport_a.pdf", "rapport_b.pdf"], corpus_id="corpus_b"
    )

    assert resolution_a.exploitable and resolution_b.exploitable
    assert {d.doc_id for d in resolution_a.documents} == {"A1", "A2"}
    assert {d.doc_id for d in resolution_b.documents} == {"B1", "B2"}


def test_compare_corpus_a_ne_recupere_jamais_beta(monkeypatch):
    _cabler_multi_corpus(monkeypatch)

    r = comparer(
        "Compare les objectifs.",
        ["rapport_a.pdf", "rapport_b.pdf"],
        llm=LLMScripte(),
        corpus_id="corpus_a",
    )

    assert r.succes, r.message
    assert {s.doc_id for s in r.sources} <= {"A1", "A2"}
    texte = str(r.donnees)
    assert BETA not in texte


def test_compare_corpus_b_ne_recupere_jamais_alpha(monkeypatch):
    _cabler_multi_corpus(monkeypatch)

    r = comparer(
        "Compare les objectifs.",
        ["rapport_a.pdf", "rapport_b.pdf"],
        llm=LLMScripte(),
        corpus_id="corpus_b",
    )

    assert r.succes, r.message
    assert {s.doc_id for s in r.sources} <= {"B1", "B2"}
    texte = str(r.donnees)
    assert ALPHA not in texte


def test_synthesize_isolation_stricte_entre_corpus(monkeypatch):
    _cabler_multi_corpus(monkeypatch)

    r_a = synthetiser_documents(
        "Synthétise les objectifs.",
        ["rapport_a.pdf", "rapport_b.pdf"],
        llm=LLMScripte(),
        corpus_id="corpus_a",
    )
    r_b = synthetiser_documents(
        "Synthétise les objectifs.",
        ["rapport_a.pdf", "rapport_b.pdf"],
        llm=LLMScripte(),
        corpus_id="corpus_b",
    )

    assert r_a.succes and r_b.succes
    assert {s.doc_id for s in r_a.sources} <= {"A1", "A2"}
    assert {s.doc_id for s in r_b.sources} <= {"B1", "B2"}
    assert BETA not in str(r_a.donnees)
    assert ALPHA not in str(r_b.donnees)


def test_meme_nom_de_document_resout_le_bon_corpus_pour_compare(monkeypatch):
    """§13 — "rapport_a.pdf" existe dans les deux corpus avec un contenu
    différent : le resolver du corpus actif ne doit voir QUE le sien."""
    _cabler_multi_corpus(monkeypatch)

    resolution_a = multidoc_pipeline.resoudre_cibles(
        ["rapport_a.pdf", "rapport_b.pdf"], corpus_id="corpus_a"
    )
    doc_a = next(d for d in resolution_a.documents if d.nom_fichier == "rapport_a.pdf")
    assert doc_a.doc_id == "A1"

    resolution_b = multidoc_pipeline.resoudre_cibles(
        ["rapport_a.pdf", "rapport_b.pdf"], corpus_id="corpus_b"
    )
    doc_b = next(d for d in resolution_b.documents if d.nom_fichier == "rapport_a.pdf")
    assert doc_b.doc_id == "B1"
