"""
Tests pytest de l'outil `classify`.

`tests/tools/test_classify_smoke.py` (script manuel `main()`, non collecté
par pytest, exerce un vrai LLM via `construire_llm()`) reste tel quel : ce
fichier ajoute les cas minimums en pytest déterministe, nécessaires avant de
brancher `classify` dans LangGraph — aucun Ollama, aucun Qdrant.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.tools.base import ContexteOutil, SourceOutil
from src.tools.classify import definir_classify

CATEGORIES = ["Rapport ESG", "Rapport financier", "Contrat"]


def _source(doc_id: str, texte: str = "Contenu.") -> SourceOutil:
    return SourceOutil(
        doc_id=doc_id,
        source=f"{doc_id}.pdf",
        nom_fichier=f"{doc_id}.pdf",
        page=1,
        categorie="",
        score=0.9,
        extrait=texte,
    )


class LLMScripte:
    def __init__(self, repondre):
        self._repondre = repondre

    def invoke(self, messages):
        return AIMessage(content=self._repondre(messages[0].content, messages[1].content))


class LLMExplose:
    def invoke(self, messages):
        raise RuntimeError("Ollama injoignable.")


def _contexte(llm, sources: list[SourceOutil]) -> ContexteOutil:
    return ContexteOutil(question="peu importe", llm=llm, sources=sources)


# ===========================================================================
# 1-2 — document unique connu, classification valide
# ===========================================================================


def test_document_unique_classification_valide():
    llm = LLMScripte(
        lambda s, u: (
            '{"categorie": "Rapport ESG", "confiance": 0.9, '
            '"sources": ["S1"], "justification": "ok"}'
        )
    )
    contexte = _contexte(llm, [_source("A")])

    resultat = definir_classify().executer(contexte=contexte, categories=CATEGORIES)

    assert resultat.succes
    assert resultat.donnees["categorie"] == "Rapport ESG"
    assert resultat.donnees["document"] == "A.pdf"


# ===========================================================================
# 3 — citation valide
# ===========================================================================


def test_citation_valide_rattache_la_bonne_source():
    llm = LLMScripte(
        lambda s, u: (
            '{"categorie": "Contrat", "confiance": 0.8, '
            '"sources": ["S1"], "justification": "ok"}'
        )
    )
    contexte = _contexte(llm, [_source("A")])

    resultat = definir_classify().executer(contexte=contexte, categories=CATEGORIES)

    assert resultat.succes
    assert resultat.donnees["citations"] == ["S1"]
    assert len(resultat.sources) == 1
    assert resultat.sources[0].doc_id == "A"


# ===========================================================================
# 4 — catégorie sans citation -> rejet
# ===========================================================================


def test_categorie_sans_citation_rejetee():
    llm = LLMScripte(
        lambda s, u: '{"categorie": "Contrat", "confiance": 0.9, "sources": [], "justification": "ok"}'
    )
    contexte = _contexte(llm, [_source("A")])

    resultat = definir_classify().executer(contexte=contexte, categories=CATEGORIES)

    # Comportement existant, non modifié : un rejet pour manque de provenance
    # reste un succès sans catégorie (comme un search sans résultat), pas un échec.
    assert resultat.succes
    assert resultat.donnees["categorie"] is None
    assert any("source" in a.lower() for a in resultat.avertissements)


# ===========================================================================
# 5 — document ambigu (cible fournie mais introuvable/non résolue) -> refus
# ===========================================================================


def test_document_ambigu_refuse():
    llm = LLMScripte(
        lambda s, u: '{"categorie": "Contrat", "confiance": 0.9, "sources": ["S1"], "justification": "ok"}'
    )
    contexte = _contexte(llm, [_source("A"), _source("B")])

    resultat = definir_classify().executer(
        contexte=contexte, categories=CATEGORIES, document="inexistant"
    )

    assert not resultat.succes
    assert "inexistant" in resultat.message


# ===========================================================================
# 6 — plusieurs documents sans cible -> refus
# ===========================================================================


def test_plusieurs_documents_sans_cible_refuse():
    llm = LLMScripte(
        lambda s, u: '{"categorie": "Contrat", "confiance": 0.9, "sources": ["S1"], "justification": "ok"}'
    )
    contexte = _contexte(llm, [_source("A"), _source("B")])

    resultat = definir_classify().executer(contexte=contexte, categories=CATEGORIES)

    assert not resultat.succes
    assert set(resultat.donnees["documents_disponibles"]) == {"A.pdf", "B.pdf"}


# ===========================================================================
# 7 — erreur LLM -> échec propre
# ===========================================================================


def test_erreur_llm_echec_propre():
    contexte = _contexte(LLMExplose(), [_source("A")])

    resultat = definir_classify().executer(contexte=contexte, categories=CATEGORIES)

    assert not resultat.succes
    assert "Classification impossible" in resultat.message


# ===========================================================================
# 8 — aucune catégorie fiable -> comportement explicite (succès, categorie=None)
# ===========================================================================


def test_aucune_categorie_fiable_comportement_explicite():
    llm = LLMScripte(
        lambda s, u: '{"categorie": null, "confiance": 0.0, "sources": [], "justification": "insuffisant"}'
    )
    contexte = _contexte(llm, [_source("A")])

    resultat = definir_classify().executer(contexte=contexte, categories=CATEGORIES)

    assert resultat.succes
    assert resultat.donnees["categorie"] is None
    assert resultat.donnees["confiance"] == 0.0
    assert "Aucune catégorie fiable" in resultat.message


def test_categorie_inventee_rejetee():
    llm = LLMScripte(
        lambda s, u: (
            '{"categorie": "Categorie inexistante", "confiance": 0.9, '
            '"sources": ["S1"], "justification": "ok"}'
        )
    )
    contexte = _contexte(llm, [_source("A")])

    resultat = definir_classify().executer(contexte=contexte, categories=CATEGORIES)

    assert resultat.succes
    assert resultat.donnees["categorie"] is None
    assert any("appartient pas" in a for a in resultat.avertissements)
