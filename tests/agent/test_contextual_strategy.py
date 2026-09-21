"""
Tests de la stratégie contextuelle (`src.agent.contextual_strategy`) — un
seul appel LLM par requête, à partir d'un retrieval déjà dédoublonné.

Aucun Ollama, aucun Qdrant : `rechercher_passages` est remplacé par une
doublure qui renvoie un `RapportRecherche` fabriqué ; le LLM est une
doublure scriptée (même patron que `tests/tools/test_summarize.py`).

Ces tests vérifient exclusivement le NOUVEAU chemin (`strategy="contextual"`
/ `resumer_contextuel`). Les tests existants de `comparer()`/
`synthetiser_documents()` (chemin `map_reduce`, défaut inchangé) restent
dans `tests/agent/test_compare.py` / `test_synthesize.py` et continuent de
passer sans modification (voir `report.md`).
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage

from src.agent import contextual_strategy as cs
from src.rag.retrieval import Passage, RapportRecherche
from src.tools import compare, synthesize


# ===========================================================================
# Fabriques
# ===========================================================================


def _passage(
    doc_id: str,
    chunk_index: int,
    texte: str,
    *,
    page: int = 1,
    nom_fichier: str | None = None,
    rang: int | None = None,
) -> Passage:
    rang = rang if rang is not None else chunk_index + 1
    nom = nom_fichier or f"{doc_id}.pdf"
    return Passage(
        citation=f"S{rang}",
        rang=rang,
        point_id=f"{doc_id}-{chunk_index}",
        doc_id=doc_id,
        chunk_index=chunk_index,
        texte=texte,
        source=nom,
        nom_fichier=nom,
        page=page,
        categorie="autre",
        score_recherche=1.0 / rang,
        score_reranking=None,
        payload={},
    )


def _rapport(
    passages: list[Passage],
    *,
    candidats_recuperes: int | None = None,
    motif_absence: str | None = None,
) -> RapportRecherche:
    return RapportRecherche(
        requete="peu importe",
        profil="generic",
        filtres={},
        passages=passages,
        candidats_recuperes=candidats_recuperes if candidats_recuperes is not None else len(passages),
        reranking_utilise=True,
        seuil_applique=None,
        duree_secondes=0.01,
        perimetre=None,
        diversification_active=True,
        motif_absence=motif_absence,
    )


class LLMScripte:
    """LLM factice : délègue la réponse à une fonction fournie par le test.
    Compte les appels réels — c'est cette liste que les tests
    `test_contextuel_compte_appels_llm*` inspectent. Accepte `think=`
    (ignoré) : `executer_contextual` appelle `invoquer_llm(..., reasoning=False)`,
    ce qui se traduit par `llm.invoke(messages, think=False)`."""

    def __init__(self, repondre):
        self._repondre = repondre
        self.appels: list[tuple[str, str]] = []
        self.appels_think: list[bool | None] = []

    def invoke(self, messages, think: bool | None = None):
        systeme, utilisateur = messages[0].content, messages[1].content
        self.appels.append((systeme, utilisateur))
        self.appels_think.append(think)
        return AIMessage(content=self._repondre(systeme, utilisateur, len(self.appels)))


class _LLMExplose:
    def invoke(self, messages, think: bool | None = None):
        raise RuntimeError("Ollama injoignable.")


def _cite_tout(systeme: str, utilisateur: str, numero_appel: int) -> str:
    """LLM « honnête » : ne cite que ce qu'il voit réellement dans le prompt
    — jamais une citation qu'il n'aurait pas dû connaître."""
    citations = list(dict.fromkeys(re.findall(r"\[S\d+\]", utilisateur)))
    return "Réponse synthétique. " + " ".join(citations)


def _cable_retrieval(monkeypatch, rapport: RapportRecherche, capture: dict | None = None):
    """Remplace `rechercher_passages` par une doublure qui renvoie
    `rapport` et, si `capture` est fourni, y enregistre les arguments
    d'appel (pour vérifier l'isolation par corpus_id, le top_k, etc.)."""

    def _faux_rechercher_passages(**kwargs):
        if capture is not None:
            capture.update(kwargs)
        return rapport

    monkeypatch.setattr(cs, "rechercher_passages", _faux_rechercher_passages)


# ===========================================================================
# Tests
# ===========================================================================


def test_contextuel_document_unique(monkeypatch):
    passages = [_passage("rapport", 0, "Le chiffre d'affaires a progressé de 12%.")]
    _cable_retrieval(monkeypatch, _rapport(passages))
    llm = LLMScripte(_cite_tout)

    resultat = cs.executer_contextual(
        operation="summarize", question="Résume ce document.",
        corpus_id="default", llm=llm,
    )

    assert resultat.succes
    assert len(llm.appels) == 1
    assert resultat.donnees["llm_calls"] == 1
    assert resultat.sources and resultat.sources[0].doc_id == "rapport"


def test_contextuel_document_long_budget(monkeypatch):
    # Beaucoup de passages volumineux : dépasse largement le budget de
    # caractères d'un seul appel LLM.
    passages = [
        _passage("long", i, "Contenu répétitif du document. " * 200, rang=i + 1)
        for i in range(40)
    ]
    _cable_retrieval(monkeypatch, _rapport(passages, candidats_recuperes=120))
    llm = LLMScripte(_cite_tout)

    resultat = cs.executer_contextual(
        operation="summarize", question="Résume ce document long.",
        corpus_id="default", llm=llm,
    )

    assert resultat.succes
    assert len(llm.appels) == 1, "budget dépassé ou non, un seul appel LLM"
    assert resultat.donnees["selected_chunks"] < resultat.donnees["deduplicated_chunks"]
    assert resultat.donnees["retrieved_chunks"] == 120
    assert resultat.donnees["context_size"] <= cs.budget_caracteres_entree_llm()


def test_contextuel_question_ciblee(monkeypatch):
    capture: dict = {}
    passages = [_passage("doc", 0, "Section cybersécurité : détail du dispositif.")]
    _cable_retrieval(monkeypatch, _rapport(passages), capture=capture)
    llm = LLMScripte(_cite_tout)

    cs.executer_contextual(
        operation="summarize",
        question="Résume la partie concernant la cybersécurité.",
        corpus_id="default", llm=llm,
    )

    assert capture["requete"] == "Résume la partie concernant la cybersécurité."
    assert capture["resolution_document"] is True


def test_contextuel_contexte_vide_abstention(monkeypatch):
    _cable_retrieval(monkeypatch, _rapport([], motif_absence="aucun document ne traite ce sujet"))
    llm = LLMScripte(_cite_tout)

    resultat = cs.executer_contextual(
        operation="summarize", question="Sujet absent du corpus.",
        corpus_id="default", llm=llm,
    )

    assert not resultat.succes
    assert len(llm.appels) == 0, "abstention déterministe AVANT tout appel LLM"
    assert "aucun document ne traite ce sujet" in resultat.message


def test_contextuel_citations(monkeypatch):
    passages = [_passage("doc", 0, "Fait vérifiable numéro un.")]
    _cable_retrieval(monkeypatch, _rapport(passages))

    # Le LLM invente une citation absente du contexte : doit être rejetée,
    # pas transmise comme provenance valide.
    llm_invente = LLMScripte(lambda s, u, n: "Une affirmation non vérifiable. [S9]")
    resultat = cs.executer_contextual(
        operation="summarize", question="Résume.", corpus_id="default", llm=llm_invente,
    )
    assert not resultat.succes
    assert "aucune citation documentaire valide" in resultat.message
    assert resultat.donnees["citations_invalides"] == ["S9"]

    # Le LLM cite correctement : succès, provenance conservée.
    llm_honnete = LLMScripte(_cite_tout)
    resultat2 = cs.executer_contextual(
        operation="summarize", question="Résume.", corpus_id="default", llm=llm_honnete,
    )
    assert resultat2.succes
    assert resultat2.donnees["citations_valides"] == ["S1"]


def test_contextuel_isolation_corpus(monkeypatch):
    capture: dict = {}
    passages = [_passage("doc", 0, "Contenu.")]
    _cable_retrieval(monkeypatch, _rapport(passages), capture=capture)
    llm = LLMScripte(_cite_tout)

    cs.executer_contextual(
        operation="summarize", question="Résume.",
        corpus_id="corpus-rh", llm=llm,
    )

    assert capture["corpus_id"] == "corpus-rh"


def test_contextuel_compte_appels_llm(monkeypatch):
    """`llm_calls` doit refléter EXACTEMENT le nombre d'appels réels au LLM,
    quel que soit le nombre de passages sélectionnés."""
    for nb_passages in (1, 5, 20):
        passages = [_passage("doc", i, f"Passage {i}.") for i in range(nb_passages)]
        _cable_retrieval(monkeypatch, _rapport(passages))
        llm = LLMScripte(_cite_tout)

        resultat = cs.executer_contextual(
            operation="synthesize", question="Synthétise.",
            corpus_id="default", llm=llm, documents=["doc"],
        )

        assert len(llm.appels) == 1
        assert resultat.donnees["llm_calls"] == 1


def test_contextuel_budget_contexte(monkeypatch):
    passages = [_passage("doc", i, "x" * 500, rang=i + 1) for i in range(10)]
    _cable_retrieval(monkeypatch, _rapport(passages))
    llm = LLMScripte(_cite_tout)

    resultat = cs.executer_contextual(
        operation="summarize", question="Résume.", corpus_id="default", llm=llm,
    )

    assert resultat.succes
    budget = cs.budget_caracteres_entree_llm()
    assert resultat.donnees["context_size"] <= budget, (
        "le contexte envoyé au LLM ne doit jamais dépasser le budget déclaré"
    )


def test_contextuel_deduplication(monkeypatch):
    """`retrieval.py` a déjà dédoublonné/diversifié (`_selectionner_diversifie`,
    `etendre_contexte`) — ce test vérifie que la stratégie contextuelle
    EXPOSE fidèlement retrieved/deduplicated/selected sans re-dupliquer ni
    perdre d'information de comptage, sans réimplémenter de déduplication."""
    passages = [_passage("doc", i, f"Contenu utile {i}.") for i in range(6)]
    # candidats_recuperes (avant dédup) > len(passages) (après dédup) :
    # simule des chunks voisins/doublons déjà filtrés par retrieval.py.
    _cable_retrieval(monkeypatch, _rapport(passages, candidats_recuperes=18))
    llm = LLMScripte(_cite_tout)

    resultat = cs.executer_contextual(
        operation="summarize", question="Résume.", corpus_id="default", llm=llm,
    )

    assert resultat.donnees["retrieved_chunks"] == 18
    assert resultat.donnees["deduplicated_chunks"] == 6
    assert resultat.donnees["selected_chunks"] <= 6


def test_contextuel_multi_document(monkeypatch):
    passages = [
        _passage("doc-a", 0, "Le document A traite du budget.", nom_fichier="doc-a.pdf", rang=1),
        _passage("doc-b", 0, "Le document B traite des risques.", nom_fichier="doc-b.pdf", rang=2),
    ]
    _cable_retrieval(monkeypatch, _rapport(passages))
    llm = LLMScripte(_cite_tout)

    resultat = cs.executer_contextual(
        operation="synthesize",
        question="Résume les principaux éléments de ces documents.",
        corpus_id="default", llm=llm, documents=["doc-a.pdf", "doc-b.pdf"],
    )

    assert resultat.succes
    assert set(resultat.donnees["documents"]) == {"doc-a.pdf", "doc-b.pdf"}
    # L'identité du document reste explicite dans le prompt envoyé au LLM.
    _, utilisateur = llm.appels[0]
    assert "doc-a.pdf" in utilisateur and "doc-b.pdf" in utilisateur


# ===========================================================================
# Câblage depuis compare.py / synthesize.py (`strategy="contextual"`)
# ===========================================================================


def test_comparer_strategy_contextual_delegue(monkeypatch):
    appels: list[dict] = []

    def _faux_executer_contextual(**kwargs):
        appels.append(kwargs)
        from src.tools.base import ResultatOutil

        return ResultatOutil(outil="compare_contextual", succes=True, message="ok")

    monkeypatch.setattr(cs, "executer_contextual", _faux_executer_contextual)
    monkeypatch.setattr(
        "src.agent.contextual_strategy.executer_contextual", _faux_executer_contextual
    )

    resultat = compare.comparer(
        "Compare ces deux documents.", ["a.pdf", "b.pdf"],
        llm=object(), corpus_id="c1", strategy="contextual",
    )

    assert resultat.succes
    assert appels and appels[0]["operation"] == "compare"
    assert appels[0]["corpus_id"] == "c1"
    assert list(appels[0]["documents"]) == ["a.pdf", "b.pdf"]


def test_synthetiser_strategy_contextual_delegue(monkeypatch):
    appels: list[dict] = []

    def _faux_executer_contextual(**kwargs):
        appels.append(kwargs)
        from src.tools.base import ResultatOutil

        return ResultatOutil(outil="synthesize_contextual", succes=True, message="ok")

    monkeypatch.setattr(
        "src.agent.contextual_strategy.executer_contextual", _faux_executer_contextual
    )

    resultat = synthesize.synthetiser_documents(
        "Synthétise ces documents.", ["a.pdf", "b.pdf"],
        llm=object(), corpus_id="c1", strategy="contextual",
    )

    assert resultat.succes
    assert appels and appels[0]["operation"] == "synthesize"


def test_contextuel_resumer_contextuel_wrapper(monkeypatch):
    passages = [_passage("doc", 0, "Contenu.")]
    _cable_retrieval(monkeypatch, _rapport(passages))
    llm = LLMScripte(_cite_tout)

    resultat = cs.resumer_contextuel("Résume.", llm=llm, corpus_id="default")

    assert resultat.succes
    assert resultat.outil == "summarize_contextual"


def test_contextuel_desactive_le_raisonnement(monkeypatch):
    """`executer_contextual` doit toujours appeler `invoquer_llm` avec
    `reasoning=False` — diagnostic de suivi (report.md) : sans ce paramètre,
    `think` reste au défaut du modèle (activé pour qwen3), ce qui peut
    consommer tout `num_predict` en raisonnement caché et renvoyer une
    réponse vide (reproduit en direct sur compare/court/contextual)."""
    passages = [_passage("doc", 0, "Fait vérifiable.")]
    _cable_retrieval(monkeypatch, _rapport(passages))
    llm = LLMScripte(_cite_tout)

    resultat = cs.executer_contextual(
        operation="summarize", question="Résume.", corpus_id="default", llm=llm,
    )

    assert resultat.succes
    assert llm.appels_think == [False]


# ===========================================================================
# Mode HYBRID (COMPARE / SYNTHESIZE) — `tenter_avec_repli`
# ===========================================================================


def _resultat(outil: str, succes: bool, message: str = "", **donnees) -> object:
    from src.tools.base import ResultatOutil

    if succes:
        return ResultatOutil(outil=outil, succes=True, message=message, donnees=donnees)
    return ResultatOutil.echec(outil, message, **donnees)


def _repli_espion(reponse=None):
    """Fabrique un `repli` factice qui enregistre s'il a été appelé."""
    appels: list[int] = []

    def _repli():
        appels.append(1)
        from src.tools.base import ResultatOutil

        return reponse or ResultatOutil(
            outil="compare", succes=True, message="map_reduce ok", donnees={}
        )

    _repli.appels = appels
    return _repli


def test_hybrid_contextual_succes_aucun_repli(monkeypatch):
    monkeypatch.setattr(
        cs, "executer_contextual",
        lambda **kw: _resultat("compare_contextual", True, "ok", selected_chunks=3),
    )
    repli = _repli_espion()

    resultat = cs.tenter_avec_repli(
        operation="compare", question="Q", corpus_id="default", llm=object(),
        documents=["a.pdf", "b.pdf"], profil_domaine=None, repli=repli,
    )

    assert resultat.succes
    assert resultat.message == "ok"
    assert repli.appels == [], "aucun double appel : contextual a réussi"


def test_hybrid_citation_invalide_declenche_repli(monkeypatch):
    monkeypatch.setattr(
        cs, "executer_contextual",
        lambda **kw: _resultat(
            "compare_contextual", False,
            "La réponse produite ne contient aucune citation documentaire valide.",
            citations_invalides=[], selected_chunks=5,
        ),
    )
    repli = _repli_espion()

    resultat = cs.tenter_avec_repli(
        operation="compare", question="Q", corpus_id="default", llm=object(),
        documents=["a.pdf"], profil_domaine=None, repli=repli,
    )

    assert repli.appels == [1], "repli appelé exactement une fois"
    assert resultat.succes
    assert resultat.donnees["fallback_depuis_contextual"] is True
    assert "citation" in resultat.donnees["motif_fallback_contextual"].lower()
    assert any("Repli map_reduce" in a for a in resultat.avertissements)


def test_hybrid_exception_contextual_declenche_repli(monkeypatch):
    monkeypatch.setattr(
        cs, "executer_contextual",
        lambda **kw: _resultat(
            "compare_contextual", False,
            "Réponse impossible : Ollama injoignable.",
            selected_chunks=4,  # après sélection, avant l'exception LLM
        ),
    )
    repli = _repli_espion()

    resultat = cs.tenter_avec_repli(
        operation="compare", question="Q", corpus_id="default", llm=object(),
        documents=["a.pdf"], profil_domaine=None, repli=repli,
    )

    assert repli.appels == [1]
    assert resultat.donnees["fallback_depuis_contextual"] is True


def test_hybrid_abstention_pas_de_repli(monkeypatch):
    monkeypatch.setattr(
        cs, "executer_contextual",
        lambda **kw: _resultat(
            "compare_contextual", False,
            "Aucune réponse fiable ne peut être établie : aucun passage pertinent trouvé.",
            selected_chunks=0, retrieved_chunks=0, deduplicated_chunks=0,
        ),
    )
    repli = _repli_espion()

    resultat = cs.tenter_avec_repli(
        operation="compare", question="Q", corpus_id="default", llm=object(),
        documents=["a.pdf"], profil_domaine=None, repli=repli,
    )

    assert repli.appels == [], "abstention déterministe : jamais de repli inutile"
    assert not resultat.succes
    assert "fallback_depuis_contextual" not in resultat.donnees


def test_hybrid_refus_resolution_pas_de_repli(monkeypatch):
    """`DocumentInconnu` etc. : `donnees` vide (pas de `selected_chunks`) —
    même règle que l'abstention, `.get(..., 0)` retombe sur 0."""
    monkeypatch.setattr(
        cs, "executer_contextual",
        lambda **kw: _resultat("compare_contextual", False, "Document inconnu du corpus."),
    )
    repli = _repli_espion()

    resultat = cs.tenter_avec_repli(
        operation="compare", question="Q", corpus_id="default", llm=object(),
        documents=["inconnu.pdf"], profil_domaine=None, repli=repli,
    )

    assert repli.appels == []
    assert not resultat.succes


def test_hybrid_comparer_strategy_delegue(monkeypatch):
    appels: list[dict] = []

    def _faux_tenter_avec_repli(**kwargs):
        appels.append(kwargs)
        from src.tools.base import ResultatOutil

        return ResultatOutil(outil="compare", succes=True, message="ok")

    monkeypatch.setattr(
        "src.agent.contextual_strategy.tenter_avec_repli", _faux_tenter_avec_repli
    )

    resultat = compare.comparer(
        "Compare ces documents.", ["a.pdf", "b.pdf"],
        llm=object(), corpus_id="c1", strategy="hybrid",
    )

    assert resultat.succes
    assert appels and appels[0]["operation"] == "compare"
    assert appels[0]["corpus_id"] == "c1"
    assert callable(appels[0]["repli"])


def test_hybrid_synthetiser_strategy_delegue(monkeypatch):
    appels: list[dict] = []

    def _faux_tenter_avec_repli(**kwargs):
        appels.append(kwargs)
        from src.tools.base import ResultatOutil

        return ResultatOutil(outil="synthesize", succes=True, message="ok")

    monkeypatch.setattr(
        "src.agent.contextual_strategy.tenter_avec_repli", _faux_tenter_avec_repli
    )

    resultat = synthesize.synthetiser_documents(
        "Synthétise ces documents.", ["a.pdf", "b.pdf"],
        llm=object(), corpus_id="c1", strategy="hybrid",
    )

    assert resultat.succes
    assert appels and appels[0]["operation"] == "synthesize"
    assert appels[0]["corpus_id"] == "c1"


def test_contextuel_llm_error_non_masque(monkeypatch):
    passages = [_passage("doc", 0, "Contenu.")]
    _cable_retrieval(monkeypatch, _rapport(passages))

    resultat = cs.executer_contextual(
        operation="summarize", question="Résume.",
        corpus_id="default", llm=_LLMExplose(),
    )

    assert not resultat.succes
    assert "Ollama injoignable" in resultat.message
