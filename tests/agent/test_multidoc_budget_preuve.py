"""
Preuve de bout en bout (audit budget LLM — étape de vérification).

Aucune modification du code de production. Deux preuves déterministes,
sans Ollama ni Qdrant :

  1. Une information placée UNIQUEMENT dans le DERNIER lot d'un document long
     (>= 3 lots) survit :
        document complet -> dernier lot -> MAP structuré -> fusion
        déterministe -> REDUCE -> réponse finale COMPARE / SYNTHESIZE
     avec la bonne provenance (citation + page).

  2. Un document produisant plus de NB_LOTS_MAX lots est REFUSÉ explicitement
     (jamais tronqué, jamais de repli SEARCH).
"""

from __future__ import annotations

import json
import re

import pytest
from langchain_core.messages import AIMessage

from src.agent import multidoc_pipeline as mp
from src.tools.compare import comparer
from src.tools.synthesize import synthetiser_documents
from tests.agent._multidoc_fakes import (
    EST_MAP_LOT,
    EST_PLAN,
    cabler_corpus,
    passage,
)

_CIT = re.compile(r"\[(D\d+S\d+)\]")
_MARQUEUR = "LA VALEUR SECRETE EST 4242"
_PAGE_MARQUEUR = 99


def _task_spec(operation: str = "compare") -> mp.TaskSpec:
    return mp._plan_repli(operation)


def _axes_depuis_prompt_map(utilisateur: str) -> list[str]:
    axes: list[str] = []
    dans_bloc = False
    for ligne in utilisateur.splitlines():
        if ligne.startswith("AXES DE LA DEMANDE"):
            dans_bloc = True
            continue
        if dans_bloc:
            if ligne.startswith("- "):
                axes.append(ligne[2:].strip())
            else:
                break
    return axes


class _LLMTracant:
    """
    LLM déterministe qui PROPAGE toute citation vue :
      - PLAN           -> plan générique borné (axes fixes, sans contenu de
                          document) ;
      - MAP d'un lot   -> JSON structuré listant chaque [D_S_] du lot,
                          contenu recopiant le marqueur s'il est présent ;
      - REDUCE         -> JSON dont chaque champ liste TOUTES les citations
                          vues, marqueur inclus.
    Ne cite jamais rien qu'il n'ait pas vu. Aucune agrégation intra-document
    LLM (fusion déterministe côté pipeline depuis P1.8).
    """

    def __init__(self) -> None:
        self.appels: list[tuple[str, str]] = []

    def invoke(self, messages, think: bool | None = None) -> AIMessage:
        systeme, utilisateur = messages[0].content, messages[1].content
        self.appels.append((systeme, utilisateur))

        if EST_PLAN(systeme):
            objet = {
                "objectif": "Analyser les documents fournis selon les axes ci-dessous.",
                "axes": ["éléments pertinents"],
                "informations_attendues": [],
            }
            return AIMessage(content=json.dumps(objet, ensure_ascii=False))

        cites = list(dict.fromkeys(_CIT.findall(utilisateur)))
        porte_marqueur = _MARQUEUR in utilisateur

        if EST_MAP_LOT(systeme):
            axes = _axes_depuis_prompt_map(utilisateur) or ["éléments pertinents"]
            corps = " ".join(f"[{c}]" for c in cites)
            suffixe = f" — {_MARQUEUR}" if porte_marqueur else ""
            objet = {
                "pertinent": True,
                "elements": [
                    {
                        "axe": axes[0],
                        "contenu": f"Éléments du lot : {corps}{suffixe}",
                        "citations": cites,
                    }
                ],
                "warnings": [],
            }
            return AIMessage(content=json.dumps(objet, ensure_ascii=False))

        # REDUCE (COMPARAISON ou SYNTHÈSE TRANSVERSALE)
        toutes = " ".join(f"[{c}]" for c in cites)
        m = f" {_MARQUEUR}" if porte_marqueur else ""
        if "COMPARAISON" in systeme:
            objet = {
                "points_communs": [f"Constat commun. {toutes}{m}"],
                "differences": [f"Différence rapportée. {toutes}{m}"],
                "positions_par_document": {},
                "contradictions": [f"Divergence. {toutes}"],
                "conclusion": None,
            }
        else:
            objet = {
                "themes_communs": [f"Thème. {toutes}{m}"],
                "elements_complementaires": [f"Complément. {toutes}"],
                "divergences": [f"Divergence. {toutes}"],
                "synthese_transversale": f"Synthèse. {toutes}{m}",
            }
        return AIMessage(content=json.dumps(objet, ensure_ascii=False))


def _document_long_avec_marqueur_au_dernier_lot(nb_passages: int = 12):
    """Renvoie une liste de passages : filler jusqu'à l'avant-dernier, le
    marqueur (page 99) dans le tout dernier passage."""
    passages = [
        passage("LONG", i, f"paragraphe de remplissage numéro {i}. " * 4, page=i)
        for i in range(1, nb_passages)
    ]
    passages.append(passage("LONG", nb_passages, _MARQUEUR + ".", page=_PAGE_MARQUEUR))
    return passages


# =========================================================================
# H.1 — l'information du DERNIER lot survit jusqu'à la réponse finale
# =========================================================================


@pytest.mark.parametrize("operation", ["compare", "synthesize"])
def test_information_du_dernier_lot_survit_jusqua_la_reponse(monkeypatch, operation):
    # LIMITE réduite -> un document de ~12 passages produit plusieurs lots.
    monkeypatch.setattr(mp, "LIMITE_CARACTERES_LOT", 300)

    longs = _document_long_avec_marqueur_au_dernier_lot(12)
    autre = [passage("AUT", 1, "Le sujet est traité différemment ici.", page=1)]
    cabler_corpus(
        monkeypatch,
        mp,
        fiches={"long.pdf": "LONG", "autre.pdf": "AUT"},
        passages_par_doc={"LONG": longs, "AUT": autre},
    )

    llm = _LLMTracant()
    fn = comparer if operation == "compare" else synthetiser_documents
    cle = "comparaison" if operation == "compare" else "synthese"

    r = fn("Quelle est la valeur secrète ?", ["long.pdf", "autre.pdf"], llm=llm)

    assert r.succes, r.message

    # (a) le document long a bien été découpé en >= 3 lots
    nb_lots = r.donnees["par_document"]["long.pdf"]["nombre_lots"]
    assert nb_lots >= 3, f"attendu >= 3 lots, obtenu {nb_lots}"

    # (b) la citation du DERNIER passage (marqueur) a traversé toute la chaîne
    derniere_citation = f"D1S{len(longs)}"
    assert derniere_citation in r.donnees["par_document"]["long.pdf"]["citations"]
    assert derniere_citation in r.donnees["citations_utilisees"]

    # (c) provenance : la source citée porte la bonne PAGE (99)
    pages_citees = {s.page for s in r.sources}
    assert _PAGE_MARQUEUR in pages_citees, pages_citees
    src_marqueur = next(s for s in r.sources if s.page == _PAGE_MARQUEUR)
    assert _MARQUEUR in src_marqueur.extrait

    # (d) le texte de la réponse finale porte le marqueur (info réellement remontée)
    bloc = r.donnees[cle]
    texte_final = " ".join(
        v if isinstance(v, str) else " ".join(v)
        for v in bloc.values()
        if isinstance(v, (str, list))
    )
    assert "4242" in texte_final

    # (e) chaîne complète tracée : le marqueur est passé par un MAP de lot,
    #     PUIS par le REDUCE. Il n'y a plus d'agrégation intra-document LLM
    #     depuis P1.8 (fusion déterministe, pure Python, cf.
    #     `multidoc_pipeline._fusionner_elements` — 0 appel LLM, donc rien à
    #     tracer ici pour cette étape).
    vu_map = any(EST_MAP_LOT(s) and _MARQUEUR in u for s, u in llm.appels)
    vu_reduce = any(
        ("COMPARAISON" in s or "SYNTHÈSE TRANSVERSALE" in s) and _MARQUEUR in u
        for s, u in llm.appels
    )
    assert vu_map and vu_reduce

    # (f) aucune fuite : toutes les sources appartiennent aux 2 documents visés
    assert {s.doc_id for s in r.sources} <= {"LONG", "AUT"}


# =========================================================================
# H.2 — au-delà de NB_LOTS_MAX : refus explicite, pas de troncature
# =========================================================================


@pytest.mark.parametrize("operation", ["compare", "synthesize"])
def test_au_dela_de_nb_lots_max_refus_explicite_pas_de_repli(monkeypatch, operation):
    """NB_LOTS_MAX reste intact : le document trop volumineux est TOUJOURS
    rejeté explicitement (jamais tronqué, jamais traité silencieusement).
    Depuis l'audit long-documents (section D), ce rejet ne bloque plus la
    demande entière quand l'AUTRE document reste exploitable : réponse
    PARTIELLE sourcée sur ce qui est prouvé, plutôt qu'un refus global."""
    monkeypatch.setattr(mp, "LIMITE_CARACTERES_LOT", 120)
    monkeypatch.setattr(mp, "NB_LOTS_MAX", 3)

    trop_long = [passage("TL", i, "z" * 110, page=i) for i in range(1, 12)]  # ~11 lots
    autre = [passage("AUT", 1, "contenu court", page=1)]
    cabler_corpus(
        monkeypatch,
        mp,
        fiches={"trop_long.pdf": "TL", "autre.pdf": "AUT"},
        passages_par_doc={"TL": trop_long, "AUT": autre},
    )

    fn = comparer if operation == "compare" else synthetiser_documents
    r = fn("?", ["trop_long.pdf", "autre.pdf"], llm=_LLMTracant())
    cle = "comparaison" if operation == "compare" else "synthese"

    assert r.succes  # réponse partielle, pas un refus global
    assert r.outil == operation
    assert r.donnees.get("statut") == "partiel"
    # NB_LOTS_MAX reste appliqué : le document trop volumineux est bien
    # rejeté explicitement (jamais silencieusement traité ni tronqué).
    assert "trop_long.pdf" in r.donnees[cle]["documents_en_echec"]
    assert "trop volumineux" in r.donnees["par_document"]["trop_long.pdf"]["echec"]
    # seul le document exploitable (autre.pdf) est réellement sourcé.
    assert r.sources
    assert {s.doc_id for s in r.sources} == {"AUT"}


def test_au_dela_de_nb_lots_max_map_document_directement(monkeypatch):
    """map_document seul : le document dépassant la limite est marqué `echec`
    avec le nombre réel de lots (compté, pas tronqué)."""
    monkeypatch.setattr(mp, "LIMITE_CARACTERES_LOT", 100)
    monkeypatch.setattr(mp, "NB_LOTS_MAX", 4)
    passages = [passage("X", i, "q" * 95, page=i) for i in range(1, 20)]
    cabler_corpus(monkeypatch, mp, fiches={"x.pdf": "X"}, passages_par_doc={"X": passages})

    m = mp.map_document(
        mp.DocumentCible(index=1, doc_id="X", libelle="x.pdf", nom_fichier="x.pdf"),
        _task_spec(),
        llm=_LLMTracant(),
    )
    assert m.echec is not None
    assert "trop volumineux" in m.echec
    assert m.nombre_lots == 19  # tous les lots comptés
    assert not m.utilisable
    assert m.citations_valides == []  # aucun traitement partiel présenté


# =========================================================================
# §2.5 / cas (b) — le prompt REDUCE reste dans le budget OU refus explicite
# =========================================================================


def _corpus_deux_docs_longs(monkeypatch, *, limite_lot: int = 400):
    monkeypatch.setattr(mp, "LIMITE_CARACTERES_LOT", limite_lot)
    a = [passage("DA", i, f"analyse A {i} " * 8, page=i) for i in range(1, 10)]
    b = [passage("DB", i, f"analyse B {i} " * 8, page=i) for i in range(1, 8)]
    cabler_corpus(
        monkeypatch,
        mp,
        fiches={"doc_a.pdf": "DA", "doc_b.pdf": "DB"},
        passages_par_doc={"DA": a, "DB": b},
    )


@pytest.mark.parametrize("operation", ["compare", "synthesize"])
def test_reduce_dans_le_budget_ou_refus_explicite(monkeypatch, operation):
    """Deux documents longs. Si le prompt REDUCE tient dans le budget -> succès
    et AUCUN prompt envoyé ne dépasse le budget. Si on force un budget trop
    petit -> refus explicite motif='budget_reduce_depasse', jamais d'envoi
    oversized, jamais de repli SEARCH."""
    _corpus_deux_docs_longs(monkeypatch)
    fn = comparer if operation == "compare" else synthetiser_documents
    module = __import__(
        "src.tools.compare" if operation == "compare" else "src.tools.synthesize",
        fromlist=["x"],
    )

    # --- cas nominal : budget réel, doit tenir -------------------------------
    llm = _LLMTracant()
    r = fn("Question de fond ?", ["doc_a.pdf", "doc_b.pdf"], llm=llm)
    assert r.succes, r.message
    budget = mp.budget_caracteres_entree_llm()
    trop_gros = [
        (len(s) + len(u), s[:40]) for s, u in llm.appels if len(s) + len(u) > budget
    ]
    assert not trop_gros, f"prompt(s) envoyé(s) hors budget : {trop_gros}"

    # --- budget artificiellement minuscule : refus explicite ---------------
    monkeypatch.setattr(module, "budget_caracteres_entree_llm", lambda: 200)
    llm2 = _LLMTracant()
    r2 = fn("Question de fond ?", ["doc_a.pdf", "doc_b.pdf"], llm=llm2)
    assert not r2.succes
    assert r2.donnees.get("motif") == "budget_reduce_depasse"
    assert r2.outil == operation  # pas de bascule / pas de repli SEARCH
    # le REDUCE n'a jamais été envoyé
    marqueur = "COMPARAISON" if operation == "compare" else "SYNTHÈSE TRANSVERSALE"
    assert not any(marqueur in s for s, _ in llm2.appels)


# =========================================================================
# cas (f) — aucune troncature silencieuse : TOUT prompt <= budget
# =========================================================================


@pytest.mark.parametrize("operation", ["compare", "synthesize"])
def test_aucun_prompt_ne_depasse_le_budget(monkeypatch, operation):
    _corpus_deux_docs_longs(monkeypatch, limite_lot=600)
    fn = comparer if operation == "compare" else synthetiser_documents
    llm = _LLMTracant()
    r = fn("Analyse comparée ?", ["doc_a.pdf", "doc_b.pdf"], llm=llm)
    assert r.succes, r.message
    budget = mp.budget_caracteres_entree_llm()
    assert llm.appels, "au moins un appel LLM"
    for systeme, utilisateur in llm.appels:
        assert len(systeme) + len(utilisateur) <= budget


# =========================================================================
# §2.3 / cas (c) — passage individuel oversized => refus du document,
#                  puis abstention explicite de comparer/synthetiser
# =========================================================================


@pytest.mark.parametrize("operation", ["compare", "synthesize"])
def test_passage_oversized_refuse_le_document_puis_abstention(monkeypatch, operation):
    # Budget choisi pour que la cohérence 2.1 PASSE (lot plein 16000 + système
    # ~1118 < budget) mais qu'un passage géant dépasse le seuil passage
    # (budget - coût incompressible).
    monkeypatch.setattr(mp, "budget_caracteres_entree_llm", lambda: 22_000)

    geant = [passage("G", 1, "X" * 21_000, page=1)]  # bloc ~21044 > seuil ~20882
    autre = [passage("H", 1, "contenu court et pertinent", page=1)]
    cabler_corpus(
        monkeypatch,
        mp,
        fiches={"geant.pdf": "G", "autre.pdf": "H"},
        passages_par_doc={"G": geant, "H": autre},
    )

    # map_document seul : refus explicite du document géant
    m = mp.map_document(
        mp.DocumentCible(index=1, doc_id="G", libelle="geant.pdf", nom_fichier="geant.pdf"),
        _task_spec(),
        llm=_LLMTracant(),
    )
    assert m.echec is not None
    assert "passage unique hors budget" in m.echec
    assert "D1S1" in m.echec  # citation identifiée
    assert not m.utilisable

    # bout en bout : le document géant reste rejeté explicitement (jamais
    # tronqué), mais l'AUTRE document exploitable permet désormais une
    # réponse PARTIELLE sourcée plutôt qu'un refus global (audit
    # long-documents, section D) — pas de crash, pas de repli SEARCH.
    fn = comparer if operation == "compare" else synthetiser_documents
    r = fn("?", ["geant.pdf", "autre.pdf"], llm=_LLMTracant())
    cle = "comparaison" if operation == "compare" else "synthese"

    assert r.succes
    assert r.outil == operation
    assert r.donnees.get("statut") == "partiel"
    assert "geant.pdf" in r.donnees[cle]["documents_en_echec"]
    assert "passage unique hors budget" in r.donnees["par_document"]["geant.pdf"]["echec"]
    assert r.sources
    assert {s.doc_id for s in r.sources} == {"H"}
