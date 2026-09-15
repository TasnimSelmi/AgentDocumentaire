"""
Preuve de bout en bout (audit budget LLM — étape de vérification).

Aucune modification du code de production. Deux preuves déterministes,
sans Ollama ni Qdrant :

  1. Une information placée UNIQUEMENT dans le DERNIER lot d'un document long
     (>= 3 lots) survit :
        document complet -> dernier lot -> MAP -> agrégation intra-document
        -> REDUCE -> réponse finale COMPARE / SYNTHESIZE
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
from tools.compare import comparer
from tools.synthesize import synthetiser_documents
from tests.agent._multidoc_fakes import (
    EST_AGREGATION_INTRA_DOC,
    EST_MAP_LOT,
    cabler_corpus,
    passage,
)

_CIT = re.compile(r"\[(D\d+S\d+)\]")
_MARQUEUR = "LA VALEUR SECRETE EST 4242"
_PAGE_MARQUEUR = 99


class _LLMTracant:
    """
    LLM déterministe qui PROPAGE toute citation vue :
      - MAP d'un lot   -> renvoie chaque [D_S_] du lot + recopie le marqueur
                          s'il est présent dans le lot ;
      - agrégation     -> renvoie toutes les citations vues + le marqueur ;
      - REDUCE         -> JSON dont chaque champ liste TOUTES les citations
                          vues, marqueur inclus.
    Ne cite jamais rien qu'il n'ait pas vu.
    """

    def __init__(self) -> None:
        self.appels: list[tuple[str, str]] = []

    def invoke(self, messages) -> AIMessage:
        systeme, utilisateur = messages[0].content, messages[1].content
        self.appels.append((systeme, utilisateur))
        cites = list(dict.fromkeys(_CIT.findall(utilisateur)))
        porte_marqueur = _MARQUEUR in utilisateur

        if EST_MAP_LOT(systeme):
            corps = " ".join(f"[{c}]" for c in cites)
            suffixe = f" — {_MARQUEUR}" if porte_marqueur else ""
            return AIMessage(content=f"Éléments du lot : {corps}{suffixe}")

        if EST_AGREGATION_INTRA_DOC(systeme):
            corps = " ".join(f"[{c}]" for c in cites)
            suffixe = f" {_MARQUEUR}" if porte_marqueur else ""
            return AIMessage(content=f"Liste consolidée : {corps}{suffixe}")

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
    #     par l'agrégation intra-document, puis par le REDUCE.
    vu_map = any(EST_MAP_LOT(s) and _MARQUEUR in u for s, u in llm.appels)
    vu_agg = any(EST_AGREGATION_INTRA_DOC(s) and _MARQUEUR in u for s, u in llm.appels)
    vu_reduce = any(
        ("COMPARAISON" in s or "SYNTHÈSE TRANSVERSALE" in s) and _MARQUEUR in u
        for s, u in llm.appels
    )
    assert vu_map and vu_agg and vu_reduce

    # (f) aucune fuite : toutes les sources appartiennent aux 2 documents visés
    assert {s.doc_id for s in r.sources} <= {"LONG", "AUT"}


# =========================================================================
# H.2 — au-delà de NB_LOTS_MAX : refus explicite, pas de troncature
# =========================================================================


@pytest.mark.parametrize("operation", ["compare", "synthesize"])
def test_au_dela_de_nb_lots_max_refus_explicite_pas_de_repli(monkeypatch, operation):
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

    assert not r.succes  # refus
    assert r.outil == operation  # pas de bascule vers un autre outil
    # message honnête : moins de 2 documents exploitables
    assert "moins de deux documents" in r.message.lower()
    assert "trop_long.pdf" in " ".join(r.donnees.get("documents_en_echec", []))
    # rien n'a été présenté comme une comparaison/synthèse complète
    assert operation not in r.donnees  # pas de bloc "comparaison"/"synthese"


def test_au_dela_de_nb_lots_max_map_document_directement(monkeypatch):
    """map_document seul : le document dépassant la limite est marqué `echec`
    avec le nombre réel de lots (compté, pas tronqué)."""
    monkeypatch.setattr(mp, "LIMITE_CARACTERES_LOT", 100)
    monkeypatch.setattr(mp, "NB_LOTS_MAX", 4)
    passages = [passage("X", i, "q" * 95, page=i) for i in range(1, 20)]
    cabler_corpus(monkeypatch, mp, fiches={"x.pdf": "X"}, passages_par_doc={"X": passages})

    m = mp.map_document(
        mp.DocumentCible(index=1, doc_id="X", libelle="x.pdf", nom_fichier="x.pdf"),
        "?",
        llm=_LLMTracant(),
    )
    assert m.echec is not None
    assert "trop volumineux" in m.echec
    assert m.nombre_lots == 19  # tous les lots comptés
    assert not m.utilisable
    assert m.citations_valides == []  # aucun traitement partiel présenté
