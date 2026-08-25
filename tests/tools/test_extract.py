"""
Tests pytest de l'outil `extract` (Action 04) : mode historique (Cas B,
``ContexteOutil.sources``) et mode document complet (Cas A, ``documents=[...]``).

Aucun Ollama, aucun Qdrant réel : ``catalogue``/``charger_document`` sont
injectés exactement comme dans `tests/tools/test_summarize.py` et
`tests/tools/test_classify_document_complet.py` (même patron), et le LLM
est une doublure scriptée.
"""

from __future__ import annotations

import json
import re

import pytest
from langchain_core.messages import AIMessage

from src.rag.retrieval import CollectionIndisponible, DocumentInconnu, Passage, PerimetreDocumentaire
from src.tools import extract
from src.tools.base import ContexteOutil, SourceOutil


# ===========================================================================
# Fabriques génériques
# ===========================================================================


def _passage(doc_id: str, chunk_index: int, texte: str, *, page: int = 1, nom_fichier: str = "doc.pdf") -> Passage:
    rang = chunk_index + 1
    return Passage(
        citation=f"S{rang}",
        rang=rang,
        point_id=f"{doc_id}-{chunk_index}",
        doc_id=doc_id,
        chunk_index=chunk_index,
        texte=texte,
        source=nom_fichier,
        nom_fichier=nom_fichier,
        page=page,
        categorie="",
        score_recherche=0.0,
        score_reranking=None,
        payload={},
    )


def _source(doc_id: str, texte: str = "Contenu.", *, page: int | None = 1, nom_fichier: str | None = None) -> SourceOutil:
    nom_fichier = nom_fichier if nom_fichier is not None else f"{doc_id}.pdf"
    return SourceOutil(
        doc_id=doc_id,
        source=nom_fichier,
        nom_fichier=nom_fichier,
        page=page,
        categorie="",
        score=0.9,
        extrait=texte,
    )


def _perimetre_exact(doc_id: str) -> PerimetreDocumentaire:
    return PerimetreDocumentaire(statut="exact", valeurs_filtre=(doc_id,), libelles=(doc_id,))


def _perimetre_compatible(doc_ids: tuple[str, ...]) -> PerimetreDocumentaire:
    return PerimetreDocumentaire(statut="compatible", valeurs_filtre=doc_ids, libelles=doc_ids)


class _FauxCatalogue:
    def __init__(self, perimetre_ou_exception):
        self._p = perimetre_ou_exception

    def perimetre_explicite(self, documents):
        if isinstance(self._p, Exception):
            raise self._p
        return self._p


def _resoudre_vers(monkeypatch, doc_id: str, passages: list[Passage]) -> None:
    monkeypatch.setattr(extract, "get_profil", lambda: None)
    monkeypatch.setattr(extract, "catalogue", lambda profil=None: _FauxCatalogue(_perimetre_exact(doc_id)))
    monkeypatch.setattr(
        extract,
        "charger_document",
        lambda cible: passages if cible == doc_id else (_ for _ in ()).throw(DocumentInconnu(cible)),
    )


def _citations_du_prompt(texte: str) -> list[str]:
    vues: list[str] = []
    for m in re.findall(r"\[(S\d+)\]", texte):
        if m not in vues:
            vues.append(m)
    return vues


def _rep(par_champ: dict[str, list[tuple[str, list[str]]]]) -> dict:
    """Construit une réponse JSON LLM : {champ: [(valeur, [citations]), ...]}."""
    return {
        "extractions": {
            champ: {
                "valeurs": [
                    {"valeur": valeur, "sources": citations, "justification": "ok"}
                    for valeur, citations in entrees
                ]
            }
            for champ, entrees in par_champ.items()
        }
    }


class LLMExtractions:
    """LLM factice : renvoie la n-ième réponse scriptée au n-ième appel."""

    def __init__(self, reponses: list[dict]):
        self._reponses = reponses
        self.appels: list[str] = []

    def invoke(self, messages):
        _systeme, utilisateur = messages[0].content, messages[1].content
        self.appels.append(utilisateur)
        indice = len(self.appels) - 1
        payload = self._reponses[indice] if indice < len(self._reponses) else {"extractions": {}}
        return AIMessage(content=json.dumps(payload))


class _LLMExplose:
    def invoke(self, messages):
        raise RuntimeError("Ollama injoignable.")


class _LLMJSONInvalide:
    def invoke(self, messages):
        return AIMessage(content="Ceci n'est pas du JSON valide {{{")


class _LLMExplosePartiellement:
    """Explose aux appels dont le numéro (1-indexé) est dans ``indices_erreur``,
    répond normalement (via ``reponses``) sinon."""

    def __init__(self, indices_erreur: set[int], reponses: list[dict]):
        self._indices_erreur = indices_erreur
        self._reponses = reponses
        self.appels = 0

    def invoke(self, messages):
        self.appels += 1
        if self.appels in self._indices_erreur:
            raise RuntimeError(f"Timeout lot {self.appels}.")
        indice = self.appels - 1
        payload = self._reponses[indice] if indice < len(self._reponses) else {"extractions": {}}
        return AIMessage(content=json.dumps(payload))


def _contexte_b(llm, sources: list[SourceOutil]) -> ContexteOutil:
    return ContexteOutil(question="peu importe", llm=llm, sources=sources)


def _contexte_a(llm) -> ContexteOutil:
    return ContexteOutil(question="peu importe", llm=llm, sources=[])


def _executer_b(contexte, *, champs, document=None, instruction=None):
    return extract.definir_extract().executer(
        contexte=contexte, champs=champs, document=document, instruction=instruction
    )


def _executer_a(contexte, *, champs, documents=("A",), instruction=None):
    return extract.definir_extract().executer(
        contexte=contexte, champs=champs, documents=list(documents), instruction=instruction
    )


# ===========================================================================
# A — extraction simple (Cas B)
# ===========================================================================


def test_extraction_simple_cas_b():
    llm = LLMExtractions([_rep({"montant": [("1000 EUR", ["S1"])]})])
    contexte = _contexte_b(llm, [_source("A", "Montant : 1000 EUR")])

    resultat = _executer_b(contexte, champs=["montant"])

    assert resultat.succes
    assert resultat.donnees["extractions"]["montant"]["trouve"] is True
    assert resultat.donnees["extractions"]["montant"]["valeur"] == "1000 EUR"
    assert resultat.donnees["extractions"]["montant"]["valeur_unique"] is True


# ===========================================================================
# B — plusieurs champs
# ===========================================================================


def test_plusieurs_champs_cas_b():
    llm = LLMExtractions(
        [_rep({"fournisseur": [("ACME", ["S1"])], "date": [("2026-01-01", ["S1"])]})]
    )
    contexte = _contexte_b(llm, [_source("A", "Fournisseur ACME, date 2026-01-01")])

    resultat = _executer_b(contexte, champs=["fournisseur", "date"])

    assert resultat.succes
    assert resultat.donnees["extractions"]["fournisseur"]["valeur"] == "ACME"
    assert resultat.donnees["extractions"]["date"]["valeur"] == "2026-01-01"
    assert resultat.donnees["nombre_trouves"] == 2
    assert resultat.donnees["nombre_demandes"] == 2


# ===========================================================================
# C — champ absent
# ===========================================================================


def test_champ_absent():
    llm = LLMExtractions([_rep({"montant": [("1000 EUR", ["S1"])], "date_echeance": []})])
    contexte = _contexte_b(llm, [_source("A", "Montant : 1000 EUR")])

    resultat = _executer_b(contexte, champs=["montant", "date_echeance"])

    assert resultat.succes
    assert resultat.donnees["extractions"]["date_echeance"]["trouve"] is False
    assert resultat.donnees["extractions"]["date_echeance"]["valeur"] is None
    assert resultat.donnees["extractions"]["date_echeance"]["valeurs"] == []


# ===========================================================================
# D — plusieurs valeurs légitimes
# ===========================================================================


def test_plusieurs_valeurs_legitimes():
    llm = LLMExtractions(
        [_rep({"parties": [("Société A", ["S1"]), ("Société B", ["S1"])]})]
    )
    contexte = _contexte_b(llm, [_source("A", "Entre Société A et Société B")])

    resultat = _executer_b(contexte, champs=["parties"])

    assert resultat.succes
    extraction = resultat.donnees["extractions"]["parties"]
    assert extraction["trouve"] is True
    assert extraction["valeur_unique"] is False
    assert extraction["valeur"] is None  # jamais un choix arbitraire entre plusieurs valeurs
    valeurs = {v["valeur"] for v in extraction["valeurs"]}
    assert valeurs == {"Société A", "Société B"}
    assert resultat.donnees["nombre_champs_multiples"] == 1


# ===========================================================================
# E — contradiction
# ===========================================================================


def test_contradiction_deux_valeurs_conservees_sourcees(monkeypatch):
    """
    2 lots trouvent 2 dates DIFFÉRENTES pour date_signature : les deux
    restent présentes, sourcées séparément, aucune n'est éliminée
    silencieusement (voir CAS 3 de la mission).
    """
    passages = [_passage("A", i, f"contenu {i}") for i in range(2)]
    _resoudre_vers(monkeypatch, "A", passages)
    monkeypatch.setattr(extract, "LIMITE_CARACTERES_LOT", 60)  # 2 lots

    llm = LLMExtractions(
        [
            _rep({"date_signature": [("2026-01-12", ["S1"])]}),
            _rep({"date_signature": [("2026-02-12", ["S2"])]}),
        ]
    )
    resultat = _executer_a(_contexte_a(llm), champs=["date_signature"])

    assert resultat.succes
    extraction = resultat.donnees["extractions"]["date_signature"]
    assert extraction["valeur_unique"] is False
    assert extraction["valeur"] is None
    valeurs_citations = {v["valeur"]: v["citations"] for v in extraction["valeurs"]}
    assert valeurs_citations == {"2026-01-12": ["S1"], "2026-02-12": ["S2"]}


# ===========================================================================
# F — citation valide
# ===========================================================================


def test_citation_valide_rattache_la_bonne_source():
    llm = LLMExtractions([_rep({"montant": [("1000 EUR", ["S1"])]})])
    contexte = _contexte_b(llm, [_source("A", "Montant : 1000 EUR")])

    resultat = _executer_b(contexte, champs=["montant"])

    assert resultat.succes
    assert resultat.donnees["extractions"]["montant"]["valeurs"][0]["citations"] == ["S1"]
    assert len(resultat.sources) == 1
    assert resultat.sources[0].doc_id == "A"


# ===========================================================================
# G — citation inventée rejetée
# ===========================================================================


def test_citation_inventee_rejetee():
    llm = LLMExtractions([_rep({"montant": [("1000 EUR", ["S99"])]})])
    contexte = _contexte_b(llm, [_source("A", "Montant : 1000 EUR")])

    resultat = _executer_b(contexte, champs=["montant"])

    assert resultat.succes
    assert resultat.donnees["extractions"]["montant"]["trouve"] is False
    assert any("aucune source" in a.lower() for a in resultat.avertissements)


# ===========================================================================
# H — JSON LLM invalide
# ===========================================================================


def test_json_invalide_cas_b_resultat_vide_trace():
    contexte = _contexte_b(_LLMJSONInvalide(), [_source("A")])

    resultat = _executer_b(contexte, champs=["montant"])

    assert resultat.succes  # échec technique = résultat vide, pas un crash
    assert resultat.donnees["extractions"]["montant"]["trouve"] is False
    assert any("extraction impossible" in a.lower() for a in resultat.avertissements)


def test_erreur_llm_cas_b_resultat_vide_trace():
    contexte = _contexte_b(_LLMExplose(), [_source("A")])

    resultat = _executer_b(contexte, champs=["montant"])

    assert resultat.succes
    assert resultat.donnees["extractions"]["montant"]["trouve"] is False
    assert any("extraction impossible" in a.lower() for a in resultat.avertissements)


# ===========================================================================
# I — document inconnu
# ===========================================================================


def test_document_inconnu_echec_propre(monkeypatch):
    monkeypatch.setattr(extract, "get_profil", lambda: None)
    monkeypatch.setattr(extract, "catalogue", lambda profil=None: _FauxCatalogue(DocumentInconnu("Z introuvable")))

    resultat = _executer_a(_contexte_a(LLMExtractions([])), champs=["montant"], documents=("Z",))

    assert not resultat.succes
    assert "introuvable" in resultat.message


# ===========================================================================
# J — document ambigu
# ===========================================================================


def test_document_ambigu_echec_propre(monkeypatch):
    monkeypatch.setattr(extract, "get_profil", lambda: None)
    monkeypatch.setattr(
        extract, "catalogue", lambda profil=None: _FauxCatalogue(_perimetre_compatible(("A", "B")))
    )

    resultat = _executer_a(_contexte_a(LLMExtractions([])), champs=["montant"], documents=("A et B",))

    assert not resultat.succes
    assert "mélange jamais" in resultat.message


def test_plusieurs_documents_cas_b_sans_cible_refuse():
    llm = LLMExtractions([])
    contexte = _contexte_b(llm, [_source("A"), _source("B")])

    resultat = _executer_b(contexte, champs=["montant"])

    assert not resultat.succes
    assert set(resultat.donnees["documents_disponibles"]) == {"A.pdf", "B.pdf"}


# ===========================================================================
# K — document vide
# ===========================================================================


def test_document_resolu_sans_contenu_echec_propre(monkeypatch):
    _resoudre_vers(monkeypatch, "A", [])

    resultat = _executer_a(_contexte_a(LLMExtractions([])), champs=["montant"])

    assert not resultat.succes
    assert "aucun contenu indexé" in resultat.message


def test_collection_indisponible_echec_propre(monkeypatch):
    monkeypatch.setattr(extract, "get_profil", lambda: None)
    monkeypatch.setattr(extract, "catalogue", lambda profil=None: _FauxCatalogue(_perimetre_exact("A")))
    monkeypatch.setattr(
        extract, "charger_document", lambda cible: (_ for _ in ()).throw(CollectionIndisponible("absente"))
    )

    resultat = _executer_a(_contexte_a(LLMExtractions([])), champs=["montant"])

    assert not resultat.succes
    assert "Corpus indisponible" in resultat.message


# ===========================================================================
# L / M — document long multi-lots, aucun passage abandonné
# ===========================================================================


def test_document_volumineux_couverture_totale_sans_troncature(monkeypatch):
    passages = [_passage("A", i, f"contenu {i} " * 8) for i in range(60)]
    _resoudre_vers(monkeypatch, "A", passages)
    monkeypatch.setattr(extract, "LIMITE_CARACTERES_LOT", 400)

    llm = LLMExtractions([_rep({"champ": []})] * 50)
    resultat = _executer_a(_contexte_a(llm), champs=["champ"])

    assert resultat.succes
    assert resultat.donnees["nombre_passages"] == 60
    assert resultat.donnees["nombre_lots"] > 1

    citations_vues: set[str] = set()
    for utilisateur in llm.appels:
        vues_ce_lot = set(_citations_du_prompt(utilisateur))
        assert not (citations_vues & vues_ce_lot)  # N — aucune collision entre lots
        citations_vues |= vues_ce_lot
    assert citations_vues == {p.citation for p in passages}


def test_partitionner_un_passage_depasse_le_budget_seul():
    petit = _passage("A", 0, "court")
    enorme = _passage("A", 1, "x" * 5_000)
    petit2 = _passage("A", 2, "court aussi")

    paires = [
        ("S1", extract._source_depuis_passage(petit)),
        ("S2", extract._source_depuis_passage(enorme)),
        ("S3", extract._source_depuis_passage(petit2)),
    ]

    lots = extract._partitionner(paires, limite_caracteres=1_000)

    toutes_citations = [c for lot in lots for c, _ in lot]
    assert sorted(toutes_citations) == ["S1", "S2", "S3"]
    assert any(("S2", paires[1][1]) in lot for lot in lots)


# ===========================================================================
# O — provenance finale vers les passages originaux
# ===========================================================================


def test_provenance_pointe_vers_les_vrais_passages(monkeypatch):
    passages = [_passage("A", i, f"contenu {i}", page=i + 1) for i in range(3)]
    _resoudre_vers(monkeypatch, "A", passages)
    monkeypatch.setattr(extract, "LIMITE_CARACTERES_LOT", 60)  # 3 lots

    llm = LLMExtractions(
        [
            _rep({"montant": [("100 EUR", ["S1"])]}),
            _rep({"montant": []}),
            _rep({"montant": []}),
        ]
    )
    resultat = _executer_a(_contexte_a(llm), champs=["montant"])

    assert resultat.succes
    assert len(resultat.sources) == 1
    source = resultat.sources[0]
    assert source.doc_id == "A"
    assert source.page == 1
    assert source.extrait == passages[0].texte


# ===========================================================================
# P / V — aucun champ métier hardcodé (généricité)
# ===========================================================================


def test_generique_champs_fictifs_sans_rapport_avec_le_corpus(monkeypatch):
    passages = [_passage("VAISSEAU", i, f"contenu {i}") for i in range(2)]
    _resoudre_vers(monkeypatch, "VAISSEAU", passages)

    llm = LLMExtractions(
        [
            _rep(
                {
                    "couleur_du_dragon": [("rouge", ["S1"])],
                    "temperature_du_reacteur": [("310K", ["S1"])],
                    "nom_du_capitaine": [],
                }
            )
        ]
    )
    resultat = _executer_a(
        _contexte_a(llm),
        champs=["couleur_du_dragon", "temperature_du_reacteur", "nom_du_capitaine"],
        documents=("VAISSEAU",),
    )

    assert resultat.succes
    assert resultat.donnees["extractions"]["couleur_du_dragon"]["valeur"] == "rouge"
    assert resultat.donnees["extractions"]["temperature_du_reacteur"]["valeur"] == "310K"
    assert resultat.donnees["extractions"]["nom_du_capitaine"]["trouve"] is False


# ===========================================================================
# Q — panne d'un lot sans crash global si d'autres extractions restent fiables
# ===========================================================================


def test_panne_dun_lot_nempeche_pas_les_autres(monkeypatch):
    passages = [_passage("A", i, f"contenu {i}") for i in range(3)]
    _resoudre_vers(monkeypatch, "A", passages)
    monkeypatch.setattr(extract, "LIMITE_CARACTERES_LOT", 60)  # 3 lots

    llm = _LLMExplosePartiellement(
        indices_erreur={2},
        reponses=[
            _rep({"montant": [("100 EUR", ["S1"])]}),
            _rep({}),  # jamais utilisé (lot 2 explose)
            _rep({"montant": [("200 EUR", ["S3"])]}),
        ],
    )
    resultat = _executer_a(_contexte_a(llm), champs=["montant"])

    assert resultat.succes
    valeurs = {v["valeur"] for v in resultat.donnees["extractions"]["montant"]["valeurs"]}
    assert valeurs == {"100 EUR", "200 EUR"}
    assert resultat.donnees["lots_invalides"] == 1
    assert resultat.donnees["lots_valides"] == 2
    assert any("erreur technique" in a for a in resultat.avertissements)


# ===========================================================================
# R — aucune hallucination lorsqu'aucune valeur n'est présente
# ===========================================================================


def test_aucune_hallucination_quand_rien_nest_present():
    llm = LLMExtractions([_rep({"date_expiration": []})])
    contexte = _contexte_b(llm, [_source("A", "Aucune date ici.")])

    resultat = _executer_b(contexte, champs=["date_expiration"])

    assert resultat.succes
    assert resultat.donnees["extractions"]["date_expiration"] == {
        "trouve": False,
        "valeur_unique": False,
        "valeur": None,
        "valeurs": [],
    }


# ===========================================================================
# S / U — déduplication + conservation des citations
# ===========================================================================


def test_deduplication_meme_valeur_plusieurs_lots_fusionne_citations(monkeypatch):
    passages = [_passage("A", i, f"contenu {i}") for i in range(2)]
    _resoudre_vers(monkeypatch, "A", passages)
    monkeypatch.setattr(extract, "LIMITE_CARACTERES_LOT", 60)  # 2 lots

    llm = LLMExtractions(
        [
            _rep({"entreprise": [("ABC", ["S1"])]}),
            _rep({"entreprise": [("abc", ["S2"])]}),  # même valeur, casse différente
        ]
    )
    resultat = _executer_a(_contexte_a(llm), champs=["entreprise"])

    assert resultat.succes
    extraction = resultat.donnees["extractions"]["entreprise"]
    assert extraction["valeur_unique"] is True
    assert extraction["valeur"] == "ABC"  # première formulation rencontrée conservée
    assert extraction["valeurs"][0]["citations"] == ["S1", "S2"]  # provenance fusionnée, pas perdue


# ===========================================================================
# T — conflit entre deux valeurs différentes (variante Cas A déjà couverte
# par test E ci-dessus ; ici une variante intra-lot)
# ===========================================================================


def test_conflit_intra_lot_deux_valeurs_conservees():
    llm = LLMExtractions(
        [_rep({"montant": [("100 EUR", ["S1"]), ("120 EUR", ["S1"])]})]
    )
    contexte = _contexte_b(llm, [_source("A", "Montant HT 100 EUR, montant TTC 120 EUR")])

    resultat = _executer_b(contexte, champs=["montant"])

    assert resultat.succes
    valeurs = {v["valeur"] for v in resultat.donnees["extractions"]["montant"]["valeurs"]}
    assert valeurs == {"100 EUR", "120 EUR"}


# ===========================================================================
# Aucun search / embedding / reranker dans le chemin document complet
# ===========================================================================


def test_aucun_appel_rechercher_passages_embeddings_reranker(monkeypatch):
    passages = [_passage("A", i, f"contenu {i}") for i in range(3)]
    _resoudre_vers(monkeypatch, "A", passages)

    def _explose(*args, **kwargs):
        raise AssertionError("Ne doit jamais être appelé en mode document complet.")

    import src.rag.embeddings as embeddings_mod
    import src.rag.retrieval as retrieval_mod

    monkeypatch.setattr(retrieval_mod, "rechercher_passages", _explose)
    monkeypatch.setattr(embeddings_mod, "encoder_requete", _explose)
    monkeypatch.setattr(embeddings_mod, "reranker", _explose)

    llm = LLMExtractions([_rep({"montant": [("100 EUR", ["S1"])]})])
    resultat = _executer_a(_contexte_a(llm), champs=["montant"])

    assert resultat.succes
    assert resultat.donnees["extractions"]["montant"]["valeur"] == "100 EUR"


# ===========================================================================
# Cloisonnement Cas B : document précisé filtre correctement les sources
# ===========================================================================


def test_cas_b_document_precise_filtre_les_sources():
    llm = LLMExtractions([_rep({"montant": [("100 EUR", ["S1"])]})])
    contexte = _contexte_b(llm, [_source("A", "Montant A : 100 EUR"), _source("B", "Montant B : 200 EUR")])

    resultat = _executer_b(contexte, champs=["montant"], document="A")

    assert resultat.succes
    assert resultat.donnees["document"] == "A.pdf"
    assert "S1" in llm.appels[0]
    assert "200 EUR" not in llm.appels[0]


# ===========================================================================
# Aucun champ valide fourni -> échec propre
# ===========================================================================


def test_aucun_champ_valide_echec_propre():
    contexte = _contexte_b(LLMExtractions([]), [_source("A")])

    resultat = _executer_b(contexte, champs=["   ", ""])

    assert not resultat.succes
    assert "champ" in resultat.message.lower()
