"""
Tests du pipeline COMPARE / SYNTHESIZE (P1.5) — couverture INTÉGRALE du
document par lots bornés (`src.agent.multidoc_pipeline`).

Objet : garantir qu'un document ciblé est traité en entier — jamais tronqué
aux premiers N caractères — via partitionnement borné, MAP par lot, puis
agrégation intra-document. Provenance (citations, pages) conservée à travers
toutes les étapes. Aucun Ollama / Qdrant.
"""

from __future__ import annotations

import re

import pytest

from src.agent import multidoc_pipeline as mp
from src.agent.multidoc_pipeline import DocumentCible, map_document
from tests.agent._multidoc_fakes import (
    EST_AGREGATION_INTRA_DOC,
    EST_MAP_LOT,
    SANS_EVIDENCE,
    LLMScripte,
    cabler_corpus,
    passage,
)

_CIT = re.compile(r"\[(D\d+S\d+)\]")


def _cible(index=1, doc_id="A", libelle="rapport_a.pdf"):
    return DocumentCible(index=index, doc_id=doc_id, libelle=libelle, nom_fichier=libelle)


# --------------------------------------------------------------------------
# Partitionnement — jamais de perte, jamais de troncature
# --------------------------------------------------------------------------


def test_partitionner_ne_perd_aucun_passage(monkeypatch) -> None:
    monkeypatch.setattr(mp, "LIMITE_CARACTERES_LOT", 120)
    passages = [passage("A", i, "x" * 80, page=i) for i in range(1, 9)]  # 8 passages
    lots, table = mp._partitionner_passages(_cible(), passages)
    # tous les passages répartis, aucun abandonné
    citations_lots = [c for lot in lots for c, _ in lot]
    assert citations_lots == [f"D1S{i}" for i in range(1, 9)]
    assert len(lots) > 1  # a bien été découpé
    assert set(table) == {f"D1S{i}" for i in range(1, 9)}
    # les pages sont conservées pour chaque passage
    assert [table[f"D1S{i}"].page for i in range(1, 9)] == list(range(1, 9))


def test_passage_plus_gros_que_la_limite_forme_son_propre_lot(monkeypatch) -> None:
    monkeypatch.setattr(mp, "LIMITE_CARACTERES_LOT", 100)
    passages = [passage("A", 1, "y" * 500, page=1), passage("A", 2, "z" * 10, page=2)]
    lots, _ = mp._partitionner_passages(_cible(), passages)
    assert [c for lot in lots for c, _ in lot] == ["D1S1", "D1S2"]  # rien perdu


# --------------------------------------------------------------------------
# map_document — couverture intégrale multi-lots
# --------------------------------------------------------------------------


def _cabler(monkeypatch, passages) -> None:
    cabler_corpus(
        monkeypatch,
        mp,
        fiches={"rapport_a.pdf": "A"},
        passages_par_doc={"A": passages},
    )


def test_map_document_couvre_tous_les_lots(monkeypatch) -> None:
    monkeypatch.setattr(mp, "LIMITE_CARACTERES_LOT", 150)
    passages = [passage("A", i, f"contenu numero {i} " * 4, page=i) for i in range(1, 11)]
    _cabler(monkeypatch, passages)

    llm = LLMScripte()
    m = map_document(_cible(), "Quelle est la valeur ?", llm=llm)

    assert m.utilisable
    assert m.nombre_lots > 1
    # chaque passage a été montré au LLM dans au moins un appel MAP de lot
    citations_vues: set[str] = set()
    for systeme, utilisateur in llm.appels:
        if EST_MAP_LOT(systeme):
            citations_vues |= set(_CIT.findall(utilisateur))
    assert citations_vues == {f"D1S{i}" for i in range(1, 11)}
    # l'agrégation intra-document a bien eu lieu (plusieurs lots)
    assert any(EST_AGREGATION_INTRA_DOC(s) for s, _ in llm.appels)


def test_information_au_dela_de_l_ancien_seuil_12000_contribue(monkeypatch) -> None:
    # Un document dont le contenu dépasse largement 12 000 caractères :
    # l'info du dernier passage doit rester analysée (pas de troncature).
    monkeypatch.setattr(mp, "LIMITE_CARACTERES_LOT", 4_000)
    gros = [passage("A", i, "bloc de remplissage. " * 250, page=i) for i in range(1, 6)]
    marqueur = passage("A", 6, "La valeur cible est 42.", page=6)  # ~15 000 car. avant lui
    _cabler(monkeypatch, gros + [marqueur])

    llm = LLMScripte()
    m = map_document(_cible(), "Quelle est la valeur cible ?", llm=llm)

    assert m.utilisable
    assert "D1S6" in m.citations_valides  # le passage final a été analysé et cité
    assert m.sources_map["D1S6"].page == 6  # provenance/page conservées


def test_map_document_provenance_multi_lots(monkeypatch) -> None:
    monkeypatch.setattr(mp, "LIMITE_CARACTERES_LOT", 130)
    passages = [passage("A", i, f"fait {i} pertinent " * 3, page=10 + i) for i in range(1, 7)]
    _cabler(monkeypatch, passages)

    m = map_document(_cible(), "?", llm=LLMScripte())
    assert m.utilisable
    for c in m.citations_valides:
        idx = int(c.replace("D1S", ""))
        assert m.sources_map[c].page == 10 + idx  # page correcte pour chaque lot


# --------------------------------------------------------------------------
# Bornes & abstention
# --------------------------------------------------------------------------


def test_au_dela_de_nb_lots_max_le_document_est_refuse_pas_tronque(monkeypatch) -> None:
    monkeypatch.setattr(mp, "LIMITE_CARACTERES_LOT", 60)
    monkeypatch.setattr(mp, "NB_LOTS_MAX", 3)
    passages = [passage("A", i, "z" * 55, page=i) for i in range(1, 10)]  # -> 9 lots
    _cabler(monkeypatch, passages)

    m = map_document(_cible(), "?", llm=LLMScripte())
    assert not m.utilisable
    assert m.echec is not None and "trop volumineux" in m.echec
    assert m.nombre_lots == 9  # compté, pas tronqué


def test_un_lot_en_echec_ne_casse_pas_le_document(monkeypatch) -> None:
    monkeypatch.setattr(mp, "LIMITE_CARACTERES_LOT", 120)
    passages = [passage("A", i, f"contenu {i} " * 6, page=i) for i in range(1, 7)]
    _cabler(monkeypatch, passages)

    class _LLM:
        def __init__(self):
            self.n = 0

        def invoke(self, messages):
            s = messages[0].content
            if EST_MAP_LOT(s):
                self.n += 1
                if self.n == 2:  # le 2e lot échoue
                    raise RuntimeError("lot KO")
            return LLMScripte().invoke(messages)

    m = map_document(_cible(), "?", llm=_LLM())
    assert m.utilisable  # les autres lots suffisent
    assert m.lots_en_echec == 1
    assert any("non analysé" in a for a in m.avertissements)


def test_tous_les_lots_en_echec_document_inexploitable(monkeypatch) -> None:
    monkeypatch.setattr(mp, "LIMITE_CARACTERES_LOT", 120)
    passages = [passage("A", i, f"c {i} " * 6, page=i) for i in range(1, 5)]
    _cabler(monkeypatch, passages)

    class _LLM:
        def invoke(self, messages):
            if EST_MAP_LOT(messages[0].content):
                raise RuntimeError("KO")
            return LLMScripte().invoke(messages)

    m = map_document(_cible(), "?", llm=_LLM())
    assert not m.utilisable
    assert m.echec is not None


def test_document_sans_aucun_lot_pertinent_est_sans_evidence(monkeypatch) -> None:
    monkeypatch.setattr(mp, "LIMITE_CARACTERES_LOT", 200)
    passages = [passage("A", i, SANS_EVIDENCE, page=i) for i in range(1, 4)]
    _cabler(monkeypatch, passages)

    m = map_document(_cible(), "?", llm=LLMScripte())
    assert m.sans_evidence
    assert not m.utilisable


# --------------------------------------------------------------------------
# Agrégation intra-document
# --------------------------------------------------------------------------


def test_agreger_un_seul_texte_ne_fait_aucun_appel(monkeypatch) -> None:
    appels: list = []

    class _LLM:
        def invoke(self, messages):  # pragma: no cover
            appels.append(1)
            raise AssertionError("aucun appel attendu")

    out = mp._agreger_intra_document(
        _cible(), ["texte unique [D1S1]"], llm=_LLM(), profil_domaine=None, budget=10_000
    )
    assert out == "texte unique [D1S1]"
    assert appels == []


def test_agregation_hierarchique_bornee(monkeypatch) -> None:
    monkeypatch.setattr(mp, "LIMITE_CARACTERES_LOT", 80)
    monkeypatch.setattr(mp, "PROFONDEUR_MAX_AGREGATION", 2)
    textes = [f"analyse partielle {i} [D1S{i}]" for i in range(1, 12)]
    llm = LLMScripte()
    out = mp._agreger_intra_document(
        _cible(), textes, llm=llm, profil_domaine=None, budget=10_000
    )
    # toutes les citations d'origine sont conservées dans l'agrégat final
    assert set(_CIT.findall(out)) == {f"D1S{i}" for i in range(1, 12)}
    assert all(EST_AGREGATION_INTRA_DOC(s) for s, _ in llm.appels)


# --------------------------------------------------------------------------
# §2.4 / cas (e) — agrégation à profondeur max encore hors budget => échec
#                  explicite, JAMAIS de troncature
# --------------------------------------------------------------------------


def test_agregation_hors_budget_a_profondeur_max_leve_erreur_bornee(monkeypatch) -> None:
    monkeypatch.setattr(mp, "PROFONDEUR_MAX_AGREGATION", 1)
    # Chaque texte fait ~500 c ; 6 textes = ~3000 c > budget 400.
    textes = [f"analyse {'y'*480} [D1S{i}]" for i in range(1, 7)]
    with pytest.raises(mp.BudgetLLMDepasse) as exc:
        mp._agreger_intra_document(
            _cible(), textes, llm=LLMScripte(), profil_domaine=None, budget=400
        )
    assert "profondeur" in str(exc.value)


def test_map_document_agregation_hors_budget_produit_echec_explicite(monkeypatch) -> None:
    monkeypatch.setattr(mp, "LIMITE_CARACTERES_LOT", 200)
    monkeypatch.setattr(mp, "PROFONDEUR_MAX_AGREGATION", 1)
    # budget : assez grand pour la cohérence 2.1 (lot plein + système) et pour
    # que chaque MAP de lot passe, mais trop petit pour agréger tous les
    # résultats de lots -> BudgetLLMDepasse à profondeur max.
    monkeypatch.setattr(mp, "budget_caracteres_entree_llm", lambda: 1_600)

    # LLM dont la sortie MAP est volontairement volumineuse (ignore « bref »).
    class _LLMMapVerbeux:
        def invoke(self, messages):
            from langchain_core.messages import AIMessage

            s, u = messages[0].content, messages[1].content
            if EST_MAP_LOT(s):
                cites = " ".join(f"[{c}]" for c in _CIT.findall(u))
                return AIMessage(content="détail " * 120 + cites)
            return AIMessage(content="agg " + " ".join(f"[{c}]" for c in _CIT.findall(u)))

    passages = [passage("Z", i, f"contenu {i} " * 8, page=i) for i in range(1, 12)]
    cabler_corpus(monkeypatch, mp, fiches={"z.pdf": "Z"}, passages_par_doc={"Z": passages})

    m = mp.map_document(
        mp.DocumentCible(index=1, doc_id="Z", libelle="z.pdf", nom_fichier="z.pdf"),
        "?",
        llm=_LLMMapVerbeux(),
    )
    assert m.echec is not None
    assert "agrégation intra-document impossible dans le budget" in m.echec
    assert "profondeur max atteinte" in m.echec
    assert not m.utilisable
    assert m.citations_valides == []  # aucun résultat partiel présenté


# --------------------------------------------------------------------------
# cas (g) — toute citation finale appartient réellement à un passage traité
# --------------------------------------------------------------------------


def test_citations_du_map_appartiennent_toutes_a_la_table(monkeypatch) -> None:
    monkeypatch.setattr(mp, "LIMITE_CARACTERES_LOT", 140)
    passages = [passage("A", i, f"fait {i} pertinent " * 3, page=i) for i in range(1, 9)]
    _cabler(monkeypatch, passages)

    m = mp.map_document(_cible(), "?", llm=LLMScripte())
    assert m.utilisable
    for c in m.citations_valides:
        # forme D<k>S<j>, préfixe du document, et présent dans la table complète
        assert re.fullmatch(r"D1S\d+", c)
        assert c in m.sources_map
        assert m.sources_map[c].doc_id == "A"
