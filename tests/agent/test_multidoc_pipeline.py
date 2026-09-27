"""
Tests du pipeline COMPARE / SYNTHESIZE (P1.5 -> P1.8 PLAN borné + MAP
structuré) — couverture INTÉGRALE du document par lots bornés
(`src.agent.multidoc_pipeline`).

Objet : garantir qu'un document ciblé est traité en entier — jamais tronqué
aux premiers N caractères — via partitionnement borné, MAP structuré par lot,
puis fusion déterministe (0 appel LLM, cf. `_fusionner_elements`). Provenance
(citations, pages) conservée à travers toutes les étapes. Aucun Ollama /
Qdrant.
"""

from __future__ import annotations

import re

from src.agent import multidoc_pipeline as mp
from src.agent.multidoc_pipeline import DocumentCible, ElementMap, MapResult, map_document
from tests.agent._multidoc_fakes import (
    EST_MAP_LOT,
    EST_PLAN,
    SANS_EVIDENCE,
    LLMScripte,
    cabler_corpus,
    passage,
)

_CIT = re.compile(r"\[(D\d+S\d+)\]")


def _cible(index=1, doc_id="A", libelle="rapport_a.pdf"):
    return DocumentCible(index=index, doc_id=doc_id, libelle=libelle, nom_fichier=libelle)


def _task_spec(*, axes=("valeur",), operation="compare"):
    return mp.TaskSpec(operation=operation, objectif="Analyser.", axes=tuple(axes))


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
    m = map_document(_cible(), _task_spec(), llm=llm)

    assert m.utilisable
    assert m.nombre_lots > 1
    # chaque passage a été montré au LLM dans au moins un appel MAP de lot
    citations_vues: set[str] = set()
    for systeme, utilisateur in llm.appels:
        if EST_MAP_LOT(systeme):
            citations_vues |= set(_CIT.findall(utilisateur))
    assert citations_vues == {f"D1S{i}" for i in range(1, 11)}
    # aucune agrégation LLM : la fusion multi-lots est déterministe. Ici
    # `map_document` est appelé directement (task_spec déjà construit) donc
    # aucun appel PLAN n'a lieu : exactement 1 appel MAP par lot, rien de plus.
    appels_map = [s for s, _ in llm.appels if EST_MAP_LOT(s)]
    assert len(appels_map) == m.nombre_lots
    assert len(llm.appels) == m.nombre_lots


def test_information_au_dela_de_l_ancien_seuil_12000_contribue(monkeypatch) -> None:
    # Un document dont le contenu dépasse largement 12 000 caractères :
    # l'info du dernier passage doit rester analysée (pas de troncature).
    monkeypatch.setattr(mp, "LIMITE_CARACTERES_LOT", 4_000)
    gros = [passage("A", i, "bloc de remplissage. " * 250, page=i) for i in range(1, 6)]
    marqueur = passage("A", 6, "La valeur cible est 42.", page=6)  # ~15 000 car. avant lui
    _cabler(monkeypatch, gros + [marqueur])

    llm = LLMScripte()
    m = map_document(_cible(), _task_spec(), llm=llm)

    assert m.utilisable
    assert "D1S6" in m.citations_valides  # le passage final a été analysé et cité
    assert m.sources_map["D1S6"].page == 6  # provenance/page conservées


def test_map_document_provenance_multi_lots(monkeypatch) -> None:
    monkeypatch.setattr(mp, "LIMITE_CARACTERES_LOT", 130)
    passages = [passage("A", i, f"fait {i} pertinent " * 3, page=10 + i) for i in range(1, 7)]
    _cabler(monkeypatch, passages)

    m = map_document(_cible(), _task_spec(), llm=LLMScripte())
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

    m = map_document(_cible(), _task_spec(), llm=LLMScripte())
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

        def invoke(self, messages, think: bool | None = None):
            s = messages[0].content
            if EST_MAP_LOT(s):
                self.n += 1
                if self.n == 2:  # le 2e lot échoue
                    raise RuntimeError("lot KO")
            return LLMScripte().invoke(messages, think=think)

    m = map_document(_cible(), _task_spec(), llm=_LLM())
    assert m.utilisable  # les autres lots suffisent
    assert m.lots_en_echec == 1
    assert any("non analysé" in a for a in m.avertissements)


def test_tous_les_lots_en_echec_document_inexploitable(monkeypatch) -> None:
    monkeypatch.setattr(mp, "LIMITE_CARACTERES_LOT", 120)
    passages = [passage("A", i, f"c {i} " * 6, page=i) for i in range(1, 5)]
    _cabler(monkeypatch, passages)

    class _LLM:
        def invoke(self, messages, think: bool | None = None):
            if EST_MAP_LOT(messages[0].content):
                raise RuntimeError("KO")
            return LLMScripte().invoke(messages, think=think)

    m = map_document(_cible(), _task_spec(), llm=_LLM())
    assert not m.utilisable
    assert m.echec is not None


def test_document_sans_aucun_lot_pertinent_est_sans_evidence(monkeypatch) -> None:
    monkeypatch.setattr(mp, "LIMITE_CARACTERES_LOT", 200)
    passages = [passage("A", i, SANS_EVIDENCE, page=i) for i in range(1, 4)]
    _cabler(monkeypatch, passages)

    m = map_document(_cible(), _task_spec(), llm=LLMScripte())
    assert m.sans_evidence
    assert not m.utilisable


# --------------------------------------------------------------------------
# Fusion multi-lots — DÉTERMINISTE, 0 appel LLM (remplace l'agrégation
# intra-document LLM de P1.5, cf. historique en tête de module)
# --------------------------------------------------------------------------


def test_fusionner_elements_deduplique_sur_axe_et_contenu() -> None:
    a = ElementMap(axe="axe1", contenu="fait X", citations=("D1S1",))
    a_bis = ElementMap(axe="axe1", contenu="fait X", citations=("D1S1",))  # doublon exact
    b = ElementMap(axe="axe1", contenu="fait Y", citations=("D1S2",))
    resultats = [
        MapResult(pertinent=True, elements=(a,)),
        MapResult(pertinent=True, elements=(a_bis, b)),
    ]
    fusion = mp._fusionner_elements(resultats)
    assert fusion == (a, b)  # doublon écarté, ordre d'apparition conservé


def test_fusionner_elements_conserve_toutes_les_citations_de_tous_les_lots() -> None:
    resultats = [
        MapResult(
            pertinent=True,
            elements=(ElementMap(axe="axe", contenu=f"lot {i}", citations=(f"D1S{i}",)),),
        )
        for i in range(1, 6)
    ]
    fusion = mp._fusionner_elements(resultats)
    toutes_citations = {c for el in fusion for c in el.citations}
    assert toutes_citations == {f"D1S{i}" for i in range(1, 6)}


def test_map_document_budget_insuffisant_refuse_explicitement(monkeypatch) -> None:
    # Budget d'entrée trop petit pour qu'un lot plein + le prompt système/
    # gabarit MAP y tienne jamais -> refus explicite du document, AUCUN
    # appel LLM (jamais de troncature silencieuse).
    monkeypatch.setattr(mp, "budget_caracteres_entree_llm", lambda: 500)
    passages = [passage("A", 1, "contenu court", page=1)]
    _cabler(monkeypatch, passages)

    class _LLM:
        def invoke(self, messages, think=None):  # pragma: no cover
            raise AssertionError("aucun appel LLM attendu")

    m = map_document(_cible(), _task_spec(), llm=_LLM())
    assert m.echec is not None
    assert "budget LLM insuffisant" in m.echec
    assert not m.utilisable


# --------------------------------------------------------------------------
# cas (g) — toute citation finale appartient réellement à un passage traité
# --------------------------------------------------------------------------


def test_citations_du_map_appartiennent_toutes_a_la_table(monkeypatch) -> None:
    monkeypatch.setattr(mp, "LIMITE_CARACTERES_LOT", 140)
    passages = [passage("A", i, f"fait {i} pertinent " * 3, page=i) for i in range(1, 9)]
    _cabler(monkeypatch, passages)

    m = mp.map_document(_cible(), _task_spec(), llm=LLMScripte())
    assert m.utilisable
    for c in m.citations_valides:
        # forme D<k>S<j>, préfixe du document, et présent dans la table complète
        assert re.fullmatch(r"D1S\d+", c)
        assert c in m.sources_map
        assert m.sources_map[c].doc_id == "A"


# --------------------------------------------------------------------------
# PLAN — borné, jamais la question brute en axe, repli déterministe systématique
# --------------------------------------------------------------------------

_CIBLES_PLAN = [DocumentCible(index=1, doc_id="A", libelle="rapport_a.pdf", nom_fichier="rapport_a.pdf")]


def test_planifier_llm_absent_repli_deterministe() -> None:
    ts = mp.planifier("compare", "Compare A et B.", _CIBLES_PLAN, llm=None)
    assert ts.axes  # jamais vide
    assert ts.axes != ("Compare A et B.",)  # jamais la question brute en axe


def test_planifier_json_valide_reprend_les_axes_du_llm() -> None:
    class _LLM:
        def invoke(self, messages, think=None):
            assert think is False  # PLAN désactive le raisonnement
            import json

            from langchain_core.messages import AIMessage

            return AIMessage(
                content=json.dumps(
                    {
                        "objectif": "Comparer les montants.",
                        "axes": ["montants", "dates"],
                        "informations_attendues": ["chiffres"],
                    }
                )
            )

    ts = mp.planifier("compare", "Compare A et B.", _CIBLES_PLAN, llm=_LLM())
    assert ts.axes == ("montants", "dates")
    assert ts.objectif == "Comparer les montants."
    assert ts.informations_attendues == ("chiffres",)


def test_planifier_json_invalide_repli_deterministe() -> None:
    class _LLM:
        def invoke(self, messages, think=None):
            from langchain_core.messages import AIMessage

            return AIMessage(content="pas du JSON")

    ts = mp.planifier("compare", "Compare A et B.", _CIBLES_PLAN, llm=_LLM())
    assert ts.axes == mp._plan_repli("compare").axes


def test_planifier_axes_vides_repli_deterministe() -> None:
    class _LLM:
        def invoke(self, messages, think=None):
            import json

            from langchain_core.messages import AIMessage

            return AIMessage(content=json.dumps({"objectif": "x", "axes": []}))

    ts = mp.planifier("compare", "Compare A et B.", _CIBLES_PLAN, llm=_LLM())
    assert ts.axes == mp._plan_repli("compare").axes


def test_planifier_llm_echoue_repli_deterministe() -> None:
    class _LLM:
        def invoke(self, messages, think=None):
            raise RuntimeError("PLAN KO")

    ts = mp.planifier("synthesize", "Synthèse.", _CIBLES_PLAN, llm=_LLM())
    assert ts.axes == mp._plan_repli("synthesize").axes


def test_planifier_hors_budget_repli_deterministe(monkeypatch) -> None:
    monkeypatch.setattr(mp, "budget_caracteres_entree_llm", lambda: 10)

    class _LLM:
        def invoke(self, messages, think=None):  # pragma: no cover
            raise AssertionError("aucun appel attendu si hors budget")

    ts = mp.planifier("compare", "Compare A et B.", _CIBLES_PLAN, llm=_LLM())
    assert ts.axes == mp._plan_repli("compare").axes


def test_planifier_axes_bornes_a_nb_axes_max() -> None:
    class _LLM:
        def invoke(self, messages, think=None):
            import json

            from langchain_core.messages import AIMessage

            axes = [f"axe{i}" for i in range(1, 20)]  # bien au-delà de NB_AXES_MAX
            return AIMessage(content=json.dumps({"objectif": "x", "axes": axes}))

    ts = mp.planifier("compare", "?", _CIBLES_PLAN, llm=_LLM())
    assert len(ts.axes) <= mp.NB_AXES_MAX


# --------------------------------------------------------------------------
# Validation déterministe de la sortie MAP (`_valider_map_result`)
# --------------------------------------------------------------------------


def test_valider_map_result_element_valide() -> None:
    objet = {
        "pertinent": True,
        "elements": [{"axe": "montants", "contenu": "3 %", "citations": ["D1S1"]}],
    }
    r = mp._valider_map_result(objet, axes={"montants"}, citations_du_lot={"D1S1"})
    assert r.pertinent
    assert len(r.elements) == 1
    assert r.elements[0].citations == ("D1S1",)


def test_valider_map_result_axe_hors_perimetre_ecarte() -> None:
    objet = {
        "pertinent": True,
        "elements": [{"axe": "axe_invente", "contenu": "x", "citations": ["D1S1"]}],
    }
    r = mp._valider_map_result(objet, axes={"montants"}, citations_du_lot={"D1S1"})
    assert r.elements == ()
    assert r.warnings


def test_valider_map_result_citation_hors_lot_ecartee() -> None:
    objet = {
        "pertinent": True,
        "elements": [{"axe": "montants", "contenu": "x", "citations": ["D9S9"]}],
    }
    r = mp._valider_map_result(objet, axes={"montants"}, citations_du_lot={"D1S1"})
    assert r.elements == ()  # aucune citation valide -> élément écarté en entier


def test_valider_map_result_json_pas_un_objet() -> None:
    r = mp._valider_map_result("pas un dict", axes={"montants"}, citations_du_lot={"D1S1"})
    assert not r.pertinent
    assert r.elements == ()


# --------------------------------------------------------------------------
# P1.9 — portée intra-document et résolveur catalogue
# --------------------------------------------------------------------------

from src.rag.retrieval import FicheDocument, PerimetreDocumentaire  # noqa: E402
from tests.agent._multidoc_fakes import FauxCatalogue  # noqa: E402


def _cabler_catalogue(monkeypatch, catalogue_factice) -> None:
    monkeypatch.setattr(mp, "catalogue", lambda profil=None, corpus_id=None: catalogue_factice)


def test_resoudre_cibles_intra_accepte_un_seul_document(monkeypatch) -> None:
    _cabler_catalogue(monkeypatch, FauxCatalogue({"a.pdf": "A", "b.pdf": "B"}))
    r = mp.resoudre_cibles(["a.pdf"], portee="intra")
    assert r.refus is None and r.exploitable
    assert [c.doc_id for c in r.documents] == ["A"]
    assert r.portee == "intra"


def test_resoudre_cibles_intra_sans_document_refuse(monkeypatch) -> None:
    _cabler_catalogue(monkeypatch, FauxCatalogue({"a.pdf": "A"}))
    r = mp.resoudre_cibles([], portee="intra")
    assert r.motif == "references_insuffisantes"
    assert not r.exploitable


def test_resoudre_cibles_intra_refuse_plusieurs_documents(monkeypatch) -> None:
    _cabler_catalogue(monkeypatch, FauxCatalogue({"a.pdf": "A", "b.pdf": "B"}))
    r = mp.resoudre_cibles(["a.pdf", "b.pdf"], portee="intra")
    assert r.motif == "intra_plusieurs_documents"


def test_resoudre_cibles_inter_exige_toujours_deux_documents(monkeypatch) -> None:
    _cabler_catalogue(monkeypatch, FauxCatalogue({"a.pdf": "A"}))
    r = mp.resoudre_cibles(["a.pdf"])
    assert r.motif == "references_insuffisantes"


class _CatalogueResolvant(FauxCatalogue):
    """Résout chaque fragment contenant un mot-clé vers un document."""

    def __init__(self, fiches_par_nom: dict[str, str], motcles: dict[str, str]) -> None:
        super().__init__(fiches_par_nom)
        self._motcles = motcles  # {mot-clé: nom de fichier}
        self.fragments: list[str] = []

    def resoudre(self, fragment: str) -> PerimetreDocumentaire:
        self.fragments.append(fragment)
        trouves = [nom for mot, nom in self._motcles.items() if mot in fragment.lower()]
        if len(trouves) != 1:
            return PerimetreDocumentaire(statut="ambigu" if trouves else "aucun")
        return PerimetreDocumentaire(
            statut="exact", champ_filtre="document_id", valeurs_filtre=(self._m[trouves[0]],)
        )

    def par_identifiant(self, identifiant: str) -> FicheDocument | None:
        for nom, doc_id in self._m.items():
            if identifiant in (nom, doc_id):
                return FicheDocument(document_id=doc_id, champ_id="document_id", nom_fichier=nom)
        return None


def test_resolveur_catalogue_resout_fragment_par_fragment(monkeypatch) -> None:
    cat = _CatalogueResolvant(
        {"barometre.pdf": "A", "etude.pdf": "B"},
        {"barometre": "barometre.pdf", "etude": "etude.pdf"},
    )
    _cabler_catalogue(monkeypatch, cat)

    # La requête entière est ambiguë ; chaque fragment désigne un document.
    trouves = mp.resolveur_catalogue("default")("Compare le barometre et l'etude")

    assert trouves == ("barometre.pdf", "etude.pdf")
    assert "Compare le barometre et l'etude" in cat.fragments


def test_resolveur_catalogue_ignore_les_perimetres_non_fiables(monkeypatch) -> None:
    cat = _CatalogueResolvant({"a.pdf": "A"}, {"alpha": "a.pdf"})
    _cabler_catalogue(monkeypatch, cat)
    assert mp.resolveur_catalogue("default")("Compare les rapports trimestriels") == ()


def test_resolveur_catalogue_resultat_reconnu_par_resoudre_cibles(monkeypatch) -> None:
    cat = _CatalogueResolvant({"a.pdf": "A"}, {"alpha": "a.pdf"})
    _cabler_catalogue(monkeypatch, cat)
    refs = mp.resolveur_catalogue("default")("Compare les sections du rapport alpha")
    r = mp.resoudre_cibles(refs, portee="intra")
    assert [c.doc_id for c in r.documents] == ["A"]


def test_plan_intra_utilise_le_prompt_et_le_repli_intra() -> None:
    cible = DocumentCible(index=1, doc_id="A", libelle="Rapport A", nom_fichier="a.pdf")

    repli = mp.planifier("synthesize", "Synthétise les chapitres.", [cible], llm=None, portee="intra")
    assert repli.axes == mp._AXES_REPLI_INTRA["synthesize"]

    llm = LLMScripte()
    mp.planifier("compare", "Compare 2023 et 2024.", [cible], llm=llm, portee="intra")
    assert "PARTIES D'UN SEUL document" in llm.appels[0][0]


def test_plan_inter_inchange() -> None:
    cible = DocumentCible(index=1, doc_id="A", libelle="Rapport A", nom_fichier="a.pdf")
    llm = LLMScripte()
    mp.planifier("compare", "Compare A et B.", [cible], llm=llm)
    assert llm.appels[0][0] == mp._SYSTEME_PLAN
