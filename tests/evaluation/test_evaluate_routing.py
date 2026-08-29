"""
Tests du banc de routage (`evaluation/evaluate_routing.py`) — étape P1.1.

Portée : validité du jeu de cas, respect de la taxonomie, couverture
(intentions, langues, domaines), déterminisme du runner, et **absence totale
d'appel LLM / réseau / Qdrant**.

Aucun de ces tests n'ouvre de connexion : le banc n'appelle que
`nodes._detecter_intention` (fonction pure) puis applique le repli
déterministe documenté des deux zones grises.
"""

from __future__ import annotations

import re

import pytest

from evaluation import evaluate_routing as er
from src.agent import nodes

# Noms de documents fictifs autorisés dans les requêtes : aucun document
# métier réel, aucun artefact CQuAE.
_DOCS_FICTIFS_AUTORISES: frozenset[str] = frozenset(
    {
        "rapport_alpha.pdf",
        "rapport_beta.pdf",
        "rapport_gamma.pdf",
        "report_alpha.pdf",
        "report_a.pdf",
        "report_b.pdf",
        "contrat_2025.pdf",
        "contrat_2024.pdf",
        "contract_2025.pdf",
        "facture_2025.pdf",
        "invoice_2025.pdf",
        "document_b.txt",
        "notice_produit.pdf",
        "manuel_histoire.pdf",
        "essai_philo.pdf",
        "note_rh.pdf",
        "politique_securite.pdf",
    }
)

_MOTIF_FICHIER = re.compile(r"[\w./-]+\.(?:pdf|txt|docx|csv|json)", re.IGNORECASE)


@pytest.fixture(scope="module")
def cas() -> list[dict]:
    return er.charger_cas(er.CHEMIN_DATASET_DEFAUT)


@pytest.fixture(scope="module")
def rapport(cas: list[dict]) -> er.RapportRoutage:
    return er.evaluer(cas, dataset="test")


@pytest.fixture(scope="module")
def bloc_multidoc(cas: list[dict]) -> dict:
    return er.evaluer_multidoc(cas)


# --------------------------------------------------------------------------
# 1. Validité du dataset
# --------------------------------------------------------------------------


def test_dataset_charge_et_non_vide(cas: list[dict]) -> None:
    assert len(cas) >= 45, "Le banc doit compter au moins 45 cas."
    assert len(cas) <= 90, "Banc anormalement volumineux — vérifier le fichier."


def test_dataset_champs_requis_presents(cas: list[dict]) -> None:
    requis = {"id", "query", "expected_intent", "category", "language", "notes"}
    for c in cas:
        manquants = requis - c.keys()
        assert not manquants, f"{c.get('id')} : champs manquants {manquants}."
        assert isinstance(c["query"], str) and c["query"].strip()
        assert isinstance(c["notes"], str) and c["notes"].strip()


def test_dataset_ids_uniques(cas: list[dict]) -> None:
    ids = [c["id"] for c in cas]
    doublons = {i for i in ids if ids.count(i) > 1}
    assert not doublons, f"Identifiants en double : {sorted(doublons)}."


def test_expected_intent_dans_taxonomie(cas: list[dict]) -> None:
    for c in cas:
        assert c["expected_intent"] in er.TAXONOMIE, (
            f"{c['id']} : intention {c['expected_intent']!r} hors taxonomie."
        )


def test_charger_cas_rejette_intention_hors_taxonomie(tmp_path) -> None:
    mauvais = tmp_path / "mauvais.jsonl"
    mauvais.write_text(
        '{"id": "X-1", "query": "q", "expected_intent": "TRANSLATE", '
        '"category": "generique", "language": "fr", "notes": "n"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        er.charger_cas(mauvais)


def test_charger_cas_rejette_id_en_double(tmp_path) -> None:
    mauvais = tmp_path / "doublon.jsonl"
    ligne = (
        '{"id": "X-1", "query": "q", "expected_intent": "SEARCH", '
        '"category": "generique", "language": "fr", "notes": "n"}\n'
    )
    mauvais.write_text(ligne + ligne, encoding="utf-8")
    with pytest.raises(ValueError):
        er.charger_cas(mauvais)


# --------------------------------------------------------------------------
# 2. Couverture
# --------------------------------------------------------------------------


def test_toutes_les_intentions_de_la_taxonomie_sont_couvertes(cas: list[dict]) -> None:
    presentes = {c["expected_intent"] for c in cas}
    assert presentes == set(er.TAXONOMIE), (
        f"Intentions non couvertes : {set(er.TAXONOMIE) - presentes}."
    )


def test_chaque_intention_a_plusieurs_cas(cas: list[dict]) -> None:
    from collections import Counter

    compte = Counter(c["expected_intent"] for c in cas)
    for intent in er.TAXONOMIE:
        assert compte[intent] >= 3, f"Intention {intent} sous-représentée ({compte[intent]})."


def test_francais_et_anglais_presents(cas: list[dict]) -> None:
    from collections import Counter

    langues = Counter(c["language"] for c in cas)
    assert set(langues) == {"fr", "en"}, f"Langues inattendues : {set(langues)}."
    assert langues["fr"] >= 10 and langues["en"] >= 10, (
        f"Déséquilibre linguistique : {dict(langues)}."
    )


def test_au_moins_30_pct_hors_histoire_culture(cas: list[dict]) -> None:
    hors = [c for c in cas if c["category"] != "histoire_culture"]
    part = len(hors) / len(cas)
    assert part >= 0.30, f"Seulement {part:.0%} des cas hors histoire/culture."


def test_diversite_des_domaines(cas: list[dict]) -> None:
    domaines = {c["category"] for c in cas}
    assert len(domaines) >= 6, f"Trop peu de domaines couverts : {sorted(domaines)}."


def test_aucun_artefact_cquae_ni_document_metier_reel(cas: list[dict]) -> None:
    for c in cas:
        q = c["query"].lower()
        assert "cquae" not in q, f"{c['id']} : référence CQuAE interdite."
        assert "cquae_doc" not in q
        for fichier in _MOTIF_FICHIER.findall(c["query"]):
            assert fichier in _DOCS_FICTIFS_AUTORISES, (
                f"{c['id']} : document non fictif {fichier!r} dans la requête."
            )


def test_facettes_structurelles_couvertes(cas: list[dict]) -> None:
    """Requêtes sans document, un document, deux documents, impératif,
    interrogatif — toutes présentes (nécessaire au futur détecteur multi-doc)."""
    doc_counts = {str(c.get("doc_count")) for c in cas}
    assert {"0", "1", "2"} <= doc_counts
    phrasings = {c.get("phrasing") for c in cas}
    assert {"imperative", "interrogative"} <= phrasings
    textes = " ".join(c["query"].lower() for c in cas)
    assert "ce document" in textes and "ces deux documents" in textes


# --------------------------------------------------------------------------
# 3. Runner : forme du rapport et cohérence arithmétique
# --------------------------------------------------------------------------


def test_rapport_structure_et_accuracy(rapport: er.RapportRoutage, cas: list[dict]) -> None:
    assert rapport.total == len(cas)
    assert 0.0 <= rapport.accuracy <= 1.0
    assert rapport.corrects == sum(
        1 for r in rapport.resultats if r["correct"]
    )
    attendu = rapport.corrects / rapport.total
    assert rapport.accuracy == pytest.approx(attendu, abs=1e-4)


def test_par_intention_somme_au_total(rapport: er.RapportRoutage) -> None:
    total = sum(int(s["total"]) for s in rapport.par_intention.values())
    corrects = sum(int(s["corrects"]) for s in rapport.par_intention.values())
    assert total == rapport.total
    assert corrects == rapport.corrects


def test_matrice_confusion_coherente(rapport: er.RapportRoutage) -> None:
    for attendu, ligne in rapport.matrice_confusion.items():
        somme_ligne = sum(ligne.values())
        assert somme_ligne == rapport.par_intention[attendu]["total"]
        # La diagonale = nombre de cas corrects pour cette intention.
        assert ligne.get(attendu, 0) == rapport.par_intention[attendu]["corrects"]


def test_echecs_sont_bien_des_desaccords(rapport: er.RapportRoutage) -> None:
    for e in rapport.echecs:
        assert e["expected_intent"] != e["routed_intent"]
    corrects = [r for r in rapport.resultats if r["correct"]]
    for r in corrects:
        assert r["routed_intent"] == r["expected_intent"]
    assert len(rapport.echecs) + len(corrects) == rapport.total


# --------------------------------------------------------------------------
# 4. Déterminisme
# --------------------------------------------------------------------------


def test_runner_deterministe(cas: list[dict]) -> None:
    r1 = er.evaluer(cas, dataset="d")
    r2 = er.evaluer(cas, dataset="d")
    assert r1.vers_dict() == r2.vers_dict()


def test_router_cas_stable_sur_plusieurs_appels() -> None:
    for query in (
        "Résume rapport_alpha.pdf.",
        "Compare rapport_alpha.pdf et rapport_beta.pdf.",
        "Quel est le montant total ?",
    ):
        sorties = {er.router_cas(query) for _ in range(5)}
        assert len(sorties) == 1


def test_sentinelles_repliees_sur_search_deterministe() -> None:
    routed, brut, deferred = er.router_cas(
        "Donne-moi le fournisseur, la date et le montant total de facture_2025.pdf."
    )
    assert brut == nodes._AMBIGU_SEARCH_EXTRACT
    assert routed == "SEARCH"
    assert deferred == "search_vs_extract"


# --------------------------------------------------------------------------
# 5. Aucun appel LLM / réseau / Qdrant
# --------------------------------------------------------------------------


def test_runner_n_appelle_jamais_les_desambiguiseurs_llm(
    monkeypatch: pytest.MonkeyPatch, cas: list[dict]
) -> None:
    def _interdit(*_a, **_k):  # pragma: no cover - ne doit jamais être atteint
        raise AssertionError("Le banc a appelé un désambiguïsateur LLM.")

    monkeypatch.setattr(nodes, "_desambiguiser_intention_classify", _interdit)
    monkeypatch.setattr(nodes, "_desambiguiser_intention_search_extract", _interdit)
    monkeypatch.setattr(nodes, "invoquer_llm", _interdit, raising=False)

    rapport = er.evaluer(cas, dataset="sans-llm")
    assert rapport.total == len(cas)


def test_runner_fonctionne_sans_llm_joignable(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.llm.common as llm_common

    def _explose(*_a, **_k):  # pragma: no cover
        raise RuntimeError("réseau LLM indisponible")

    monkeypatch.setattr(llm_common, "invoquer_llm", _explose, raising=False)
    rapport = er.executer(er.CHEMIN_DATASET_DEFAUT)
    assert rapport.total >= 45


# --------------------------------------------------------------------------
# 6. Garanties de baseline utiles pour P1.2 (référence, pas objectif)
# --------------------------------------------------------------------------


def test_baseline_search_intact(rapport: er.RapportRoutage) -> None:
    """SEARCH doit rester à 100 % (24/24) : c'est la garantie anti-faux-positif
    (les mots « points », « compare … dans ce document », « classification »,
    « type/nature », « récupère » ne doivent jamais détourner une question
    factuelle). Toute régression ici est un vrai défaut, pas un compromis."""
    assert rapport.par_intention["SEARCH"]["accuracy"] == 1.0
    assert rapport.par_intention["SEARCH"]["total"] == 24
    assert rapport.par_intention["SEARCH"]["corrects"] == 24


def test_deterministe_accuracy_plancher_post_p1_2(rapport: er.RapportRoutage) -> None:
    """Baseline P1.1 = 0.569 ; après P1.2 (11 cas bande B) ~0.72 en
    deterministic_only. Ce test est un PLANCHER anti-régression, pas un
    objectif à optimiser."""
    assert rapport.accuracy >= 0.69


def test_bande_b_summarize_et_extract_resolus_en_deterministe(
    rapport: er.RapportRoutage,
) -> None:
    """Les 5 cas SUMMARIZE et les 2 cas EXTRACT « récupère » de la bande B
    doivent être routés correctement sans LLM après P1.2."""
    par_id = {r["id"]: r for r in rapport.resultats}
    for cid in ("RT-028", "RT-031", "RT-032", "RT-033", "RT-034"):
        assert par_id[cid]["routed_intent"] == "SUMMARIZE", cid
    for cid in ("RT-048", "RT-049"):
        assert par_id[cid]["routed_intent"] == "EXTRACT", cid


def test_bande_b_classify_partiellement_deterministe(rapport: er.RapportRoutage) -> None:
    """RT-038/039/041 : résolus sans LLM. RT-040 : volontairement laissé en
    zone grise (repli SEARCH en deterministic_only, CLASSIFY en production)."""
    par_id = {r["id"]: r for r in rapport.resultats}
    for cid in ("RT-038", "RT-039", "RT-041"):
        assert par_id[cid]["routed_intent"] == "CLASSIFY", cid
    assert par_id["RT-040"]["routed_intent"] == "SEARCH"
    assert par_id["RT-040"]["raw_detected"] == "ambigu_classify"


def test_intentions_non_implementees_echouent_toutes(rapport: er.RapportRoutage) -> None:
    for intent in ("COMPARE", "SYNTHESIZE", "CLARIFY"):
        assert intent not in er.INTENTIONS_IMPLEMENTEES
        assert rapport.par_intention[intent]["corrects"] == 0


# --------------------------------------------------------------------------
# 7. Mesure P1.4 — détecteur multi-document (bloc séparé, hors routing)
# --------------------------------------------------------------------------


def test_multidoc_sous_ensemble_present(bloc_multidoc: dict, cas: list[dict]) -> None:
    assert bloc_multidoc["total"] == 14
    ids = {r["id"] for r in bloc_multidoc["resultats"]}
    assert {"RT-017", "RT-018", "RT-023", "RT-030"} <= ids
    assert {f"RT-{n:03d}" for n in range(52, 62)} <= ids
    # La mesure n'altère pas expected_intent : les cases restent celles du banc.
    par_id = {c["id"]: c for c in cas}
    assert par_id["RT-052"]["expected_intent"] == "COMPARE"


def test_multidoc_detection_parfaite(bloc_multidoc: dict) -> None:
    """Critère de sortie : sous-ensemble multi-doc >= 95 %."""
    assert bloc_multidoc["detection_accuracy"] >= 0.95
    assert bloc_multidoc["operation_hint_accuracy"] >= 0.95
    assert bloc_multidoc["exact_accuracy"] >= 0.95


def test_multidoc_rt017_rt018_restent_mono(bloc_multidoc: dict) -> None:
    par_id = {r["id"]: r for r in bloc_multidoc["resultats"]}
    for cid in ("RT-017", "RT-018"):
        assert par_id[cid]["detected_multidoc"] is False, cid
        assert par_id[cid]["detected_operation"] == "none", cid


def test_multidoc_rt030_synthetiser_un_doc_reste_mono(bloc_multidoc: dict) -> None:
    par_id = {r["id"]: r for r in bloc_multidoc["resultats"]}
    assert par_id["RT-030"]["detected_multidoc"] is False
    assert par_id["RT-030"]["detected_operation"] == "none"


def test_multidoc_rt052_057_signales_compare(bloc_multidoc: dict) -> None:
    par_id = {r["id"]: r for r in bloc_multidoc["resultats"]}
    for n in range(52, 58):
        r = par_id[f"RT-{n:03d}"]
        assert r["detected_multidoc"] is True, r["id"]
        assert r["detected_operation"] == "compare", r["id"]


def test_multidoc_rt058_061_signales_synthesize(bloc_multidoc: dict) -> None:
    par_id = {r["id"]: r for r in bloc_multidoc["resultats"]}
    for n in range(58, 62):
        r = par_id[f"RT-{n:03d}"]
        assert r["detected_multidoc"] is True, r["id"]
        assert r["detected_operation"] == "synthesize", r["id"]


def test_multidoc_rt023_multidoc_sans_operation(bloc_multidoc: dict) -> None:
    par_id = {r["id"]: r for r in bloc_multidoc["resultats"]}
    assert par_id["RT-023"]["detected_multidoc"] is True
    assert par_id["RT-023"]["detected_operation"] == "none"


def test_multidoc_deterministe(cas: list[dict]) -> None:
    assert er.evaluer_multidoc(cas) == er.evaluer_multidoc(cas)


def test_multidoc_aucun_faux_positif_sur_les_autres_cas_search(cas: list[dict]) -> None:
    """En dehors du sous-ensemble tagué, seul RT-064 (« these reports », un
    CLARIFY vague) déclenche is_multidoc — avec operation_hint none, ce qui
    est correct et inoffensif."""
    from src.agent.multidoc import detecter_multidoc

    tagues_true = {"RT-023", *(f"RT-{n:03d}" for n in range(52, 62))}
    faux_positifs = []
    for c in cas:
        if c["id"] in tagues_true:
            continue
        if detecter_multidoc(c["query"]).is_multidoc:
            faux_positifs.append(c["id"])
    assert faux_positifs == ["RT-064"], faux_positifs
