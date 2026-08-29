"""
Banc de mesure du ROUTAGE de l'agent documentaire (étape P1.1).

Ce script mesure **uniquement la détection d'intention** — pas la qualité de
la réponse finale, pas le retrieval, pas la génération. Il n'ouvre aucune
connexion réseau : ni Ollama, ni Qdrant, ni embeddings.

Logique mesurée
---------------
La seule fonction appelée du code applicatif est
`src.agent.nodes._detecter_intention` : le classifieur lexical déterministe
qui tranche l'intention dans l'immense majorité des cas, sans LLM.

Deux zones grises (`_AMBIGU_CLASSIFY`, `_AMBIGU_SEARCH_EXTRACT`) sont
normalement résolues par un classifieur LLM borné. Le banc **ne les appelle
jamais** et applique à la place leur **repli déterministe documenté** : toute
indisponibilité ou sortie invalide du LLM retombe sur ``"search"`` (voir
`nodes._desambiguiser_intention_classify` / `_desambiguiser_intention_search_extract`).
C'est exactement le comportement réel du graphe quand le LLM n'est pas
joignable. Le champ `deferred_to_llm` de chaque résultat signale les cas où
ce repli a joué, pour l'analyse.

Taxonomie cible
---------------
SEARCH, SUMMARIZE, CLASSIFY, EXTRACT, COMPARE, SYNTHESIZE, CLARIFY.

`_detecter_intention` ne sait produire que SEARCH / SUMMARIZE / CLASSIFY /
EXTRACT. COMPARE, SYNTHESIZE et CLARIFY ne sont pas encore implémentés : les
cas correspondants apparaîtront donc en échec dans le baseline — c'est
l'information recherchée, pas un bug du banc.

Usage
-----
    python -m evaluation.evaluate_routing
    python -m evaluation.evaluate_routing --jsonl evaluation/data/routing_cases.jsonl
    python -m evaluation.evaluate_routing --json /tmp/routing.json --quiet
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.common import (
    DOSSIER_DONNEES,
    DOSSIER_RAPPORTS,
    configurer_logs,
    horodatage,
    lire_jsonl,
)
from src.agent import nodes

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------

TAXONOMIE: tuple[str, ...] = (
    "SEARCH",
    "SUMMARIZE",
    "CLASSIFY",
    "EXTRACT",
    "COMPARE",
    "SYNTHESIZE",
    "CLARIFY",
)

# Intentions que le routage actuel sait réellement produire.
INTENTIONS_IMPLEMENTEES: frozenset[str] = frozenset(
    {"SEARCH", "SUMMARIZE", "CLASSIFY", "EXTRACT"}
)

# Deux modes de mesure, JAMAIS fusionnés en un score unique :
#   - deterministic_only : les 2 zones grises retombent sur leur repli
#     déterministe documenté ("search"). Aucun appel LLM.
#   - production_routing : les 2 zones grises sont résolues par LES MÊMES
#     désambiguïsateurs bornés que le graphe (`nodes._desambiguiser_intention_*`),
#     avec un vrai LLM. Leurs prompts ne sont pas modifiés.
MODE_DETERMINISTE = "deterministic_only"
MODE_PRODUCTION = "production_routing"
MODES: tuple[str, ...] = (MODE_DETERMINISTE, MODE_PRODUCTION)

# Repli déterministe des deux zones grises (comportement réel du graphe
# quand le LLM n'est pas joignable).
_SENTINELLES_VERS_REPLI: dict[str, str] = {
    nodes._AMBIGU_CLASSIFY: "search",
    nodes._AMBIGU_SEARCH_EXTRACT: "search",
}
_ETIQUETTE_SENTINELLE: dict[str, str] = {
    nodes._AMBIGU_CLASSIFY: "classify_vs_search",
    nodes._AMBIGU_SEARCH_EXTRACT: "search_vs_extract",
}
# Désambiguïsateurs bornés RÉELS du graphe, utilisés tels quels en mode
# production_routing (aucune modification de leurs prompts).
_SENTINELLE_VERS_DESAMBIGUISEUR = {
    nodes._AMBIGU_CLASSIFY: nodes._desambiguiser_intention_classify,
    nodes._AMBIGU_SEARCH_EXTRACT: nodes._desambiguiser_intention_search_extract,
}

CHEMIN_DATASET_DEFAUT: Path = DOSSIER_DONNEES / "routing_cases.jsonl"
DOSSIER_RAPPORTS_ROUTAGE: Path = DOSSIER_RAPPORTS / "routing"

_ORDRE_COLONNES: dict[str, int] = {nom: i for i, nom in enumerate(TAXONOMIE)}


# --------------------------------------------------------------------------
# Structures de sortie
# --------------------------------------------------------------------------


@dataclass
class ResultatCas:
    id: str
    query: str
    expected_intent: str
    routed_intent: str
    raw_detected: str
    deferred_to_llm: str | None
    correct: bool
    category: str
    language: str
    grey_zone: Any
    notes: str

    def vers_dict(self) -> dict[str, Any]:
        return dict(vars(self))


@dataclass
class RapportRoutage:
    dataset: str
    mode: str
    total: int
    corrects: int
    accuracy: float
    par_intention: dict[str, dict[str, float]]
    matrice_confusion: dict[str, dict[str, int]]
    echecs: list[dict[str, Any]]
    resultats: list[dict[str, Any]]

    def resume(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "mode": self.mode,
            "total": self.total,
            "corrects": self.corrects,
            "echecs": len(self.echecs),
            "accuracy": self.accuracy,
        }

    def vers_dict(self) -> dict[str, Any]:
        return {
            "resume": self.resume(),
            "par_intention": self.par_intention,
            "matrice_confusion": self.matrice_confusion,
            "echecs": self.echecs,
            "resultats": self.resultats,
        }


# --------------------------------------------------------------------------
# Cœur : routage déterministe d'un cas
# --------------------------------------------------------------------------


def router_cas(
    query: str,
    *,
    mode: str = MODE_DETERMINISTE,
    llm: Any = None,
) -> tuple[str, str, str | None]:
    """
    Applique la détection d'intention réelle (`nodes._detecter_intention`)
    puis résout les deux zones grises selon `mode` :

    - ``deterministic_only`` : repli déterministe documenté (``"search"``).
      Aucun appel LLM / réseau.
    - ``production_routing`` : appel des désambiguïsateurs bornés RÉELS du
      graphe avec `llm`. Prompts inchangés.

    Renvoie ``(routed_intent_majuscule, raw_detected, deferred_to_llm | None)``.
    """
    if mode not in MODES:
        raise ValueError(f"Mode inconnu : {mode!r} (attendu : {MODES}).")

    brut = nodes._detecter_intention(query)
    if brut not in _SENTINELLES_VERS_REPLI:
        return brut.upper(), brut, None

    etiquette = _ETIQUETTE_SENTINELLE[brut]

    if mode == MODE_DETERMINISTE:
        return _SENTINELLES_VERS_REPLI[brut].upper(), brut, etiquette

    if llm is None:
        raise ValueError(
            "mode 'production_routing' exige un LLM (désambiguïsateurs bornés)."
        )
    resolu = _SENTINELLE_VERS_DESAMBIGUISEUR[brut](llm, query)
    return resolu.upper(), brut, etiquette


# --------------------------------------------------------------------------
# Chargement et validation du dataset
# --------------------------------------------------------------------------

_CHAMPS_REQUIS: tuple[str, ...] = (
    "id",
    "query",
    "expected_intent",
    "category",
    "language",
    "notes",
)


def charger_cas(chemin: Path) -> list[dict[str, Any]]:
    """Charge le JSONL et vérifie la forme minimale de chaque ligne."""
    cas = list(lire_jsonl(chemin))
    if not cas:
        raise ValueError(f"Dataset de routage vide : {chemin}")

    vus: set[str] = set()
    for i, c in enumerate(cas, start=1):
        manquants = [champ for champ in _CHAMPS_REQUIS if champ not in c]
        if manquants:
            raise ValueError(
                f"Ligne {i} ({c.get('id', '?')}) : champs manquants {manquants}."
            )
        identifiant = str(c["id"])
        if identifiant in vus:
            raise ValueError(f"Identifiant en double dans le dataset : {identifiant!r}.")
        vus.add(identifiant)

        attendu = str(c["expected_intent"]).upper()
        if attendu not in TAXONOMIE:
            raise ValueError(
                f"{identifiant} : expected_intent {attendu!r} hors taxonomie {TAXONOMIE}."
            )
    return cas


# --------------------------------------------------------------------------
# Évaluation
# --------------------------------------------------------------------------


def evaluer(
    cas: list[dict[str, Any]],
    dataset: str = "",
    *,
    mode: str = MODE_DETERMINISTE,
    llm: Any = None,
) -> RapportRoutage:
    resultats: list[ResultatCas] = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    totaux: Counter[str] = Counter()
    corrects_par_intention: Counter[str] = Counter()

    for c in cas:
        attendu = str(c["expected_intent"]).upper()
        routed, brut, deferred = router_cas(str(c["query"]), mode=mode, llm=llm)
        correct = routed == attendu

        totaux[attendu] += 1
        if correct:
            corrects_par_intention[attendu] += 1
        confusion[attendu][routed] += 1

        resultats.append(
            ResultatCas(
                id=str(c.get("id", "")),
                query=str(c["query"]),
                expected_intent=attendu,
                routed_intent=routed,
                raw_detected=brut,
                deferred_to_llm=deferred,
                correct=correct,
                category=str(c.get("category", "")),
                language=str(c.get("language", "")),
                grey_zone=c.get("grey_zone"),
                notes=str(c.get("notes", "")),
            )
        )

    total = len(resultats)
    corrects = sum(1 for r in resultats if r.correct)

    par_intention: dict[str, dict[str, float]] = {}
    for intent in sorted(totaux, key=lambda n: _ORDRE_COLONNES.get(n, 99)):
        t = totaux[intent]
        ok = corrects_par_intention[intent]
        par_intention[intent] = {
            "total": t,
            "corrects": ok,
            "accuracy": round(ok / t, 4) if t else 0.0,
        }

    matrice = {
        attendu: dict(
            sorted(
                confusion[attendu].items(),
                key=lambda kv: _ORDRE_COLONNES.get(kv[0], 99),
            )
        )
        for attendu in sorted(confusion, key=lambda n: _ORDRE_COLONNES.get(n, 99))
    }

    echecs = [r.vers_dict() for r in resultats if not r.correct]

    return RapportRoutage(
        dataset=dataset,
        mode=mode,
        total=total,
        corrects=corrects,
        accuracy=round(corrects / total, 4) if total else 0.0,
        par_intention=par_intention,
        matrice_confusion=matrice,
        echecs=echecs,
        resultats=[r.vers_dict() for r in resultats],
    )


# --------------------------------------------------------------------------
# Affichage
# --------------------------------------------------------------------------


def _afficher(rapport: RapportRoutage) -> None:
    largeur = 78
    libelle_mode = {
        MODE_DETERMINISTE: "deterministic_only — détection lexicale seule, sans LLM",
        MODE_PRODUCTION: "production_routing — zones grises résolues par les désambiguïsateurs bornés (LLM réel)",
    }.get(rapport.mode, rapport.mode)
    print("=" * largeur)
    print(f"BANC DE ROUTAGE — {libelle_mode}")
    print("=" * largeur)
    print(f"Dataset          : {rapport.dataset}")
    print(f"Mode             : {rapport.mode}")
    print(f"Cas              : {rapport.total}")
    print(
        f"Accuracy globale : {rapport.accuracy:.1%} "
        f"({rapport.corrects}/{rapport.total})"
    )
    print()

    print("Accuracy par intention attendue")
    print("-" * largeur)
    print(f"  {'intention':<12} {'ok/total':>10} {'accuracy':>10}   statut")
    for intent, stats in rapport.par_intention.items():
        statut = (
            "implémentée"
            if intent in INTENTIONS_IMPLEMENTEES
            else "NON implémentée (attendu : 0 %)"
        )
        print(
            f"  {intent:<12} "
            f"{str(int(stats['corrects'])) + '/' + str(int(stats['total'])):>10} "
            f"{stats['accuracy']:>9.1%}   {statut}"
        )
    print()

    colonnes: list[str] = sorted(
        {routed for ligne in rapport.matrice_confusion.values() for routed in ligne},
        key=lambda n: _ORDRE_COLONNES.get(n, 99),
    )
    print("Matrice de confusion  (ligne = attendu, colonne = routé)")
    print("-" * largeur)
    coin = "attendu / route"
    entete = "  " + f"{coin:<16}" + "".join(f"{c[:9]:>10}" for c in colonnes)
    print(entete)
    for attendu in rapport.matrice_confusion:
        ligne = rapport.matrice_confusion[attendu]
        cellules = "".join(f"{ligne.get(c, 0):>10}" for c in colonnes)
        print(f"  {attendu:<16}{cellules}")
    print()

    print(f"Cas échoués ({len(rapport.echecs)})")
    print("-" * largeur)
    if not rapport.echecs:
        print("  (aucun)")
    for e in rapport.echecs:
        deferred = f"  [repli {e['deferred_to_llm']}]" if e["deferred_to_llm"] else ""
        print(
            f"  {e['id']:<7} {e['expected_intent']:>10} -> {e['routed_intent']:<10}"
            f" (brut: {e['raw_detected']}){deferred}"
        )
        print(f"          « {e['query']} »")
        if e["notes"]:
            print(f"          note: {e['notes']}")
    print("=" * largeur)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _ecrire_json(charge: dict[str, Any], chemin: Path) -> Path:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(
        json.dumps(charge, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return chemin


def _construire_llm_borne() -> Any:
    """Client LLM du projet (Ollama), pour le mode production_routing."""
    from src.llm.factory import construire_llm

    return construire_llm()


def executer(
    chemin_dataset: Path,
    *,
    mode: str = MODE_DETERMINISTE,
    llm: Any = None,
) -> RapportRoutage:
    cas = charger_cas(chemin_dataset)
    if mode == MODE_PRODUCTION and llm is None:
        llm = _construire_llm_borne()
    return evaluer(cas, dataset=str(chemin_dataset), mode=mode, llm=llm)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Banc de mesure du routage d'intention.")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=CHEMIN_DATASET_DEFAUT,
        help="Dataset de cas de routage (JSONL).",
    )
    parser.add_argument(
        "--mode",
        choices=(*MODES, "both"),
        default=MODE_DETERMINISTE,
        help=(
            "deterministic_only (défaut) : sans LLM. "
            "production_routing : zones grises résolues par les désambiguïsateurs "
            "bornés réels. both : les deux, mesures séparées, jamais fusionnées."
        ),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Chemin du rapport JSON (défaut : evaluation/reports/routing/routing_<mode>_<horodatage>.json).",
    )
    parser.add_argument("--quiet", action="store_true", help="Ne pas afficher le rapport.")
    args = parser.parse_args(argv)

    configurer_logs(verbeux=False)

    modes_a_lancer = MODES if args.mode == "both" else (args.mode,)
    rapports: dict[str, RapportRoutage] = {}
    llm = _construire_llm_borne() if MODE_PRODUCTION in modes_a_lancer else None

    for mode in modes_a_lancer:
        rapports[mode] = executer(
            args.jsonl,
            mode=mode,
            llm=llm if mode == MODE_PRODUCTION else None,
        )
        if not args.quiet:
            _afficher(rapports[mode])
            print()

    # Sérialisation : chaque mode conserve son propre bloc. Aucun score
    # combiné n'est jamais calculé.
    if args.mode == "both":
        charge: dict[str, Any] = {m: rapports[m].vers_dict() for m in MODES}
        defaut = DOSSIER_RAPPORTS_ROUTAGE / f"routing_both_{horodatage()}.json"
    else:
        charge = rapports[args.mode].vers_dict()
        defaut = DOSSIER_RAPPORTS_ROUTAGE / f"routing_{args.mode}_{horodatage()}.json"

    ecrit = _ecrire_json(charge, args.json or defaut)
    if not args.quiet:
        print(f"Rapport JSON écrit : {ecrit}")

    # Étape de mesure : la sortie est toujours 0, l'échec de cas n'est pas
    # une erreur d'exécution.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
