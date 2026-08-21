"""
Analyse des échecs.

Consomme les rapports produits par les autres scripts et répond à la
question opérationnelle : par quoi commencer ?

Le script ne relance ni le retrieval ni la génération. Il agrège, croise et
priorise ce qui a déjà été mesuré — un run coûteux ne doit pas être répété
pour être analysé sous un autre angle.

Sorties
-------
    <nom>.json / <nom>.csv   toutes les questions en échec, avec le document
                             attendu, les documents récupérés, les rangs, la
                             réponse produite, la réponse gold, l'evidence
                             gold, le type d'échec et les scores.
    <nom>_synthese.csv       agrégats par catégorie, par sous-jeu et par
                             document, pour repérer les échecs concentrés.

Exemple
-------
    python -m evaluation.analyze_failures \\
        --end-to-end evaluation/reports/end_to_end/baseline.json \\
        --evidence evaluation/reports/retrieval_evidence/baseline.json \\
        --document evaluation/reports/retrieval_document/baseline.json \\
        --nom analyse_v1 --top 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluation.common import (
    DOSSIER_RAPPORTS,
    configurer_logs,
    ecrire_rapport,
    horodatage,
    moyenne,
)
from evaluation.evaluate_end_to_end import (
    CORRECT_SANS_EVIDENCE,
    ECHEC_GENERATION,
    ECHEC_RETRIEVAL,
    FAUX_POSITIF,
    REFUS_INCORRECT,
    SUCCES,
    SUCCES_CITATION_INVALIDE,
)

logger = logging.getLogger("evaluation.failures")

# Catégories considérées comme des échecs à analyser. Le succès avec citation
# invalide en fait partie : la réponse est juste, mais la traçabilité est
# cassée, ce qui est bloquant pour un usage d'entreprise.
CATEGORIES_ECHEC = {
    ECHEC_RETRIEVAL,
    ECHEC_GENERATION,
    CORRECT_SANS_EVIDENCE,
    REFUS_INCORRECT,
    FAUX_POSITIF,
    SUCCES_CITATION_INVALIDE,
}

# Ordre de priorité de traitement, du plus grave au moins grave.
# FAUX_POSITIF d'abord : inventer une réponse là où il n'y en a pas est la
# défaillance la plus coûteuse pour un agent documentaire d'entreprise.
PRIORITE = [
    FAUX_POSITIF,
    CORRECT_SANS_EVIDENCE,
    ECHEC_RETRIEVAL,
    ECHEC_GENERATION,
    SUCCES_CITATION_INVALIDE,
    REFUS_INCORRECT,
]


def charger_rapport(chemin: Path | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Lit un rapport produit par `ecrire_rapport`."""
    if chemin is None:
        return {}, []
    if not chemin.exists():
        raise FileNotFoundError(f"Rapport introuvable : {chemin}")
    donnees = json.loads(chemin.read_text(encoding="utf-8"))
    return donnees.get("resume", {}), donnees.get("details", [])


def indexer(details: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {ligne["id"]: ligne for ligne in details if "id" in ligne}


def construire_echecs(
    end_to_end: list[dict[str, Any]],
    evidence: dict[str, dict[str, Any]],
    document: dict[str, dict[str, Any]],
    *,
    tronquer: int = 400,
) -> list[dict[str, Any]]:
    """
    Assemble une fiche d'échec par question, en croisant les trois rapports.

    Les textes longs sont tronqués : le CSV doit rester ouvrable dans un
    tableur, et le JSON conserve de toute façon l'intégralité via les
    rapports d'origine.
    """

    def court(valeur: Any) -> str:
        texte = str(valeur or "")
        return texte if len(texte) <= tronquer else texte[:tronquer] + " […]"

    echecs: list[dict[str, Any]] = []

    for ligne in end_to_end:
        categorie = ligne.get("categorie")
        if categorie not in CATEGORIES_ECHEC:
            continue

        identifiant = ligne.get("id", "")
        ligne_evidence = evidence.get(identifiant, {})
        ligne_document = document.get(identifiant, {})

        echecs.append(
            {
                "id": identifiant,
                "categorie": categorie,
                "priorite": (
                    PRIORITE.index(categorie) if categorie in PRIORITE else 99
                ),
                "subset": ligne.get("subset", ""),
                "answerable": ligne.get("answerable"),
                "question": court(ligne.get("question")),
                # --- Vérité terrain
                "expected_document": ligne.get("expected_document", ""),
                "expected_answer": court(ligne.get("expected_answer")),
                "evidence_gold": court(
                    ligne_evidence.get("evidence_gold")
                    or ligne.get("evidence_text")
                    or ""
                ),
                # --- Ce que le système a produit
                "reponse": court(ligne.get("reponse")),
                "documents_cites": ligne.get("documents_cites", ""),
                "documents_recuperes": ligne_document.get("documents_recuperes", ""),
                "rang_document_attendu": ligne_document.get("rang_document_attendu"),
                "rang_premier_passage_utile": ligne.get("rang_premier_utile"),
                # --- Scores intermédiaires
                "couverture_relative": ligne.get("couverture_relative"),
                "plafond": ligne_evidence.get("plafond"),
                "groundedness": ligne.get("groundedness"),
                "contexte_suffisant": ligne.get("contexte_suffisant"),
                "citations_valides": ligne.get("citations_valides"),
                "citations_hors_perimetre": ligne.get("citations_hors_perimetre"),
                "nombre_sources": ligne.get("nombre_sources"),
                "perimetre_statut": ligne.get("perimetre_statut", ""),
                "detail_exactitude": court(ligne.get("detail_exactitude")),
                "duree_secondes": ligne.get("duree_secondes"),
            }
        )

    echecs.sort(key=lambda e: (e["priorite"], e["id"]))
    return echecs


def synthetiser(
    end_to_end: list[dict[str, Any]],
    echecs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Agrégats destinés à repérer les échecs concentrés.

    Un taux d'échec global de 30 % n'a pas la même signification selon qu'il
    est réparti uniformément ou concentré sur trois documents mal parsés.
    """
    lignes: list[dict[str, Any]] = []

    # --- Par catégorie
    total = len(end_to_end)
    compteur = Counter(e["categorie"] for e in echecs)
    for categorie in PRIORITE:
        nombre = compteur.get(categorie, 0)
        if not nombre:
            continue
        concernes = [e for e in echecs if e["categorie"] == categorie]
        lignes.append(
            {
                "axe": "categorie",
                "cle": categorie,
                "echecs": nombre,
                "part_du_total": round(nombre / total, 4) if total else 0.0,
                "couverture_relative_moyenne": round(
                    moyenne(
                        [
                            e["couverture_relative"]
                            for e in concernes
                            if e["couverture_relative"] is not None
                        ]
                    ),
                    4,
                ),
                "groundedness_moyenne": round(
                    moyenne(
                        [
                            e["groundedness"]
                            for e in concernes
                            if e["groundedness"] is not None
                        ]
                    ),
                    4,
                ),
            }
        )

    # --- Par sous-jeu
    par_subset_total = Counter(l.get("subset", "") for l in end_to_end)
    par_subset_echec = Counter(e["subset"] for e in echecs)
    for subset, nombre in par_subset_echec.most_common():
        denominateur = par_subset_total.get(subset, 0)
        lignes.append(
            {
                "axe": "subset",
                "cle": subset,
                "echecs": nombre,
                "questions": denominateur,
                "taux_echec": round(nombre / denominateur, 4) if denominateur else 0.0,
            }
        )

    # --- Par document attendu : révèle les documents systématiquement ratés.
    par_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for echec in echecs:
        if echec["expected_document"]:
            par_document[echec["expected_document"]].append(echec)

    total_par_document = Counter(
        l.get("expected_document", "") for l in end_to_end
    )
    for document, concernes in sorted(
        par_document.items(), key=lambda item: -len(item[1])
    )[:30]:
        denominateur = total_par_document.get(document, 0)
        lignes.append(
            {
                "axe": "document",
                "cle": document,
                "echecs": len(concernes),
                "questions": denominateur,
                "taux_echec": round(len(concernes) / denominateur, 4)
                if denominateur
                else 0.0,
                "plafond_moyen": round(
                    moyenne(
                        [
                            e["plafond"]
                            for e in concernes
                            if e["plafond"] is not None
                        ]
                    ),
                    4,
                ),
            }
        )

    return lignes


def construire_parseur() -> argparse.ArgumentParser:
    parseur = argparse.ArgumentParser(
        description="Analyse et priorise les échecs mesurés."
    )
    parseur.add_argument(
        "--end-to-end",
        type=Path,
        required=True,
        help="rapport JSON produit par evaluate_end_to_end.py",
    )
    parseur.add_argument(
        "--evidence",
        type=Path,
        default=None,
        help="rapport JSON produit par evaluate_retrieval_evidence.py",
    )
    parseur.add_argument(
        "--document",
        type=Path,
        default=None,
        help="rapport JSON produit par evaluate_retrieval_document.py",
    )
    parseur.add_argument("--nom", default=None)
    parseur.add_argument(
        "--sortie", type=Path, default=DOSSIER_RAPPORTS / "failures"
    )
    parseur.add_argument(
        "--top", type=int, default=10, help="échecs affichés en console"
    )
    parseur.add_argument(
        "--categorie",
        default=None,
        help="n'analyser qu'une seule catégorie d'échec",
    )
    parseur.add_argument("--tronquer", type=int, default=400)
    parseur.add_argument("--verbose", action="store_true")
    return parseur


def main(argv: list[str] | None = None) -> int:
    arguments = construire_parseur().parse_args(argv)
    configurer_logs(arguments.verbose)

    resume_e2e, details_e2e = charger_rapport(arguments.end_to_end)
    _, details_evidence = charger_rapport(arguments.evidence)
    _, details_document = charger_rapport(arguments.document)

    if not details_e2e:
        logger.error("Rapport end-to-end vide : %s", arguments.end_to_end)
        return 1

    echecs = construire_echecs(
        details_e2e,
        indexer(details_evidence),
        indexer(details_document),
        tronquer=arguments.tronquer,
    )

    if arguments.categorie:
        echecs = [e for e in echecs if e["categorie"] == arguments.categorie]

    synthese = synthetiser(details_e2e, echecs)

    total = len(details_e2e)
    reussites = sum(
        1
        for l in details_e2e
        if l.get("categorie") in {SUCCES, SUCCES_CITATION_INVALIDE}
    )

    resume = {
        "source_end_to_end": str(arguments.end_to_end),
        "questions": total,
        "reussites": reussites,
        "echecs": len(echecs),
        "taux_echec": round(len(echecs) / total, 4) if total else 0.0,
        "repartition": dict(Counter(e["categorie"] for e in echecs)),
        "resume_end_to_end": resume_e2e,
    }

    nom = arguments.nom or f"failures_{horodatage()}"
    ecrire_rapport(arguments.sortie, nom, resume=resume, details=echecs)

    if synthese:
        import pandas as pd

        chemin_synthese = arguments.sortie / f"{nom}_synthese.csv"
        pd.DataFrame(synthese).to_csv(
            chemin_synthese, index=False, encoding="utf-8"
        )
        logger.info("Synthèse écrite : %s", chemin_synthese)

    # --- Console ------------------------------------------------------------
    print()
    print(f"  {len(echecs)} échec(s) sur {total} question(s)")
    print()
    print("  PAR CATÉGORIE (ordre de priorité de traitement)")
    for categorie in PRIORITE:
        nombre = resume["repartition"].get(categorie, 0)
        if nombre:
            print(f"    {categorie:<26} {nombre:4d}")

    documents = [l for l in synthese if l["axe"] == "document"]
    concentres = [
        l for l in documents if l.get("questions", 0) >= 3 and l["taux_echec"] >= 0.8
    ]
    if concentres:
        print()
        print("  DOCUMENTS QUASI SYSTÉMATIQUEMENT EN ÉCHEC")
        for ligne in concentres[:10]:
            print(
                f"    {ligne['cle'][:52]:<52} "
                f"{ligne['echecs']}/{ligne['questions']} "
                f"(plafond {ligne.get('plafond_moyen', 0):.2f})"
            )
        print(
            "\n    Un plafond bas sur ces documents désigne le PARSING,\n"
            "    pas le retrieval."
        )

    print()
    print(f"  {min(arguments.top, len(echecs))} PREMIERS ÉCHECS")
    for echec in echecs[: arguments.top]:
        print()
        print(f"    [{echec['categorie']}] {echec['id']}")
        print(f"      Q      : {echec['question'][:110]}")
        print(f"      gold   : {echec['expected_answer'][:110]}")
        print(f"      obtenu : {echec['reponse'][:110]}")
        print(
            f"      doc attendu {echec['expected_document']} | "
            f"rang {echec['rang_document_attendu']} | "
            f"couverture {echec['couverture_relative']}"
        )
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())