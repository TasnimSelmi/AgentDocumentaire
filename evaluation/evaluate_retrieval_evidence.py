"""
Évaluation du retrieval au niveau passage / evidence.

Répond à une question plus exigeante que le document-level : le système a-t-il
récupéré le passage réellement nécessaire pour répondre ?

Pourquoi aucune égalité exacte
------------------------------
L'evidence gold et les chunks du projet proviennent de deux découpages
différents. Une evidence peut être plus longue qu'un chunk, ou couper au
milieu d'un autre. Pire, elle peut avoir été extraite par un parseur
différent du `pdfplumber` du projet : mesuré sur le corpus ESG, jusqu'à 17 %
des jetons d'une evidence gold restaient introuvables **même en réunissant
tous les chunks du bon document**. Un seuil absolu ne mesurerait alors que
la divergence des parseurs.

Les trois quantités mesurées
----------------------------
    plafond   part des jetons gold présents dans les chunks du document
              attendu. Borne haute imposée par le parsing, pas par le
              retrieval. C'est un diagnostic d'extraction, utile en soi.

    atteint   part des jetons gold présents dans les passages récupérés.

    couverture_relative = atteint / plafond
              LA métrique à lire. Vaut 1.0 quand le retrieval a ramené tout
              ce qu'il était possible de ramener. Neutralise les pertes de
              parsing pour n'évaluer que la recherche.

Métriques complémentaires
-------------------------
    rang_premier_utile   rang du premier passage contribuant à l'evidence,
                         au sens de `Passage.rang`.
    passages_pour_80pct  nombre de passages nécessaires pour atteindre 80 %
                         du plafond — mesure directe de l'adéquation du
                         chunking, sans le modifier.
    recouvrement_ngrammes  contrôle anti-coïncidence : deux textes peuvent
                         partager beaucoup de mots fréquents sans partager
                         la moindre formulation.

Exemple
-------
    python -m evaluation.evaluate_retrieval_evidence \\
        --questions evaluation/data/uda_400.jsonl --nom baseline
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

from evaluation.common import (
    DOSSIER_RAPPORTS,
    Enregistrement,
    attendre_client_qdrant,
    charger_enregistrements,
    charger_textes_par_document,
    cle_document,
    configurer_logs,
    ecrire_rapport,
    horodatage,
    mesurer_couverture,
    moyenne,
    percentile,
)
from src.rag.retrieval import rechercher_passages
from src.rag.vectorstore import fermer_client

logger = logging.getLogger("evaluation.retrieval_evidence")

# Seuil de « couverture suffisante ». Il porte sur la couverture RELATIVE,
# donc sur ce qui était atteignable : il ne pénalise pas le parsing.
SEUIL_COUVERTURE = 0.70


def evaluer(
    enregistrements: Sequence[Enregistrement],
    *,
    top_k: int | None = None,
    limite_candidats: int | None = None,
    utiliser_reranker: bool = True,
    appliquer_seuil: bool = False,
    seuil_pertinence: float | None = None,
    max_par_document: int = 3,
    resolution_document: bool = True,
    seuil_couverture: float = SEUIL_COUVERTURE,
    taille_ngramme: int = 3,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Mesure la couverture d'evidence sur chaque question."""
    logger.info("Lecture du corpus indexé pour le calcul des plafonds…")
    textes_par_document = charger_textes_par_document()

    details: list[dict[str, Any]] = []
    erreurs = 0
    sans_plafond = 0

    for index, enregistrement in enumerate(enregistrements, start=1):
        cle_attendue = cle_document(enregistrement.expected_document or "")
        textes_document = textes_par_document.get(cle_attendue, [])

        if not textes_document:
            sans_plafond += 1

        ligne: dict[str, Any] = {
            "id": enregistrement.id,
            "question": enregistrement.question,
            "subset": enregistrement.subset,
            "expected_document": enregistrement.expected_document,
            "document_indexe": bool(textes_document),
            "page_gold": enregistrement.page,
        }

        try:
            rapport = rechercher_passages(
                enregistrement.question,
                top_k=top_k,
                limite_candidats=limite_candidats,
                utiliser_reranker=utiliser_reranker,
                appliquer_seuil=appliquer_seuil,
                seuil_pertinence=seuil_pertinence,
                max_par_document=max_par_document,
                resolution_document=resolution_document,
            )
        except Exception as exc:  # noqa: BLE001
            erreurs += 1
            logger.error("Erreur sur %s : %s", enregistrement.id, exc)
            ligne["erreur"] = f"{type(exc).__name__}: {exc}"
            details.append(ligne)
            continue

        # La couverture n'a de sens que sur les passages provenant du
        # document attendu : un passage d'un autre document qui partagerait
        # du vocabulaire ne constitue pas une preuve.
        passages_du_document = [
            passage
            for passage in rapport.passages
            if cle_document(passage.nom_fichier or passage.source) == cle_attendue
        ]

        couverture = mesurer_couverture(
            enregistrement.evidence_text,
            passages_du_document,
            textes_du_document=textes_document,
            taille_ngramme=taille_ngramme,
        )

        # Les pages effectivement récupérées pour le document attendu.
        pages = sorted(
            {p.page for p in passages_du_document if p.page is not None}
        )
        page_gold_atteinte = (
            enregistrement.page in pages if enregistrement.page is not None else None
        )

        ligne.update(couverture.vers_dict())
        ligne.update(
            {
                "erreur": "",
                "passages_total": len(rapport.passages),
                "passages_du_document": len(passages_du_document),
                # Le document attendu a-t-il été retrouvé par le retrieval,
                # indépendamment de la couverture d'evidence qu'il apporte ?
                "document_retrouve": bool(passages_du_document),
                "pages_recuperees": "|".join(str(p) for p in pages),
                "page_gold_atteinte": page_gold_atteinte,
                "couverture_suffisante": couverture.couverture_relative
                >= seuil_couverture,
                "duree_secondes": round(rapport.duree_secondes, 3),
            }
        )
        details.append(ligne)

        if index % 25 == 0:
            logger.info("… %d/%d questions", index, len(enregistrements))

    # --- Agrégation ---------------------------------------------------------
    evalues = [l for l in details if not l.get("erreur")]
    # Le plafond n'est interprétable que si le document est bien indexé.
    mesurables = [l for l in evalues if l.get("document_indexe")]

    plafonds = [l["plafond"] for l in mesurables]
    atteints = [l["atteint"] for l in mesurables]
    relatives = [l["couverture_relative"] for l in mesurables]
    # Sous-ensemble où le document attendu a été retrouvé par le retrieval.
    # Sur le reste, atteint vaut mécaniquement 0 : les mélanger aux couvertures
    # ci-dessus confond « le document n'a pas été retrouvé » (échec de
    # retrieval document-level) et « le document a été retrouvé mais
    # l'evidence n'y est pas bien couverte » (échec de retrieval evidence-level).
    retrouves = [l for l in mesurables if l.get("document_retrouve")]
    atteints_conditionnes = [l["atteint"] for l in retrouves]
    relatives_conditionnees = [l["couverture_relative"] for l in retrouves]
    rangs = [
        l["rang_premier_utile"]
        for l in mesurables
        if l.get("rang_premier_utile") is not None
    ]
    pour_80 = [
        l["passages_pour_80pct"]
        for l in mesurables
        if l.get("passages_pour_80pct") is not None
    ]
    pages_ok = [
        l["page_gold_atteinte"]
        for l in mesurables
        if l.get("page_gold_atteinte") is not None
    ]

    resume: dict[str, Any] = {
        "questions": len(details),
        "evaluees": len(evalues),
        "mesurables": len(mesurables),
        "documents_non_indexes": sans_plafond,
        "erreurs": erreurs,
        "seuil_couverture": seuil_couverture,
        # --- Diagnostic d'extraction (indépendant du retrieval)
        "plafond_moyen": round(moyenne(plafonds), 4),
        "plafond_median": round(percentile(plafonds, 0.5), 4),
        "plafond_p10": round(percentile(plafonds, 0.10), 4),
        "part_plafond_sup_090": round(
            sum(1 for p in plafonds if p >= 0.90) / len(plafonds)
            if plafonds
            else 0.0,
            4,
        ),
        # --- Retrieval brut, GLOBAL (toutes les questions mesurables, y
        # compris celles où le document attendu n'a même pas été retrouvé)
        "atteint_moyen": round(moyenne(atteints), 4),
        "atteint_median": round(percentile(atteints, 0.5), 4),
        # --- LA métrique, GLOBALE
        "couverture_relative_moyenne": round(moyenne(relatives), 4),
        "couverture_relative_mediane": round(percentile(relatives, 0.5), 4),
        "part_couverture_suffisante": round(
            sum(1 for l in mesurables if l["couverture_suffisante"])
            / len(mesurables)
            if mesurables
            else 0.0,
            4,
        ),
        # --- Document attendu retrouvé par le retrieval (au moins un passage
        # du document attendu figure dans les passages récupérés)
        "documents_attendus_retrouves": len(retrouves),
        "taux_documents_retrouves": round(
            len(retrouves) / len(mesurables) if mesurables else 0.0, 4
        ),
        # --- Couverture CONDITIONNÉE au fait que le document ait été
        # retrouvé : neutralise les échecs de retrieval document-level pour
        # isoler la qualité de couverture de l'evidence une fois le bon
        # document en main. Calculée réellement sur le sous-ensemble
        # `retrouves`, jamais estimée.
        "atteint_conditionne_moyen": round(moyenne(atteints_conditionnes), 4),
        "couverture_relative_conditionnee_moyenne": round(
            moyenne(relatives_conditionnees), 4
        ),
        "couverture_relative_conditionnee_mediane": round(
            percentile(relatives_conditionnees, 0.5), 4
        ),
        # --- Chunking et rang
        "rang_premier_utile_moyen": round(moyenne(rangs), 2),
        "rang_premier_utile_median": round(percentile(rangs, 0.5), 2),
        "part_evidence_atteinte": round(
            len(rangs) / len(mesurables) if mesurables else 0.0, 4
        ),
        "passages_pour_80pct_moyen": round(moyenne(pour_80), 2),
        "recouvrement_ngrammes_moyen": round(
            moyenne([l["recouvrement_ngrammes"] for l in mesurables]), 4
        ),
        "part_page_gold_atteinte": (
            round(sum(1 for v in pages_ok if v) / len(pages_ok), 4)
            if pages_ok
            else None
        ),
        "configuration": {
            "top_k": top_k,
            "limite_candidats": limite_candidats,
            "utiliser_reranker": utiliser_reranker,
            "appliquer_seuil": appliquer_seuil,
            "max_par_document": max_par_document,
            "resolution_document": resolution_document,
        },
    }

    return resume, details


def construire_parseur() -> argparse.ArgumentParser:
    parseur = argparse.ArgumentParser(
        description="Évalue le retrieval au niveau evidence."
    )
    parseur.add_argument("--questions", type=Path, required=True)
    parseur.add_argument("--limite", type=int, default=None)
    parseur.add_argument("--nom", default=None)
    parseur.add_argument(
        "--sortie", type=Path, default=DOSSIER_RAPPORTS / "retrieval_evidence"
    )
    parseur.add_argument("--top-k", type=int, default=None)
    parseur.add_argument("--candidats", type=int, default=None)
    parseur.add_argument("--max-par-document", type=int, default=3)
    parseur.add_argument("--sans-reranker", action="store_true")
    parseur.add_argument("--avec-seuil", action="store_true")
    parseur.add_argument("--seuil", type=float, default=None)
    parseur.add_argument("--sans-resolution-document", action="store_true")
    parseur.add_argument(
        "--seuil-couverture", type=float, default=SEUIL_COUVERTURE
    )
    parseur.add_argument("--ngramme", type=int, default=3)
    parseur.add_argument("--verbose", action="store_true")
    return parseur


def main(argv: list[str] | None = None) -> int:
    arguments = construire_parseur().parse_args(argv)
    configurer_logs(arguments.verbose)

    enregistrements = charger_enregistrements(
        arguments.questions,
        limite=arguments.limite,
        uniquement_answerable=True,
        uniquement_avec_evidence=True,
    )

    if not enregistrements:
        logger.error(
            "Aucune question avec evidence dans %s. "
            "Le dossier extended_qa_info a-t-il été téléchargé ?",
            arguments.questions,
        )
        return 1

    logger.info(
        "Évaluation evidence-level sur %d question(s).", len(enregistrements)
    )

    attendre_client_qdrant()

    try:
        resume, details = evaluer(
            enregistrements,
            top_k=arguments.top_k,
            limite_candidats=arguments.candidats,
            utiliser_reranker=not arguments.sans_reranker,
            appliquer_seuil=arguments.avec_seuil,
            seuil_pertinence=arguments.seuil,
            max_par_document=arguments.max_par_document,
            resolution_document=not arguments.sans_resolution_document,
            seuil_couverture=arguments.seuil_couverture,
            taille_ngramme=arguments.ngramme,
        )
    finally:
        fermer_client()

    resume["jeu"] = str(arguments.questions)
    nom = arguments.nom or f"retrieval_evidence_{horodatage()}"
    ecrire_rapport(arguments.sortie, nom, resume=resume, details=details)

    print()
    print(f"  Questions mesurables      : {resume['mesurables']}")
    print(f"  Plafond moyen (parsing)   : {resume['plafond_moyen']:.4f}")
    print(f"    dont plafond >= 0.90    : {resume['part_plafond_sup_090']:.1%}")
    print(f"  Atteint moyen (retrieval) : {resume['atteint_moyen']:.4f}")
    print(f"  COUVERTURE RELATIVE       : {resume['couverture_relative_moyenne']:.4f}")
    print(
        f"    dont >= {resume['seuil_couverture']:.2f}          : "
        f"{resume['part_couverture_suffisante']:.1%}"
    )
    print(
        f"  Document retrouvé         : "
        f"{resume['taux_documents_retrouves']:.1%} "
        f"({resume['documents_attendus_retrouves']}/{resume['mesurables']})"
    )
    print(
        f"  Couverture relative | doc retrouvé : "
        f"{resume['couverture_relative_conditionnee_moyenne']:.4f}"
    )
    print(f"  Rang 1er passage utile    : {resume['rang_premier_utile_median']}")
    print(f"  Passages pour 80% plafond : {resume['passages_pour_80pct_moyen']}")
    print()
    if resume["plafond_moyen"] < 0.80:
        print(
            "  ⚠ Plafond moyen faible : la perte vient du PARSING, pas du\n"
            "    retrieval. Lis la couverture relative, pas l'atteint brut.\n"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())