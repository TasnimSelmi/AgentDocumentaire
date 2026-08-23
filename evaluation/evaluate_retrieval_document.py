"""
Évaluation du retrieval au niveau document.

Répond à une seule question : le bon document est-il remonté, et à quel rang ?

Le retriever du projet est appelé tel quel, sans réimplémentation, et les
cinq métriques proviennent de `src/rag/evaluate_retrieval.py` — elles ne sont
pas recodées ici, afin que ce harnais et le script existant produisent
exactement les mêmes chiffres sur les mêmes données.

Les questions sans réponse sont exclues : aucun document gold n'existe, et
les inclure ferait chuter mécaniquement le Hit@K sans rien mesurer. Elles
sont évaluées par `evaluate_end_to_end.py`, qui vérifie le refus.

Exemples
--------
    # Smoke test
    python -m evaluation.evaluate_retrieval_document \\
        --questions evaluation/data/finance_esg.jsonl --limite 20

    # Run complet
    python -m evaluation.evaluate_retrieval_document \\
        --questions evaluation/data/finance_esg.jsonl --nom baseline

    # Sans résolution documentaire (recherche purement sémantique)
    python -m evaluation.evaluate_retrieval_document \\
        --questions evaluation/data/finance_esg.jsonl \\
        --sans-resolution-document --nom sans_resolution
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from evaluation.common import (
    DOSSIER_RAPPORTS,
    Enregistrement,
    attendre_client_qdrant,
    charger_enregistrements,
    cle_document,
    configurer_logs,
    ecrire_rapport,
    horodatage,
    moyenne,
    percentile,
)

# --- Métriques du projet : importées, jamais réécrites.
from src.rag.evaluate_retrieval import (
    average_precision_at_k,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from src.rag.retrieval import rechercher_passages
from src.rag.vectorstore import fermer_client

logger = logging.getLogger("evaluation.retrieval_document")

K_PAR_DEFAUT = (1, 3, 5, 10)


# ===========================================================================
# Extraction du classement documentaire
# ===========================================================================


def documents_classes(rapport: Any) -> list[str]:
    """
    Classement des documents, dans l'ordre des passages.

    Volontairement proche de `extraire_documents_recuperes()` du projet, mais
    la clé de comparaison est normalisée (`cle_document`) pour absorber
    l'écart entre le nom gold du benchmark et le nom du fichier indexé.
    """
    classement: list[str] = []
    vus: set[str] = set()

    for passage in rapport.passages:
        cle = cle_document(passage.nom_fichier or passage.source)
        if cle and cle not in vus:
            vus.add(cle)
            classement.append(cle)

    return classement


# ===========================================================================
# Évaluation
# ===========================================================================


def evaluer(
    enregistrements: Sequence[Enregistrement],
    *,
    valeurs_k: Sequence[int] = K_PAR_DEFAUT,
    top_k: int | None = None,
    limite_candidats: int | None = None,
    utiliser_reranker: bool = True,
    appliquer_seuil: bool = False,
    seuil_pertinence: float | None = None,
    max_par_document: int = 3,
    resolution_document: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Exécute le retrieval sur chaque question et agrège les métriques.

    Returns:
        (resume, details)
    """
    valeurs_k = tuple(sorted(set(valeurs_k)))
    k_max = max(valeurs_k)

    # top_k doit couvrir le plus grand k, sinon Hit@10 est plafonné par
    # la configuration du pipeline et non par sa qualité.
    top_k_effectif = top_k if top_k is not None else k_max
    if top_k_effectif < k_max:
        logger.warning(
            "top_k=%d < k_max=%d : les métriques au-delà de %d seront "
            "artificiellement basses.",
            top_k_effectif,
            k_max,
            top_k_effectif,
        )

    details: list[dict[str, Any]] = []
    erreurs = 0

    for index, enregistrement in enumerate(enregistrements, start=1):
        pertinents = {cle_document(d) for d in enregistrement.documents_pertinents}
        pertinents.discard("")

        depart = time.perf_counter()
        ligne: dict[str, Any] = {
            "id": enregistrement.id,
            "question": enregistrement.question,
            "subset": enregistrement.subset,
            "expected_document": enregistrement.expected_document,
        }

        try:
            rapport = rechercher_passages(
                enregistrement.question,
                top_k=top_k_effectif,
                limite_candidats=limite_candidats,
                utiliser_reranker=utiliser_reranker,
                appliquer_seuil=appliquer_seuil,
                seuil_pertinence=seuil_pertinence,
                max_par_document=max_par_document,
                resolution_document=resolution_document,
            )
        except Exception as exc:  # noqa: BLE001 — une erreur ne doit pas tout arrêter
            erreurs += 1
            logger.error("Erreur sur %s : %s", enregistrement.id, exc)
            ligne.update(
                {
                    "erreur": f"{type(exc).__name__}: {exc}",
                    "duree_secondes": round(time.perf_counter() - depart, 3),
                    "documents_recuperes": "",
                    "rang_document_attendu": None,
                    "nombre_passages": 0,
                }
            )
            details.append(ligne)
            continue

        recuperes = documents_classes(rapport)

        rang = None
        for position, document in enumerate(recuperes, start=1):
            if document in pertinents:
                rang = position
                break

        ligne.update(
            {
                "erreur": "",
                "duree_secondes": round(rapport.duree_secondes, 3),
                "documents_recuperes": "|".join(recuperes[:k_max]),
                "rang_document_attendu": rang,
                "nombre_passages": len(rapport.passages),
                "candidats_recuperes": rapport.candidats_recuperes,
                "reranking_utilise": rapport.reranking_utilise,
                "contexte_insuffisant": rapport.contexte_insuffisant,
                "motif_absence": rapport.motif_absence or "",
                # Le périmètre documentaire est déterminant pour interpréter
                # les scores : on le trace systématiquement.
                "perimetre_statut": (
                    rapport.perimetre.statut if rapport.perimetre else ""
                ),
                "perimetre_contraignant": bool(
                    rapport.perimetre and rapport.perimetre.contraignant
                ),
                "perimetre_libelle": (
                    rapport.perimetre.libelle if rapport.perimetre else ""
                ),
            }
        )

        for k in valeurs_k:
            ligne[f"hit@{k}"] = hit_at_k(recuperes, pertinents, k)
            ligne[f"precision@{k}"] = precision_at_k(recuperes, pertinents, k)
            ligne[f"recall@{k}"] = recall_at_k(recuperes, pertinents, k)
            ligne[f"mrr@{k}"] = reciprocal_rank_at_k(recuperes, pertinents, k)
            ligne[f"map@{k}"] = average_precision_at_k(recuperes, pertinents, k)

        details.append(ligne)

        if index % 25 == 0:
            logger.info("… %d/%d questions", index, len(enregistrements))

    # --- Agrégation ---------------------------------------------------------
    evalues = [ligne for ligne in details if not ligne["erreur"]]
    durees = [ligne["duree_secondes"] for ligne in evalues]

    resume: dict[str, Any] = {
        "questions": len(details),
        "evaluees": len(evalues),
        "erreurs": erreurs,
        "taux_erreur": erreurs / len(details) if details else 0.0,
        "duree_moyenne_secondes": round(moyenne(durees), 3),
        "duree_mediane_secondes": round(percentile(durees, 0.5), 3),
        "duree_p90_secondes": round(percentile(durees, 0.9), 3),
        "configuration": {
            "top_k": top_k_effectif,
            "limite_candidats": limite_candidats,
            "utiliser_reranker": utiliser_reranker,
            "appliquer_seuil": appliquer_seuil,
            "seuil_pertinence": seuil_pertinence,
            "max_par_document": max_par_document,
            "resolution_document": resolution_document,
        },
    }

    for k in valeurs_k:
        for metrique in ("hit", "precision", "recall", "mrr", "map"):
            cle = f"{metrique}@{k}"
            resume[cle] = round(
                moyenne([ligne[cle] for ligne in evalues if cle in ligne]), 4
            )

    # Part des requêtes où le périmètre documentaire a contraint la recherche.
    contraints = sum(
        1 for ligne in evalues if ligne.get("perimetre_contraignant")
    )
    resume["part_perimetre_contraignant"] = round(
        contraints / len(evalues) if evalues else 0.0, 4
    )

    return resume, details


# ===========================================================================
# CLI
# ===========================================================================


def construire_parseur() -> argparse.ArgumentParser:
    parseur = argparse.ArgumentParser(
        description="Évalue le retrieval au niveau document."
    )
    parseur.add_argument("--questions", type=Path, required=True)
    parseur.add_argument("--limite", type=int, default=None, help="smoke test")
    parseur.add_argument("--nom", default=None, help="nom du rapport")
    parseur.add_argument(
        "--sortie", type=Path, default=DOSSIER_RAPPORTS / "retrieval_document"
    )
    parseur.add_argument("--k", nargs="+", type=int, default=list(K_PAR_DEFAUT))
    parseur.add_argument("--top-k", type=int, default=None)
    parseur.add_argument("--candidats", type=int, default=None)
    parseur.add_argument("--max-par-document", type=int, default=3)
    parseur.add_argument("--sans-reranker", action="store_true")
    parseur.add_argument("--avec-seuil", action="store_true")
    parseur.add_argument("--seuil", type=float, default=None)
    parseur.add_argument("--sans-resolution-document", action="store_true")
    parseur.add_argument("--verbose", action="store_true")
    return parseur


def main(argv: list[str] | None = None) -> int:
    arguments = construire_parseur().parse_args(argv)
    configurer_logs(arguments.verbose)

    enregistrements = charger_enregistrements(
        arguments.questions,
        limite=arguments.limite,
        uniquement_answerable=True,
    )

    if not enregistrements:
        logger.error("Aucune question répondable dans %s.", arguments.questions)
        return 1

    logger.info(
        "Évaluation document-level sur %d question(s).", len(enregistrements)
    )

    attendre_client_qdrant()

    try:
        resume, details = evaluer(
            enregistrements,
            valeurs_k=arguments.k,
            top_k=arguments.top_k,
            limite_candidats=arguments.candidats,
            utiliser_reranker=not arguments.sans_reranker,
            appliquer_seuil=arguments.avec_seuil,
            seuil_pertinence=arguments.seuil,
            max_par_document=arguments.max_par_document,
            resolution_document=not arguments.sans_resolution_document,
        )
    finally:
        fermer_client()

    resume["jeu"] = str(arguments.questions)
    nom = arguments.nom or f"retrieval_document_{horodatage()}"
    ecrire_rapport(arguments.sortie, nom, resume=resume, details=details)

    print()
    print(f"  Questions évaluées : {resume['evaluees']}  (erreurs : {resume['erreurs']})")
    for k in sorted(set(arguments.k)):
        print(
            f"  Hit@{k:<3} {resume[f'hit@{k}']:.4f}   "
            f"Recall@{k:<3} {resume[f'recall@{k}']:.4f}   "
            f"MRR@{k:<3} {resume[f'mrr@{k}']:.4f}"
        )
    print(f"  Temps moyen        : {resume['duree_moyenne_secondes']:.2f} s")
    print(
        f"  Périmètre contraignant : "
        f"{resume['part_perimetre_contraignant']:.1%} des requêtes"
    )
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())