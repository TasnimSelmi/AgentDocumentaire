"""
Évaluation de la couche agentique (LangGraph).

Ne relance pas le RAG "brut" — voir `evaluate_end_to_end.py` pour ça — mais
mesure spécifiquement le comportement du graphe lui-même :
rechercher -> évaluer_preuves -> (répondre | reformuler -> rechercher).

Question à laquelle ce script répond : la boucle agentique (budget de
tentatives, reformulation automatique) apporte-t-elle quelque chose par
rapport à un simple appel RAG, et à quel coût en latence ?

Exemple
-------
    python -m evaluation.evaluate_agent \\
        --questions evaluation/data/finance_esg.jsonl --limite 20 --nom baseline
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
from evaluation.evaluate_end_to_end import calculer_groundedness
from src.agent.graph import construire_graphe
from src.agent.graph_state import EtatGraphe
from src.agent.session import construire_session
from src.rag.vectorstore import fermer_client
from test_rag import TOLERANCE_RELATIVE, comparer_reponse

logger = logging.getLogger("evaluation.agent")


def evaluer_agent(
    enregistrements: Sequence[Enregistrement],
    *,
    max_tentatives: int | None = None,
    tolerance: float = TOLERANCE_RELATIVE,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Exécute le graphe agentique sur chaque question et journalise son comportement."""
    graphe = construire_graphe()
    details: list[dict[str, Any]] = []

    for index, enregistrement in enumerate(enregistrements, start=1):
        debut = time.perf_counter()
        ligne: dict[str, Any] = {
            "id": enregistrement.id,
            "question": enregistrement.question,
            "subset": enregistrement.subset,
            "answerable": enregistrement.answerable,
        }

        try:
            session = construire_session(
                enregistrement.question, max_tentatives=max_tentatives
            )
            limite_recursion = max(25, session.etat.max_tentatives * 3 + 5)
            resultat = graphe.invoke(
                EtatGraphe(session=session),
                config={"recursion_limit": limite_recursion},
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Erreur sur %s : %s", enregistrement.id, exc)
            ligne.update({"erreur": f"{type(exc).__name__}: {exc}"})
            details.append(ligne)
            continue

        duree = round(time.perf_counter() - debut, 3)
        session = resultat["session"]
        reponse = resultat["reponse"]

        etapes_pertinence = [
            e for e in session.etat.trace if e.nom == "evaluation_pertinence"
        ]
        etapes_suffisance = [
            e for e in session.etat.trace if e.nom == "evaluation_suffisance"
        ]
        appels_reformulation = sum(
            1 for e in session.etat.trace if e.nom in ("reformulation", "reformulation_echec")
        )
        a_reformule = session.etat.a_ete_reformulee
        trace_reponse = session.etat.trace[-1] if session.etat.trace else None
        genere_par_llm = bool(
            trace_reponse is not None
            and trace_reponse.nom == "reponse"
            and trace_reponse.donnees.get("genere_par_llm")
        )

        score_premier = (
            etapes_pertinence[0].donnees.get("score_pertinence_maximal")
            if etapes_pertinence
            else None
        )
        score_dernier = (
            etapes_pertinence[-1].donnees.get("score_pertinence_maximal")
            if etapes_pertinence
            else None
        )
        # Résolu au premier essai : niveau 1 ET niveau 2 (s'il a été atteint)
        # validés dès la première évaluation, sans aucune reformulation.
        premier_pertinent = bool(etapes_pertinence) and bool(
            etapes_pertinence[0].donnees.get("pertinent")
        )
        premier_suffisant_niveau2 = bool(etapes_suffisance) and bool(
            etapes_suffisance[0].donnees.get("suffisant")
        )
        resolu_premier_retrieval = (
            session.etat.tentatives == 1
            and premier_pertinent
            and premier_suffisant_niveau2
        )

        refus = (not reponse.contexte_suffisant) or (not reponse.sources)

        exacte: bool | None = None
        if enregistrement.answerable and enregistrement.expected_answer:
            candidats = [enregistrement.expected_answer, *enregistrement.answer_variants]
            resultats = [
                comparer_reponse(reponse.reponse, attendu, tolerance=tolerance)
                for attendu in candidats
            ]
            exacte = any(ok for ok, _ in resultats)

        reformulation_a_ameliore_score: bool | None = None
        if a_reformule and score_premier is not None and score_dernier is not None:
            reformulation_a_ameliore_score = score_dernier > score_premier

        groundedness = (
            round(calculer_groundedness(reponse), 4) if reponse.sources else None
        )
        documents_cites = sorted(
            {cle_document(s.nom_fichier or s.source) for s in reponse.sources}
        )
        bon_document_cite = (
            cle_document(enregistrement.expected_document or "") in documents_cites
            if enregistrement.expected_document
            else None
        )

        ligne.update(
            {
                "erreur": "",
                "tentatives": session.etat.tentatives,
                "max_tentatives": session.etat.max_tentatives,
                "budget_epuise": not session.etat.peut_reessayer,
                "reformulations": appels_reformulation,
                "a_reformule": a_reformule,
                "resolu_premier_retrieval": resolu_premier_retrieval,
                "score_pertinence_premiere_tentative": score_premier,
                "score_pertinence_derniere_tentative": score_dernier,
                "reformulation_a_ameliore_score": reformulation_a_ameliore_score,
                "refus": refus,
                "groundedness": groundedness,
                "citations_valides": reponse.citations_valides,
                "bon_document_cite": bon_document_cite,
                "exactitude": exacte,
                "nombre_sources": len(reponse.sources),
                "duree_secondes": duree,
                # --- Coût, décomposé par type d'appel LLM/recherche ---------
                "recherches": session.etat.tentatives,
                "appels_llm_reformulation": appels_reformulation,
                "appels_llm_suffisance": len(etapes_suffisance),
                "appels_llm_generation": 1 if genere_par_llm else 0,
            }
        )
        details.append(ligne)

        if index % 10 == 0:
            logger.info("… %d/%d questions", index, len(enregistrements))

    # --- Agrégation ---------------------------------------------------------
    evalues = [l for l in details if not l.get("erreur")]
    total = len(details)
    aboutis = len(evalues)

    resume: dict[str, Any] = {
        "questions": total,
        "aboutis": aboutis,
        "tentatives_moyennes": round(moyenne([l["tentatives"] for l in evalues]), 3),
        "taux_resolu_premier_retrieval": (
            round(sum(1 for l in evalues if l["resolu_premier_retrieval"]) / aboutis, 4)
            if aboutis
            else 0.0
        ),
        "taux_reformulation": (
            round(sum(1 for l in evalues if l["a_reformule"]) / aboutis, 4)
            if aboutis
            else 0.0
        ),
        "reformulations_totales": sum(l["reformulations"] for l in evalues),
    }

    reformulees = [l for l in evalues if l["a_reformule"]]
    resume["nombre_reformulees"] = len(reformulees)
    mesurables_amelioration = [
        l for l in reformulees if l["reformulation_a_ameliore_score"] is not None
    ]
    resume["taux_reformulation_ameliore_score"] = (
        round(
            sum(1 for l in mesurables_amelioration if l["reformulation_a_ameliore_score"])
            / len(mesurables_amelioration),
            4,
        )
        if mesurables_amelioration
        else None
    )
    reformulees_avec_exactitude = [l for l in reformulees if l["exactitude"] is not None]
    resume["taux_correct_apres_reformulation"] = (
        round(
            sum(1 for l in reformulees_avec_exactitude if l["exactitude"])
            / len(reformulees_avec_exactitude),
            4,
        )
        if reformulees_avec_exactitude
        else None
    )

    non_repondables = [l for l in evalues if not l["answerable"]]
    repondables = [l for l in evalues if l["answerable"]]
    resume["total_unanswerable"] = len(non_repondables)
    resume["taux_refus_corrects"] = (
        round(sum(1 for l in non_repondables if l["refus"]) / len(non_repondables), 4)
        if non_repondables
        else 0.0
    )
    resume["total_answerable"] = len(repondables)
    resume["taux_faux_refus"] = (
        round(sum(1 for l in repondables if l["refus"]) / len(repondables), 4)
        if repondables
        else 0.0
    )
    resume["taux_exactitude_answerable"] = (
        round(
            sum(1 for l in repondables if l["exactitude"]) / len(repondables),
            4,
        )
        if repondables
        else None
    )

    # --- Groundedness et citations, sur les cas answerable avec sources ----
    repondables_avec_sources = [l for l in repondables if l["groundedness"] is not None]
    resume["groundedness_moyenne_answerable"] = (
        round(moyenne([l["groundedness"] for l in repondables_avec_sources]), 4)
        if repondables_avec_sources
        else None
    )
    resume["taux_citations_valides_answerable"] = (
        round(
            sum(1 for l in repondables if l["citations_valides"]) / len(repondables),
            4,
        )
        if repondables
        else None
    )
    avec_document_attendu = [
        l for l in repondables if l["bon_document_cite"] is not None
    ]
    resume["citation_correcte_answerable"] = (
        round(
            sum(1 for l in avec_document_attendu if l["bon_document_cite"])
            / len(avec_document_attendu),
            4,
        )
        if avec_document_attendu
        else None
    )

    resume["budget_epuise_nombre"] = sum(1 for l in evalues if l["budget_epuise"])
    cas_budget_epuise = [l for l in evalues if l["budget_epuise"]]
    resume["duree_moyenne_budget_epuise_secondes"] = (
        round(moyenne([l["duree_secondes"] for l in cas_budget_epuise]), 3)
        if cas_budget_epuise
        else None
    )

    durees = [l["duree_secondes"] for l in evalues]
    resume["duree_moyenne_secondes"] = round(moyenne(durees), 3)
    resume["duree_p50_secondes"] = round(percentile(durees, 0.5), 3)
    resume["duree_p95_secondes"] = round(percentile(durees, 0.95), 3)

    # --- Coût : moyennes par question, décomposées par type d'appel --------
    resume["recherches_moyennes_par_question"] = round(
        moyenne([l["recherches"] for l in evalues]), 3
    )
    resume["appels_llm_reformulation_moyens"] = round(
        moyenne([l["appels_llm_reformulation"] for l in evalues]), 3
    )
    resume["appels_llm_suffisance_moyens"] = round(
        moyenne([l["appels_llm_suffisance"] for l in evalues]), 3
    )
    resume["appels_llm_generation_moyens"] = round(
        moyenne([l["appels_llm_generation"] for l in evalues]), 3
    )
    resume["appels_llm_total_moyens"] = round(
        moyenne(
            [
                l["appels_llm_reformulation"] + l["appels_llm_suffisance"] + l["appels_llm_generation"]
                for l in evalues
            ]
        ),
        3,
    )

    return resume, details


def construire_parseur() -> argparse.ArgumentParser:
    parseur = argparse.ArgumentParser(
        description="Évaluation de la couche agentique (LangGraph)."
    )
    parseur.add_argument("--questions", type=Path, required=True)
    parseur.add_argument("--limite", type=int, default=None)
    parseur.add_argument("--nom", default=None)
    parseur.add_argument("--sortie", type=Path, default=DOSSIER_RAPPORTS / "agent")
    parseur.add_argument(
        "--max-tentatives",
        type=int,
        default=None,
        help="borne du budget de tentatives (défaut : config/default.yaml -> agent.max_iterations)",
    )
    parseur.add_argument("--tolerance", type=float, default=TOLERANCE_RELATIVE)
    parseur.add_argument("--verbose", action="store_true")
    return parseur


def main(argv: list[str] | None = None) -> int:
    arguments = construire_parseur().parse_args(argv)
    configurer_logs(arguments.verbose)

    enregistrements = charger_enregistrements(
        arguments.questions, limite=arguments.limite
    )
    if not enregistrements:
        logger.error("Aucune question dans %s.", arguments.questions)
        return 1

    logger.info("Évaluation agent sur %d question(s).", len(enregistrements))

    attendre_client_qdrant()

    try:
        resume, details = evaluer_agent(
            enregistrements,
            max_tentatives=arguments.max_tentatives,
            tolerance=arguments.tolerance,
        )
    finally:
        fermer_client()

    resume["jeu"] = str(arguments.questions)
    nom = arguments.nom or f"agent_{horodatage()}"
    ecrire_rapport(arguments.sortie, nom, resume=resume, details=details)

    print()
    print("  COUCHE AGENTIQUE")
    print(f"  Tentatives moyennes            : {resume['tentatives_moyennes']}")
    print(f"  Résolu au 1er retrieval        : {resume['taux_resolu_premier_retrieval']:.1%}")
    print(f"  Taux de reformulation          : {resume['taux_reformulation']:.1%}")
    if resume["taux_reformulation_ameliore_score"] is not None:
        print(
            "  Reformulation améliore score   : "
            f"{resume['taux_reformulation_ameliore_score']:.1%} "
            f"({resume['nombre_reformulees']} cas reformulés)"
        )
    if resume["taux_correct_apres_reformulation"] is not None:
        print(
            f"  Correct après reformulation    : {resume['taux_correct_apres_reformulation']:.1%}"
        )
    print(
        f"  Refus corrects (unanswerable)  : {resume['taux_refus_corrects']:.1%} "
        f"(n={resume['total_unanswerable']})"
    )
    print(
        f"  Faux refus (answerable)        : {resume['taux_faux_refus']:.1%} "
        f"(n={resume['total_answerable']})"
    )
    if resume["taux_exactitude_answerable"] is not None:
        print(f"  Exactitude (answerable)        : {resume['taux_exactitude_answerable']:.1%}")
    if resume["groundedness_moyenne_answerable"] is not None:
        print(f"  Groundedness (answerable)      : {resume['groundedness_moyenne_answerable']}")
    if resume["citation_correcte_answerable"] is not None:
        print(f"  Citation correcte (answerable) : {resume['citation_correcte_answerable']:.1%}")
    print(f"  Budget épuisé                  : {resume['budget_epuise_nombre']} cas")
    print(
        f"  Latence moyenne / p50 / p95 (s): {resume['duree_moyenne_secondes']} / "
        f"{resume['duree_p50_secondes']} / {resume['duree_p95_secondes']}"
    )
    print()
    print("  COÛT (moyennes par question)")
    print(f"  Recherches                     : {resume['recherches_moyennes_par_question']}")
    print(f"  Appels LLM reformulation       : {resume['appels_llm_reformulation_moyens']}")
    print(f"  Appels LLM suffisance          : {resume['appels_llm_suffisance_moyens']}")
    print(f"  Appels LLM génération          : {resume['appels_llm_generation_moyens']}")
    print(f"  Appels LLM total               : {resume['appels_llm_total_moyens']}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
