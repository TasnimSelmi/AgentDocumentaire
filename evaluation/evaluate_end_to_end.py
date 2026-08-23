"""
Évaluation end-to-end avec attribution de la cause d'échec.

C'est le script central de la phase de validation : il ne produit pas
seulement un score, il dit **pourquoi** le système échoue. Un socle qu'on
gèle sans savoir si ses erreurs viennent du retrieval ou du LLM ne peut pas
être amélioré rationnellement.

Chaîne exécutée
---------------
    question -> retrieval -> contexte -> génération -> réponse sourcée

Arbre de classification
-----------------------
    question sans réponse attendue
        refus            -> REFUS_CORRECT
        réponse produite -> FAUX_POSITIF          (hallucination franche)

    question répondable
        refus            -> REFUS_INCORRECT
        evidence couverte au-dessus du seuil ?
            OUI  réponse exacte   -> SUCCES
                                     (ou SUCCES_CITATION_INVALIDE)
                 réponse fausse   -> ECHEC_GENERATION
            NON  réponse exacte   -> CORRECT_SANS_EVIDENCE
                                     à auditer : connaissance paramétrique ?
                 réponse fausse   -> ECHEC_RETRIEVAL

La catégorie CORRECT_SANS_EVIDENCE est délibérément séparée du succès. Une
réponse juste produite sans preuve documentaire viole la règle de grounding
du projet : elle vient probablement de la mémoire du LLM, et le même
mécanisme produira une erreur sur un corpus d'entreprise privé.

Le seuil porte sur la couverture RELATIVE, calibrée sur ce que le parsing
rend atteignable. Voir evaluate_retrieval_evidence.py.

Exemple
-------
    python -m evaluation.evaluate_end_to_end \\
        --questions evaluation/data/finance_esg.jsonl --limite 50 --nom baseline
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import Counter
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
    ensemble_jetons,
    horodatage,
    jetons,
    mesurer_couverture,
    moyenne,
)
from src.rag.generation import ReponseRAG, generer_reponse
from src.rag.vectorstore import fermer_client
from test_rag import TOLERANCE_RELATIVE, comparer_reponse

logger = logging.getLogger("evaluation.end_to_end")

SEUIL_COUVERTURE = 0.70

# Mots outils les plus fréquents en anglais et en français. Ils ne prouvent
# rien sur l'ancrage documentaire : les retenir gonflerait la groundedness
# de toute réponse grammaticalement correcte.
_MOTS_OUTILS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is",
    "are", "was", "were", "be", "been", "it", "its", "this", "that", "these",
    "with", "by", "as", "from", "not", "no", "we", "they", "their", "there",
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "est",
    "sont", "dans", "pour", "par", "sur", "au", "aux", "ce", "cette", "ces",
    "il", "elle", "ils", "elles", "que", "qui", "ne", "pas", "plus",
}

_NOMBRE = re.compile(r"^\d+([.,]\d+)?$")


def jetons_porteurs(texte: str) -> set[str]:
    """
    Jetons dont la présence dans le contexte est réellement informative.

    Retient les nombres (toujours) et les mots hors liste d'outils de plus de
    trois caractères. Cette liste est linguistique, jamais métier : elle ne
    contient aucun terme de finance, d'ESG ou du benchmark.
    """
    retenus: set[str] = set()
    for jeton in jetons(texte):
        if _NOMBRE.match(jeton):
            retenus.add(jeton)
        elif len(jeton) > 3 and jeton not in _MOTS_OUTILS:
            retenus.add(jeton)
    return retenus


def calculer_groundedness(reponse: ReponseRAG) -> float:
    """
    Part des jetons porteurs de la réponse présents dans le contexte cité.

    Le contexte est reconstitué depuis `ReponseRAG.sources`, c'est-à-dire ce
    que le pipeline a effectivement présenté au modèle, et non l'ensemble des
    passages candidats.
    """
    porteurs = jetons_porteurs(reponse.reponse)
    if not porteurs:
        return 1.0  # Réponse sans contenu factuel : rien à ancrer.

    contexte: set[str] = set()
    for source in reponse.sources:
        contexte |= ensemble_jetons(source.extrait)

    if not contexte:
        return 0.0

    return len(porteurs & contexte) / len(porteurs)


def a_refuse(reponse: ReponseRAG) -> bool:
    """
    Détecte un refus sans recourir à une liste de formules.

    Le pipeline court-circuite le LLM quand le contexte est insuffisant : le
    refus se lit donc dans l'état de l'objet, pas dans le texte. C'est plus
    robuste qu'un appariement de chaînes, qui casserait au moindre
    changement de formulation ou de langue de sortie.
    """
    return (not reponse.contexte_suffisant) or (not reponse.sources)


def juger_avec_llm(reponse: ReponseRAG, llm: Any) -> tuple[float | None, str]:
    """
    Juge LLM optionnel : la réponse est-elle soutenue par le contexte ?

    Non déterministe, donc explicitement hors des métriques par défaut.
    Réutilise le LLM du projet pour ne pas introduire de dépendance.
    """
    if not reponse.sources:
        return None, "aucune source"

    contexte = "\n\n".join(
        f"[{source.citation}] {source.extrait}" for source in reponse.sources
    )
    systeme = (
        "Tu évalues si une réponse est entièrement soutenue par un contexte.\n"
        "Réponds uniquement par un entier de 0 à 100, sans aucun autre texte.\n"
        "100 = chaque affirmation est soutenue par le contexte.\n"
        "0 = la réponse contredit le contexte ou l'invente."
    )
    utilisateur = (
        f"CONTEXTE :\n{contexte}\n\n"
        f"QUESTION :\n{reponse.question}\n\n"
        f"RÉPONSE :\n{reponse.reponse}\n\n"
        "Score :"
    )

    try:
        from src.llm.common import invoquer_llm

        brut = invoquer_llm(llm, systeme=systeme, utilisateur=utilisateur)
        trouve = re.search(r"\d{1,3}", str(brut))
        if not trouve:
            return None, f"score illisible : {str(brut)[:60]}"
        return min(100, int(trouve.group())) / 100.0, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"

# Catégories exhaustives et mutuellement exclusives.
SUCCES = "SUCCES"
SUCCES_CITATION_INVALIDE = "SUCCES_CITATION_INVALIDE"
ECHEC_RETRIEVAL = "ECHEC_RETRIEVAL"
ECHEC_GENERATION = "ECHEC_GENERATION"
CORRECT_SANS_EVIDENCE = "CORRECT_SANS_EVIDENCE"
REFUS_CORRECT = "REFUS_CORRECT"
REFUS_INCORRECT = "REFUS_INCORRECT"
FAUX_POSITIF = "FAUX_POSITIF"
ERREUR_TECHNIQUE = "ERREUR_TECHNIQUE"

CATEGORIES = (
    SUCCES,
    SUCCES_CITATION_INVALIDE,
    ECHEC_RETRIEVAL,
    ECHEC_GENERATION,
    CORRECT_SANS_EVIDENCE,
    REFUS_CORRECT,
    REFUS_INCORRECT,
    FAUX_POSITIF,
    ERREUR_TECHNIQUE,
)


def classifier(
    *,
    answerable: bool,
    refus: bool,
    exacte: bool | None,
    evidence_couverte: bool | None,
    citations_valides: bool,
) -> str:
    """
    Attribue une catégorie unique à un cas.

    `evidence_couverte` vaut None quand aucune evidence gold n'existe : on ne
    peut alors pas distinguer échec de retrieval et échec de génération, et le
    cas est rangé selon l'exactitude seule.
    """
    if not answerable:
        return REFUS_CORRECT if refus else FAUX_POSITIF

    if refus:
        return REFUS_INCORRECT

    if evidence_couverte is None:
        if exacte:
            return SUCCES if citations_valides else SUCCES_CITATION_INVALIDE
        return ECHEC_GENERATION

    if evidence_couverte:
        if exacte:
            return SUCCES if citations_valides else SUCCES_CITATION_INVALIDE
        return ECHEC_GENERATION

    return CORRECT_SANS_EVIDENCE if exacte else ECHEC_RETRIEVAL


def evaluer(
    enregistrements: Sequence[Enregistrement],
    *,
    tolerance: float = TOLERANCE_RELATIVE,
    seuil_couverture: float = SEUIL_COUVERTURE,
    llm_judge: bool = False,
    **options_pipeline: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Exécute la chaîne complète et classifie chaque cas."""
    logger.info("Lecture du corpus indexé pour le calcul des plafonds…")
    textes_par_document = charger_textes_par_document()

    llm = None
    if llm_judge:
        from src.llm.factory import construire_llm

        llm = construire_llm()
        logger.info("Juge LLM activé (métrique non déterministe).")

    details: list[dict[str, Any]] = []

    for index, enregistrement in enumerate(enregistrements, start=1):
        ligne: dict[str, Any] = {
            "id": enregistrement.id,
            "question": enregistrement.question,
            "subset": enregistrement.subset,
            "answerable": enregistrement.answerable,
            "expected_answer": enregistrement.expected_answer,
            "expected_document": enregistrement.expected_document,
        }

        try:
            reponse = generer_reponse(enregistrement.question, **options_pipeline)
        except Exception as exc:  # noqa: BLE001
            logger.error("Erreur sur %s : %s", enregistrement.id, exc)
            ligne.update(
                {
                    "categorie": ERREUR_TECHNIQUE,
                    "erreur": f"{type(exc).__name__}: {exc}",
                }
            )
            details.append(ligne)
            continue

        refus = a_refuse(reponse)

        # --- Exactitude
        exacte: bool | None = None
        detail_exactitude = ""
        if enregistrement.answerable and enregistrement.expected_answer:
            candidats = [
                enregistrement.expected_answer,
                *enregistrement.answer_variants,
            ]
            resultats = [
                comparer_reponse(reponse.reponse, attendu, tolerance=tolerance)
                for attendu in candidats
            ]
            exacte = any(ok for ok, _ in resultats)
            detail_exactitude = resultats[0][1]

        # --- Couverture d'evidence, sur les passages du document attendu.
        evidence_couverte: bool | None = None
        couverture_relative: float | None = None
        rang_premier_utile: int | None = None

        if enregistrement.a_une_evidence and reponse.recherche is not None:
            cle_attendue = cle_document(enregistrement.expected_document or "")
            textes_document = textes_par_document.get(cle_attendue, [])
            passages = [
                p
                for p in reponse.recherche.passages
                if cle_document(p.nom_fichier or p.source) == cle_attendue
            ]
            couverture = mesurer_couverture(
                enregistrement.evidence_text,
                passages,
                textes_du_document=textes_document,
            )
            couverture_relative = round(couverture.couverture_relative, 4)
            rang_premier_utile = couverture.rang_premier_utile
            evidence_couverte = couverture.couverture_relative >= seuil_couverture

        categorie = classifier(
            answerable=enregistrement.answerable,
            refus=refus,
            exacte=exacte,
            evidence_couverte=evidence_couverte,
            citations_valides=reponse.citations_valides,
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
                "categorie": categorie,
                "erreur": "",
                "reponse": reponse.reponse,
                "refus": refus,
                "exactitude": exacte,
                "detail_exactitude": detail_exactitude,
                "evidence_couverte": evidence_couverte,
                "couverture_relative": couverture_relative,
                "rang_premier_utile": rang_premier_utile,
                "groundedness": round(calculer_groundedness(reponse), 4),
                "contexte_suffisant": reponse.contexte_suffisant,
                "citations_valides": reponse.citations_valides,
                "citations_reparees": reponse.citations_reparees,
                "citations_hors_perimetre": len(reponse.citations_hors_perimetre),
                "nombre_sources": len(reponse.sources),
                "documents_cites": "|".join(documents_cites),
                "bon_document_cite": bon_document_cite,
                "perimetre_statut": (
                    reponse.perimetre.statut if reponse.perimetre else ""
                ),
                "duree_secondes": round(reponse.duree_secondes, 3),
            }
        )

        if llm_judge:
            score, motif = juger_avec_llm(reponse, llm)
            ligne["fidelite_llm"] = score
            ligne["fidelite_llm_motif"] = motif

        details.append(ligne)

        if index % 10 == 0:
            logger.info(
                "… %d/%d questions", index, len(enregistrements)
            )

    # --- Agrégation ---------------------------------------------------------
    compteur = Counter(l["categorie"] for l in details)
    total = len(details)
    aboutis = total - compteur[ERREUR_TECHNIQUE]

    resume: dict[str, Any] = {
        "questions": total,
        "aboutis": aboutis,
        "seuil_couverture": seuil_couverture,
        "repartition": {c: compteur.get(c, 0) for c in CATEGORIES},
        "repartition_pct": {
            c: round(compteur.get(c, 0) / aboutis, 4) if aboutis else 0.0
            for c in CATEGORIES
        },
    }

    reussites = compteur[SUCCES] + compteur[SUCCES_CITATION_INVALIDE]
    resume["taux_succes"] = round(reussites / aboutis, 4) if aboutis else 0.0

    # Part des échecs imputables au retrieval plutôt qu'à la génération :
    # c'est le chiffre qui oriente la suite du travail.
    echecs = compteur[ECHEC_RETRIEVAL] + compteur[ECHEC_GENERATION]
    resume["part_echecs_retrieval"] = (
        round(compteur[ECHEC_RETRIEVAL] / echecs, 4) if echecs else 0.0
    )
    resume["part_echecs_generation"] = (
        round(compteur[ECHEC_GENERATION] / echecs, 4) if echecs else 0.0
    )

    evalues = [l for l in details if not l.get("erreur")]
    resume["groundedness_moyenne"] = round(
        moyenne([l["groundedness"] for l in evalues if l.get("nombre_sources")]), 4
    )
    resume["duree_moyenne_secondes"] = round(
        moyenne([l["duree_secondes"] for l in evalues]), 2
    )

    # --- Citations : correction des sources citées, sur les cas où une
    # comparaison a du sens (document attendu connu, au moins une citation).
    avec_document_attendu = [
        l for l in evalues if l.get("bon_document_cite") is not None
    ]
    resume["citation_correcte"] = (
        round(
            sum(1 for l in avec_document_attendu if l["bon_document_cite"])
            / len(avec_document_attendu),
            4,
        )
        if avec_document_attendu
        else None
    )
    avec_sources = [l for l in evalues if l.get("nombre_sources")]
    resume["citations_reparees_taux"] = (
        round(
            sum(1 for l in avec_sources if l["citations_reparees"]) / len(avec_sources),
            4,
        )
        if avec_sources
        else 0.0
    )

    if llm_judge:
        scores = [l["fidelite_llm"] for l in evalues if l.get("fidelite_llm") is not None]
        resume["fidelite_llm_moyenne"] = round(moyenne(scores), 4)
        resume["fidelite_llm_evaluees"] = len(scores)

    # --- Unanswerable : le système doit refuser, jamais halluciner.
    non_repondables = [l for l in evalues if not l["answerable"]]
    total_unanswerable = len(non_repondables)
    refus_corrects = sum(1 for l in non_repondables if l["categorie"] == REFUS_CORRECT)
    hallucinations = sum(1 for l in non_repondables if l["categorie"] == FAUX_POSITIF)
    resume["total_unanswerable"] = total_unanswerable
    resume["refus_corrects"] = refus_corrects
    resume["hallucinations"] = hallucinations
    resume["correct_refusal_rate"] = (
        round(refus_corrects / total_unanswerable, 4) if total_unanswerable else 0.0
    )
    resume["hallucination_rate"] = (
        round(hallucinations / total_unanswerable, 4) if total_unanswerable else 0.0
    )

    # --- Answerable : un refus est ici une erreur (faux refus), pas une vertu.
    repondables = [l for l in evalues if l["answerable"]]
    total_answerable = len(repondables)
    false_refusals = sum(1 for l in repondables if l["categorie"] == REFUS_INCORRECT)
    resume["total_answerable"] = total_answerable
    resume["false_refusals"] = false_refusals
    resume["false_refusal_rate"] = (
        round(false_refusals / total_answerable, 4) if total_answerable else 0.0
    )

    return resume, details


def construire_parseur() -> argparse.ArgumentParser:
    parseur = argparse.ArgumentParser(
        description="Évaluation end-to-end avec attribution des échecs."
    )
    parseur.add_argument("--questions", type=Path, required=True)
    parseur.add_argument("--limite", type=int, default=None)
    parseur.add_argument("--nom", default=None)
    parseur.add_argument(
        "--sortie", type=Path, default=DOSSIER_RAPPORTS / "end_to_end"
    )
    parseur.add_argument("--top-k", type=int, default=None)
    parseur.add_argument("--candidats", type=int, default=None)
    parseur.add_argument("--max-par-document", type=int, default=3)
    parseur.add_argument("--contexte", type=int, default=16_000)
    parseur.add_argument("--sans-reranker", action="store_true")
    parseur.add_argument("--avec-seuil", action="store_true")
    parseur.add_argument("--seuil", type=float, default=None)
    parseur.add_argument("--sans-resolution-document", action="store_true")
    parseur.add_argument("--tolerance", type=float, default=TOLERANCE_RELATIVE)
    parseur.add_argument(
        "--seuil-couverture", type=float, default=SEUIL_COUVERTURE
    )
    parseur.add_argument(
        "--llm-judge",
        action="store_true",
        help="active le juge LLM (non déterministe, hors métriques de base)",
    )
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

    logger.info("Évaluation end-to-end sur %d question(s).", len(enregistrements))

    attendre_client_qdrant()

    try:
        resume, details = evaluer(
            enregistrements,
            tolerance=arguments.tolerance,
            seuil_couverture=arguments.seuil_couverture,
            llm_judge=arguments.llm_judge,
            top_k=arguments.top_k,
            limite_candidats=arguments.candidats,
            utiliser_reranker=not arguments.sans_reranker,
            appliquer_seuil=arguments.avec_seuil,
            seuil_pertinence=arguments.seuil,
            max_par_document=arguments.max_par_document,
            limite_contexte_caracteres=arguments.contexte,
            resolution_document=not arguments.sans_resolution_document,
        )
    finally:
        fermer_client()

    resume["jeu"] = str(arguments.questions)
    nom = arguments.nom or f"end_to_end_{horodatage()}"
    ecrire_rapport(arguments.sortie, nom, resume=resume, details=details)

    print()
    print("  RÉPARTITION DES CAS")
    for categorie in CATEGORIES:
        nombre = resume["repartition"][categorie]
        if nombre:
            print(
                f"    {categorie:<26} {nombre:4d}  "
                f"({resume['repartition_pct'][categorie]:.1%})"
            )
    print()
    print(f"  Taux de succès          : {resume['taux_succes']:.1%}")
    print(f"  Échecs dus au retrieval : {resume['part_echecs_retrieval']:.1%}")
    print(f"  Échecs dus au LLM       : {resume['part_echecs_generation']:.1%}")
    print()
    print(
        f"  Unanswerable ({resume['total_unanswerable']:>3d}) : "
        f"refus correct {resume['correct_refusal_rate']:.1%}  |  "
        f"hallucination {resume['hallucination_rate']:.1%}"
    )
    print(
        f"  Answerable   ({resume['total_answerable']:>3d}) : "
        f"faux refus {resume['false_refusal_rate']:.1%}"
    )
    print()
    if resume["repartition"][CORRECT_SANS_EVIDENCE]:
        print(
            f"  ⚠ {resume['repartition'][CORRECT_SANS_EVIDENCE]} réponse(s) "
            "correcte(s) SANS evidence récupérée.\n"
            "    À auditer : connaissance paramétrique du LLM plutôt que\n"
            "    preuve documentaire. Voir analyze_failures.py.\n"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())