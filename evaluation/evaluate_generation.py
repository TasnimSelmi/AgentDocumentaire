"""
Évaluation de la génération.

Mesure la qualité de la réponse produite, en séparant autant que possible ce
qui relève du LLM de ce qui relève du retrieval. La distinction complète est
faite par `evaluate_end_to_end.py` ; ici, on mesure la production elle-même.

Métriques déterministes (par défaut, reproductibles, sans LLM juge)
-------------------------------------------------------------------
    exactitude          `comparer_reponse()` de test_rag.py, avec la
                        tolérance numérique relative du projet. Importée,
                        pas réécrite.
    exact_match         égalité stricte après normalisation légère.
    groundedness        part des jetons « porteurs » de la réponse
                        (nombres et mots rares) effectivement présents dans
                        le contexte récupéré. Un mot outil comme « le » ne
                        prouve rien ; un nombre absent du contexte est en
                        revanche un signal d'hallucination fort.
    citations_valides   `ReponseRAG.citations_valides`, calculé par le
                        pipeline lui-même.
    citations_hors_perimetre  déjà exposé par le pipeline.
    refus               refus détecté via `contexte_suffisant` et l'absence
                        de sources, sans liste de formules codée en dur.

Métrique optionnelle (--llm-judge)
----------------------------------
    fidelite_llm        un juge LLM évalue si la réponse est entièrement
                        soutenue par le contexte. Désactivé par défaut :
                        non déterministe, donc non reproductible d'un run à
                        l'autre. Utilise le LLM du projet, pas un autre.

Exemple
-------
    python -m evaluation.evaluate_generation \\
        --questions evaluation/data/uda_400.jsonl --limite 50 --nom baseline
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any, Sequence

from evaluation.common import (
    DOSSIER_RAPPORTS,
    Enregistrement,
    charger_enregistrements,
    cle_document,
    configurer_logs,
    ecrire_rapport,
    ensemble_jetons,
    horodatage,
    jetons,
    moyenne,
    normaliser_texte,
)
from src.rag.generation import ReponseRAG, generer_reponse
from src.rag.vectorstore import fermer_client

# Comparaison à la vérité terrain : fonction du projet, importée telle quelle.
from test_rag import TOLERANCE_RELATIVE, comparer_reponse

logger = logging.getLogger("evaluation.generation")

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

    Non déterministe, donc explicitement hors du jeu de métriques par défaut.
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


def evaluer(
    enregistrements: Sequence[Enregistrement],
    *,
    tolerance: float = TOLERANCE_RELATIVE,
    llm_judge: bool = False,
    **options_pipeline: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Génère une réponse par question et mesure sa qualité."""
    llm = None
    if llm_judge:
        from src.llm.factory import construire_llm

        llm = construire_llm()
        logger.info("Juge LLM activé (métrique non déterministe).")

    details: list[dict[str, Any]] = []
    erreurs = 0

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
            erreurs += 1
            logger.error("Erreur sur %s : %s", enregistrement.id, exc)
            ligne["erreur"] = f"{type(exc).__name__}: {exc}"
            details.append(ligne)
            continue

        refus = a_refuse(reponse)

        # --- Exactitude : uniquement sur les questions répondables.
        exacte: bool | None = None
        detail_exactitude = ""
        exact_match: bool | None = None

        if enregistrement.answerable and enregistrement.expected_answer:
            candidats = [enregistrement.expected_answer, *enregistrement.answer_variants]
            resultats = [
                comparer_reponse(reponse.reponse, attendu, tolerance=tolerance)
                for attendu in candidats
            ]
            exacte = any(ok for ok, _ in resultats)
            detail_exactitude = resultats[0][1]
            exact_match = any(
                normaliser_texte(attendu).strip() in normaliser_texte(reponse.reponse)
                for attendu in candidats
                if str(attendu).strip()
            )

        # --- Refus : correct sur une question sans réponse, incorrect sinon.
        refus_correct = refus if not enregistrement.answerable else None
        refus_incorrect = refus if enregistrement.answerable else None

        groundedness = calculer_groundedness(reponse)

        documents_cites = sorted(
            {cle_document(source.nom_fichier or source.source) for source in reponse.sources}
        )
        bon_document_cite = (
            cle_document(enregistrement.expected_document or "") in documents_cites
            if enregistrement.expected_document
            else None
        )

        ligne.update(
            {
                "erreur": "",
                "reponse": reponse.reponse,
                "refus": refus,
                "refus_correct": refus_correct,
                "refus_incorrect": refus_incorrect,
                "exactitude": exacte,
                "exact_match": exact_match,
                "detail_exactitude": detail_exactitude,
                "groundedness": round(groundedness, 4),
                "contexte_suffisant": reponse.contexte_suffisant,
                "citations_valides": reponse.citations_valides,
                "citations_reparees": reponse.citations_reparees,
                "citations_hors_perimetre": len(reponse.citations_hors_perimetre),
                "nombre_sources": len(reponse.sources),
                "documents_cites": "|".join(documents_cites),
                "bon_document_cite": bon_document_cite,
                "profil_domaine": reponse.profil_domaine or "",
                "avertissements": " ; ".join(reponse.avertissements),
                "duree_secondes": round(reponse.duree_secondes, 3),
            }
        )

        if llm_judge:
            score, motif = juger_avec_llm(reponse, llm)
            ligne["fidelite_llm"] = score
            ligne["fidelite_llm_motif"] = motif

        details.append(ligne)

        if index % 10 == 0:
            logger.info("… %d/%d questions", index, len(enregistrements))

    # --- Agrégation ---------------------------------------------------------
    evalues = [l for l in details if not l.get("erreur")]
    repondables = [l for l in evalues if l["answerable"]]
    sans_reponse = [l for l in evalues if not l["answerable"]]
    avec_exactitude = [l for l in repondables if l["exactitude"] is not None]
    avec_sources = [l for l in evalues if l["nombre_sources"] > 0]

    def part(sous_ensemble: Sequence[dict[str, Any]], cle: str) -> float:
        valeurs = [l[cle] for l in sous_ensemble if l.get(cle) is not None]
        return round(sum(1 for v in valeurs if v) / len(valeurs), 4) if valeurs else 0.0

    resume: dict[str, Any] = {
        "questions": len(details),
        "evaluees": len(evalues),
        "erreurs": erreurs,
        "repondables": len(repondables),
        "sans_reponse": len(sans_reponse),
        # --- Exactitude
        "exactitude": part(avec_exactitude, "exactitude"),
        "exact_match": part(avec_exactitude, "exact_match"),
        # --- Ancrage
        "groundedness_moyenne": round(
            moyenne([l["groundedness"] for l in avec_sources]), 4
        ),
        "part_groundedness_sup_080": round(
            sum(1 for l in avec_sources if l["groundedness"] >= 0.80)
            / len(avec_sources)
            if avec_sources
            else 0.0,
            4,
        ),
        # --- Citations
        "citations_valides": part(avec_sources, "citations_valides"),
        "citations_reparees": part(avec_sources, "citations_reparees"),
        "questions_avec_citation_hors_perimetre": sum(
            1 for l in evalues if l["citations_hors_perimetre"] > 0
        ),
        # --- Refus
        "refus_correct": part(sans_reponse, "refus_correct"),
        "refus_incorrect": part(repondables, "refus_incorrect"),
        "taux_refus_global": part(evalues, "refus"),
        # --- Hallucination : répond alors qu'aucune preuve n'a été récupérée.
        "reponses_sans_source": sum(
            1 for l in evalues if not l["refus"] and l["nombre_sources"] == 0
        ),
        "duree_moyenne_secondes": round(
            moyenne([l["duree_secondes"] for l in evalues]), 2
        ),
        "tolerance_numerique": tolerance,
        "llm_judge": llm_judge,
    }

    if llm_judge:
        scores = [l["fidelite_llm"] for l in evalues if l.get("fidelite_llm") is not None]
        resume["fidelite_llm_moyenne"] = round(moyenne(scores), 4)
        resume["fidelite_llm_evaluees"] = len(scores)

    return resume, details


def construire_parseur() -> argparse.ArgumentParser:
    parseur = argparse.ArgumentParser(description="Évalue la génération.")
    parseur.add_argument("--questions", type=Path, required=True)
    parseur.add_argument("--limite", type=int, default=None)
    parseur.add_argument("--nom", default=None)
    parseur.add_argument(
        "--sortie", type=Path, default=DOSSIER_RAPPORTS / "generation"
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

    logger.info(
        "Évaluation de la génération sur %d question(s). "
        "Cette étape appelle le LLM : compte plusieurs dizaines de secondes "
        "par question.",
        len(enregistrements),
    )

    try:
        resume, details = evaluer(
            enregistrements,
            tolerance=arguments.tolerance,
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
    nom = arguments.nom or f"generation_{horodatage()}"
    ecrire_rapport(arguments.sortie, nom, resume=resume, details=details)

    print()
    print(f"  Exactitude            : {resume['exactitude']:.4f}")
    print(f"  Exact match           : {resume['exact_match']:.4f}")
    print(f"  Groundedness moyenne  : {resume['groundedness_moyenne']:.4f}")
    print(f"  Citations valides     : {resume['citations_valides']:.4f}")
    print(f"  Refus correct         : {resume['refus_correct']:.4f}")
    print(f"  Refus incorrect       : {resume['refus_incorrect']:.4f}")
    print(f"  Réponses sans source  : {resume['reponses_sans_source']}")
    print(f"  Temps moyen           : {resume['duree_moyenne_secondes']:.1f} s")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())