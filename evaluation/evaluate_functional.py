"""
Évaluation fonctionnelle simple de l'agent documentaire.

Objectif :
    question UDA
        -> pipeline RAG réel
        -> réponse générée
        -> comparaison avec expected_answer
        -> vérification du document cité
        -> PASS / FAIL

Contrairement à evaluate_end_to_end.py, cette évaluation ne dépend pas
du seuil de couverture evidence-level. Elle mesure directement le
comportement observable de l'agent.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.rag.generation import generer_reponse


logger = logging.getLogger(__name__)


# ===========================================================================
# 1. Structures
# ===========================================================================


@dataclass
class ResultatFonctionnel:
    id: str
    question: str
    answerable: bool

    expected_answer: str
    generated_answer: str

    expected_document: str
    cited_documents: str

    answer_score: float
    answer_correct: bool
    document_correct: bool | None
    refusal_correct: bool | None

    success: bool
    latency_seconds: float
    error: str = ""


# ===========================================================================
# 2. Normalisation
# ===========================================================================


def normaliser_texte(texte: str | None) -> str:
    if not texte:
        return ""

    texte = unicodedata.normalize("NFKD", str(texte))
    texte = "".join(
        caractere
        for caractere in texte
        if not unicodedata.combining(caractere)
    )

    texte = texte.lower()

    # On garde lettres/chiffres mais on neutralise la ponctuation.
    texte = re.sub(r"[^\w\s]", " ", texte, flags=re.UNICODE)
    texte = re.sub(r"\s+", " ", texte).strip()

    return texte


def tokens(texte: str | None) -> list[str]:
    return normaliser_texte(texte).split()


# ===========================================================================
# 3. Score de réponse
# ===========================================================================


def token_f1(prediction: str, reference: str) -> float:
    """
    F1 lexical simple entre la réponse générée et la référence.

    Plus robuste qu'une égalité exacte :
    "The answer is Food.com"
    peut matcher
    "from Food.com".
    """
    pred = tokens(prediction)
    ref = tokens(reference)

    if not pred or not ref:
        return 0.0

    compteur_pred: dict[str, int] = {}
    compteur_ref: dict[str, int] = {}

    for mot in pred:
        compteur_pred[mot] = compteur_pred.get(mot, 0) + 1

    for mot in ref:
        compteur_ref[mot] = compteur_ref.get(mot, 0) + 1

    commun = sum(
        min(compteur_pred.get(mot, 0), compteur_ref.get(mot, 0))
        for mot in compteur_ref
    )

    if commun == 0:
        return 0.0

    precision = commun / len(pred)
    rappel = commun / len(ref)

    return 2 * precision * rappel / (precision + rappel)


def score_reponse(
    prediction: str,
    expected_answer: str,
    answer_variants: list[str],
) -> float:
    """
    Prend le meilleur score parmi expected_answer et les variantes UDA.
    """
    references = [expected_answer]

    references.extend(
        variante
        for variante in answer_variants
        if isinstance(variante, str) and variante.strip()
    )

    scores = [
        token_f1(prediction, reference)
        for reference in references
        if reference.strip()
    ]

    return max(scores, default=0.0)


# ===========================================================================
# 4. Refus
# ===========================================================================


MOTIFS_REFUS = (
    "je ne peux pas",
    "je ne trouve pas",
    "je n'ai pas trouve",
    "information insuffisante",
    "informations insuffisantes",
    "aucune information",
    "aucune source",
    "pas suffisamment d'information",
    "pas suffisamment d informations",
    "non disponible dans les documents",
    "non mentionne",
    "not enough information",
    "cannot answer",
    "can't answer",
    "could not find",
    "not found",
    "not mentioned",
    "insufficient information",
    "no information",
    "no source",
)


def est_refus(reponse: str) -> bool:
    texte = normaliser_texte(reponse)

    return any(
        normaliser_texte(motif) in texte
        for motif in MOTIFS_REFUS
    )


# ===========================================================================
# 5. Extraction robuste de la réponse et des citations
# ===========================================================================


def texte_reponse(objet: Any) -> str:
    """
    Essaie plusieurs formes possibles de ReponseSourcee.
    """

    for attribut in ("reponse", "answer", "texte", "content"):
        valeur = getattr(objet, attribut, None)

        if isinstance(valeur, str):
            return valeur

    if isinstance(objet, str):
        return objet

    return str(objet)


def _chercher_source(objet: Any) -> str | None:
    """
    Cherche récursivement un nom/path de document dans une citation.
    """

    if objet is None:
        return None

    if isinstance(objet, str):
        return objet

    if isinstance(objet, dict):
        for cle in (
            "source",
            "document",
            "fichier",
            "filename",
            "file",
            "nom_fichier",
            "path",
        ):
            valeur = objet.get(cle)

            if isinstance(valeur, str):
                return valeur

        for valeur in objet.values():
            resultat = _chercher_source(valeur)

            if resultat:
                return resultat

        return None

    for attribut in (
        "source",
        "document",
        "fichier",
        "filename",
        "file",
        "nom_fichier",
        "path",
    ):
        valeur = getattr(objet, attribut, None)

        if isinstance(valeur, str):
            return valeur

    if hasattr(objet, "__dict__"):
        return _chercher_source(vars(objet))

    return None


def documents_cites(reponse: Any) -> list[str]:
    """
    Extrait les documents cités depuis plusieurs structures possibles.
    """

    documents: list[str] = []

    for attribut in (
        "citations",
        "sources",
        "documents",
        "passages",
        "chunks",
    ):
        valeur = getattr(reponse, attribut, None)

        if not valeur:
            continue

        if not isinstance(valeur, (list, tuple)):
            valeur = [valeur]

        for element in valeur:
            source = _chercher_source(element)

            if source:
                documents.append(Path(source).name)

    # Déduplication en conservant l'ordre.
    uniques: list[str] = []

    for document in documents:
        if document not in uniques:
            uniques.append(document)

    return uniques


# ===========================================================================
# 6. Chargement dataset
# ===========================================================================


def charger_questions(chemin: Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []

    with chemin.open("r", encoding="utf-8") as fichier:
        for numero, ligne in enumerate(fichier, start=1):
            ligne = ligne.strip()

            if not ligne:
                continue

            try:
                questions.append(json.loads(ligne))

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON invalide ligne {numero} dans {chemin}: {exc}"
                ) from exc

    return questions


# ===========================================================================
# 7. Évaluation d'un cas
# ===========================================================================


def evaluer_cas(
    cas: dict[str, Any],
    seuil_f1: float,
) -> ResultatFonctionnel:

    identifiant = str(cas.get("id", ""))
    question = str(cas.get("question", ""))

    expected_answer = str(cas.get("expected_answer") or "")
    expected_document = Path(
        str(cas.get("expected_document") or "")
    ).name

    answer_variants = cas.get("answer_variants") or []

    if not isinstance(answer_variants, list):
        answer_variants = []

    answerable = bool(cas.get("answerable", True))

    debut = time.perf_counter()

    try:
        # ---------------------------------------------------------------
        # VRAI PIPELINE RAG
        # ---------------------------------------------------------------

        reponse = generer_reponse(question)

        latence = time.perf_counter() - debut

        prediction = texte_reponse(reponse)
        cites = documents_cites(reponse)

        # ---------------------------------------------------------------
        # Cas answerable
        # ---------------------------------------------------------------

        if answerable:

            score = score_reponse(
                prediction,
                expected_answer,
                answer_variants,
            )

            answer_correct = score >= seuil_f1

            if expected_document:
                document_correct: bool | None = (
                    expected_document in cites
                )
            else:
                document_correct = None

            refusal_correct = None

            # Le critère principal est la réponse.
            #
            # La citation est mesurée séparément afin de ne pas transformer
            # une bonne réponse en FAIL simplement à cause du format interne
            # des citations.
            success = answer_correct

        # ---------------------------------------------------------------
        # Cas non-answerable
        # ---------------------------------------------------------------

        else:

            score = 0.0
            answer_correct = False
            document_correct = None

            refusal_correct = est_refus(prediction)
            success = refusal_correct

        return ResultatFonctionnel(
            id=identifiant,
            question=question,
            answerable=answerable,
            expected_answer=expected_answer,
            generated_answer=prediction,
            expected_document=expected_document,
            cited_documents=" | ".join(cites),
            answer_score=round(score, 4),
            answer_correct=answer_correct,
            document_correct=document_correct,
            refusal_correct=refusal_correct,
            success=success,
            latency_seconds=round(latence, 3),
        )

    except Exception as exc:  # noqa: BLE001

        latence = time.perf_counter() - debut

        logger.exception(
            "Erreur pendant l'évaluation de %s",
            identifiant,
        )

        return ResultatFonctionnel(
            id=identifiant,
            question=question,
            answerable=answerable,
            expected_answer=expected_answer,
            generated_answer="",
            expected_document=expected_document,
            cited_documents="",
            answer_score=0.0,
            answer_correct=False,
            document_correct=False if answerable else None,
            refusal_correct=False if not answerable else None,
            success=False,
            latency_seconds=round(latence, 3),
            error=repr(exc),
        )


# ===========================================================================
# 8. Rapport
# ===========================================================================


def sauvegarder_csv(
    chemin: Path,
    resultats: list[ResultatFonctionnel],
) -> None:

    chemin.parent.mkdir(parents=True, exist_ok=True)

    if not resultats:
        return

    lignes = [asdict(resultat) for resultat in resultats]

    with chemin.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as fichier:

        writer = csv.DictWriter(
            fichier,
            fieldnames=list(lignes[0].keys()),
        )

        writer.writeheader()
        writer.writerows(lignes)


def calculer_rapport(
    resultats: list[ResultatFonctionnel],
) -> dict[str, Any]:

    answerables = [
        resultat
        for resultat in resultats
        if resultat.answerable
    ]

    non_answerables = [
        resultat
        for resultat in resultats
        if not resultat.answerable
    ]

    bonnes_reponses = sum(
        resultat.answer_correct
        for resultat in answerables
    )

    refus_corrects = sum(
        resultat.refusal_correct is True
        for resultat in non_answerables
    )

    citations_mesurables = [
        resultat
        for resultat in answerables
        if resultat.document_correct is not None
    ]

    bons_documents = sum(
        resultat.document_correct is True
        for resultat in citations_mesurables
    )

    succes = sum(
        resultat.success
        for resultat in resultats
    )

    erreurs = sum(
        bool(resultat.error)
        for resultat in resultats
    )

    temps_moyen = (
        sum(resultat.latency_seconds for resultat in resultats)
        / len(resultats)
        if resultats
        else 0.0
    )

    return {
        "questions": len(resultats),

        "answerable": len(answerables),
        "non_answerable": len(non_answerables),

        "reponses_correctes": bonnes_reponses,
        "refus_corrects": refus_corrects,

        "citations_mesurables": len(citations_mesurables),
        "documents_corrects": bons_documents,

        "succes": succes,
        "erreurs": erreurs,

        "answer_accuracy": (
            bonnes_reponses / len(answerables)
            if answerables
            else 0.0
        ),

        "refusal_accuracy": (
            refus_corrects / len(non_answerables)
            if non_answerables
            else 0.0
        ),

        "document_accuracy": (
            bons_documents / len(citations_mesurables)
            if citations_mesurables
            else 0.0
        ),

        "global_success": (
            succes / len(resultats)
            if resultats
            else 0.0
        ),

        "average_latency_seconds": temps_moyen,
    }


def afficher_rapport(rapport: dict[str, Any]) -> None:

    print()
    print("=" * 62)
    print("  ÉVALUATION FONCTIONNELLE — AGENT DOCUMENTAIRE")
    print("=" * 62)

    print(
        f"  Questions                  : "
        f"{rapport['questions']}"
    )

    print(
        f"  Questions answerable       : "
        f"{rapport['answerable']}"
    )

    print(
        f"  Questions non-answerable   : "
        f"{rapport['non_answerable']}"
    )

    print()

    print(
        f"  Réponses correctes         : "
        f"{rapport['reponses_correctes']}/"
        f"{rapport['answerable']}"
    )

    print(
        f"  Exactitude réponses        : "
        f"{rapport['answer_accuracy']:.1%}"
    )

    if rapport["non_answerable"]:

        print(
            f"  Refus corrects             : "
            f"{rapport['refus_corrects']}/"
            f"{rapport['non_answerable']}"
        )

        print(
            f"  Exactitude refus           : "
            f"{rapport['refusal_accuracy']:.1%}"
        )

    if rapport["citations_mesurables"]:

        print(
            f"  Bon document cité          : "
            f"{rapport['documents_corrects']}/"
            f"{rapport['citations_mesurables']}"
        )

        print(
            f"  Exactitude documents       : "
            f"{rapport['document_accuracy']:.1%}"
        )

    print()
    print("-" * 62)

    print(
        f"  SUCCÈS GLOBAL              : "
        f"{rapport['global_success']:.1%}"
    )

    print(
        f"  Temps moyen / question     : "
        f"{rapport['average_latency_seconds']:.2f}s"
    )

    print(
        f"  Erreurs pipeline           : "
        f"{rapport['erreurs']}"
    )

    print("=" * 62)
    print()


# ===========================================================================
# 9. CLI
# ===========================================================================


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Évaluation fonctionnelle simple du RAG"
    )

    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
        help="fichier JSONL contenant les questions",
    )

    parser.add_argument(
        "--nom",
        type=str,
        default="functional",
        help="nom du rapport",
    )

    parser.add_argument(
        "--seuil-f1",
        type=float,
        default=0.50,
        help="F1 lexical minimal pour considérer une réponse correcte",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="évaluer uniquement les N premiers cas",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    questions = charger_questions(args.questions)

    if args.limit is not None:
        questions = questions[: args.limit]

    logger.info(
        "Évaluation fonctionnelle sur %d question(s).",
        len(questions),
    )

    resultats: list[ResultatFonctionnel] = []

    for index, cas in enumerate(questions, start=1):

        resultat = evaluer_cas(
            cas,
            seuil_f1=args.seuil_f1,
        )

        resultats.append(resultat)

        statut = "PASS" if resultat.success else "FAIL"

        logger.info(
            "[%02d/%02d] %s | score=%.3f | %s",
            index,
            len(questions),
            statut,
            resultat.answer_score,
            resultat.question[:70],
        )

    rapport = calculer_rapport(resultats)

    dossier = Path("evaluation/reports/functional")
    dossier.mkdir(parents=True, exist_ok=True)

    chemin_csv = dossier / f"{args.nom}.csv"
    chemin_json = dossier / f"{args.nom}.json"

    sauvegarder_csv(
        chemin_csv,
        resultats,
    )

    with chemin_json.open(
        "w",
        encoding="utf-8",
    ) as fichier:

        json.dump(
            {
                "summary": rapport,
                "results": [
                    asdict(resultat)
                    for resultat in resultats
                ],
            },
            fichier,
            ensure_ascii=False,
            indent=2,
        )

    afficher_rapport(rapport)

    logger.info("Rapport écrit : %s", chemin_json)
    logger.info("Détail écrit  : %s", chemin_csv)


if __name__ == "__main__":
    main()