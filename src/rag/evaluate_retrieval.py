"""
Évaluation générique du retrieval au niveau document.

Le script ne dépend d'aucun dataset précis. Les éléments suivants sont
fournis depuis la ligne de commande :

- chemin du fichier de questions ;
- nom de la colonne contenant les questions ;
- nom de la colonne contenant les documents pertinents ;
- manière d'interpréter ces documents : noms de fichiers ou indices ;
- paramètres du retrieval.

Métriques calculées :
- Precision@k
- Recall@k
- Hit Rate@k
- MRR@k
- MAP@k

Exemples :

1. Colonne contenant directement un nom de fichier :
    python -m src.evaluation.evaluate_retrieval \
        --questions questions.csv \
        --question-column question \
        --relevant-column filename

2. Colonne contenant un indice numérique :
    python -m src.evaluation.evaluate_retrieval \
        --questions questions.csv \
        --question-column question \
        --relevant-column document_index \
        --relevant-mode index \
        --document-prefix document_ \
        --document-width 2 \
        --document-suffix .txt
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.rag.retrieval import rechercher_passages
from src.rag.vectorstore import fermer_client


logger = logging.getLogger(__name__)


# ===========================================================================
# Lecture des documents pertinents
# ===========================================================================


def _normaliser_nom_document(valeur: Any) -> str:
    """Normalise un nom de document sans modifier sa casse."""
    texte = str(valeur).strip()
    return Path(texte).name if texte else ""


def _convertir_indice_en_nom(
    valeur: Any,
    prefixe: str,
    suffixe: str,
    largeur: int,
) -> str:
    """
    Convertit un indice en nom de fichier.

    Exemple :
        valeur=3, prefixe='document_', largeur=2, suffixe='.txt'
        -> document_03.txt
    """
    indice = int(float(valeur))

    if largeur > 0:
        indice_formate = f"{indice:0{largeur}d}"
    else:
        indice_formate = str(indice)

    return f"{prefixe}{indice_formate}{suffixe}"


def _decoder_liste(valeur: Any, separateur: str) -> list[Any]:
    """
    Accepte plusieurs formats pour les documents pertinents :

    - une valeur simple ;
    - une liste Python/JSON : ["a.pdf", "b.pdf"] ;
    - une chaîne séparée : a.pdf|b.pdf ;
    - une liste déjà chargée par pandas.
    """
    if valeur is None:
        return []

    try:
        if pd.isna(valeur):
            return []
    except (TypeError, ValueError):
        pass

    if isinstance(valeur, (list, tuple, set)):
        return list(valeur)

    texte = str(valeur).strip()
    if not texte:
        return []

    if texte.startswith("[") and texte.endswith("]"):
        for decodeur in (json.loads, ast.literal_eval):
            try:
                resultat = decodeur(texte)
                if isinstance(resultat, (list, tuple, set)):
                    return list(resultat)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue

    if separateur and separateur in texte:
        return [
            element.strip()
            for element in texte.split(separateur)
            if element.strip()
        ]

    return [valeur]


def lire_documents_pertinents(
    valeur: Any,
    *,
    mode: str,
    separateur: str,
    prefixe: str,
    suffixe: str,
    largeur: int,
) -> list[str]:
    """Transforme la valeur du dataset en noms de documents comparables."""
    valeurs = _decoder_liste(valeur, separateur)
    documents: list[str] = []

    for element in valeurs:
        if mode == "index":
            nom = _convertir_indice_en_nom(
                element,
                prefixe=prefixe,
                suffixe=suffixe,
                largeur=largeur,
            )
        else:
            nom = _normaliser_nom_document(element)

        if nom and nom not in documents:
            documents.append(nom)

    return documents


# ===========================================================================
# Résultats récupérés
# ===========================================================================


def dedupliquer_documents(noms: list[str]) -> list[str]:
    """
    Déduplique les documents en conservant leur première apparition.

    Le retrieval renvoie des passages. Plusieurs passages peuvent provenir
    du même document.
    """
    resultat: list[str] = []
    vus: set[str] = set()

    for nom in noms:
        nom_normalise = _normaliser_nom_document(nom)

        if nom_normalise and nom_normalise not in vus:
            vus.add(nom_normalise)
            resultat.append(nom_normalise)

    return resultat


def extraire_documents_recuperes(rapport: Any) -> list[str]:
    """Extrait les documents classés depuis un RapportRecherche."""
    noms = [
        passage.nom_fichier or passage.source
        for passage in rapport.passages
    ]
    return dedupliquer_documents(noms)


# ===========================================================================
# Métriques
# ===========================================================================


def precision_at_k(
    recuperes: list[str],
    pertinents: set[str],
    k: int,
) -> float:
    top_k = recuperes[:k]
    if not top_k:
        return 0.0

    vrais_positifs = sum(
        1 for document in top_k if document in pertinents
    )
    return vrais_positifs / k


def recall_at_k(
    recuperes: list[str],
    pertinents: set[str],
    k: int,
) -> float:
    if not pertinents:
        return 0.0

    retrouves = len(set(recuperes[:k]) & pertinents)
    return retrouves / len(pertinents)


def hit_at_k(
    recuperes: list[str],
    pertinents: set[str],
    k: int,
) -> float:
    return float(bool(set(recuperes[:k]) & pertinents))


def reciprocal_rank_at_k(
    recuperes: list[str],
    pertinents: set[str],
    k: int,
) -> float:
    for rang, document in enumerate(recuperes[:k], start=1):
        if document in pertinents:
            return 1.0 / rang

    return 0.0


def average_precision_at_k(
    recuperes: list[str],
    pertinents: set[str],
    k: int,
) -> float:
    """
    Average Precision@k.

    Cette fonction fonctionne également lorsqu'une question possède
    plusieurs documents pertinents.
    """
    if not pertinents:
        return 0.0

    nombre_pertinents_trouves = 0
    somme_precisions = 0.0

    for rang, document in enumerate(recuperes[:k], start=1):
        if document in pertinents:
            nombre_pertinents_trouves += 1
            somme_precisions += nombre_pertinents_trouves / rang

    denominateur = min(len(pertinents), k)
    return (
        somme_precisions / denominateur
        if denominateur > 0
        else 0.0
    )


# ===========================================================================
# Évaluation
# ===========================================================================


def evaluer_retrieval(
    *,
    chemin_questions: Path,
    colonne_question: str,
    colonne_pertinents: str,
    dossier_sortie: Path,
    nom_experience: str,
    colonne_reponse: str | None = None,
    mode_pertinents: str = "filename",
    separateur_pertinents: str = "|",
    prefixe_document: str = "",
    suffixe_document: str = "",
    largeur_document: int = 0,
    valeurs_k: tuple[int, ...] = (1, 3, 5, 10),
    top_k_retrieval: int = 10,
    candidats: int = 30,
    utiliser_reranker: bool = True,
    appliquer_seuil: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Évalue la fonction rechercher_passages sur un fichier de questions."""
    if not chemin_questions.exists():
        raise FileNotFoundError(
            f"Fichier de questions introuvable : {chemin_questions}"
        )

    suffixe = chemin_questions.suffix.lower()

    if suffixe == ".csv":
        questions = pd.read_csv(chemin_questions)
    elif suffixe in {".json", ".jsonl"}:
        questions = pd.read_json(
            chemin_questions,
            lines=(suffixe == ".jsonl"),
        )
    else:
        raise ValueError(
            "Formats supportés pour les questions : CSV, JSON et JSONL."
        )

    colonnes_requises = {
        colonne_question,
        colonne_pertinents,
    }

    colonnes_manquantes = colonnes_requises - set(questions.columns)

    if colonnes_manquantes:
        raise ValueError(
            f"Colonnes absentes : {sorted(colonnes_manquantes)}. "
            f"Colonnes disponibles : {questions.columns.tolist()}"
        )

    valeurs_k = tuple(sorted(set(valeurs_k)))

    if not valeurs_k or min(valeurs_k) < 1:
        raise ValueError("Les valeurs de k doivent être positives.")

    top_k_retrieval = max(top_k_retrieval, max(valeurs_k))

    lignes: list[dict[str, Any]] = []

    for indice, ligne in tqdm(
        questions.iterrows(),
        total=len(questions),
        desc=f"Évaluation {nom_experience}",
    ):
        question = str(ligne[colonne_question]).strip()

        documents_pertinents = lire_documents_pertinents(
            ligne[colonne_pertinents],
            mode=mode_pertinents,
            separateur=separateur_pertinents,
            prefixe=prefixe_document,
            suffixe=suffixe_document,
            largeur=largeur_document,
        )

        pertinents = set(documents_pertinents)

        resultat_ligne: dict[str, Any] = {
            "question_id": int(indice),
            "question": question,
            "documents_pertinents": json.dumps(
                documents_pertinents,
                ensure_ascii=False,
            ),
            "documents_recuperes": "[]",
            "nombre_passages": 0,
            "duree_secondes": 0.0,
            "erreur": "",
        }

        if colonne_reponse and colonne_reponse in questions.columns:
            resultat_ligne["reponse_attendue"] = str(
                ligne[colonne_reponse]
            )

        try:
            rapport = rechercher_passages(
                requete=question,
                top_k=top_k_retrieval,
                limite_candidats=candidats,
                utiliser_reranker=utiliser_reranker,
                appliquer_seuil=appliquer_seuil,
            )

            recuperes = extraire_documents_recuperes(rapport)

            resultat_ligne.update(
                {
                    "documents_recuperes": json.dumps(
                        recuperes,
                        ensure_ascii=False,
                    ),
                    "nombre_passages": len(rapport.passages),
                    "duree_secondes": rapport.duree_secondes,
                }
            )

            for k in valeurs_k:
                resultat_ligne[f"precision_at_{k}"] = precision_at_k(
                    recuperes,
                    pertinents,
                    k,
                )
                resultat_ligne[f"recall_at_{k}"] = recall_at_k(
                    recuperes,
                    pertinents,
                    k,
                )
                resultat_ligne[f"hit_at_{k}"] = hit_at_k(
                    recuperes,
                    pertinents,
                    k,
                )
                resultat_ligne[f"rr_at_{k}"] = reciprocal_rank_at_k(
                    recuperes,
                    pertinents,
                    k,
                )
                resultat_ligne[f"ap_at_{k}"] = average_precision_at_k(
                    recuperes,
                    pertinents,
                    k,
                )

        except Exception as exc:
            logger.exception(
                "Erreur pendant l'évaluation de la question %s",
                indice,
            )
            resultat_ligne["erreur"] = str(exc)

            for k in valeurs_k:
                resultat_ligne[f"precision_at_{k}"] = 0.0
                resultat_ligne[f"recall_at_{k}"] = 0.0
                resultat_ligne[f"hit_at_{k}"] = 0.0
                resultat_ligne[f"rr_at_{k}"] = 0.0
                resultat_ligne[f"ap_at_{k}"] = 0.0

        lignes.append(resultat_ligne)

    resultats = pd.DataFrame(lignes)

    metriques: dict[str, float] = {}

    for k in valeurs_k:
        metriques[f"precision_at_{k}"] = round(
            float(resultats[f"precision_at_{k}"].mean()),
            4,
        )
        metriques[f"recall_at_{k}"] = round(
            float(resultats[f"recall_at_{k}"].mean()),
            4,
        )
        metriques[f"hit_at_{k}"] = round(
            float(resultats[f"hit_at_{k}"].mean()),
            4,
        )
        metriques[f"mrr_at_{k}"] = round(
            float(resultats[f"rr_at_{k}"].mean()),
            4,
        )
        metriques[f"map_at_{k}"] = round(
            float(resultats[f"ap_at_{k}"].mean()),
            4,
        )

    metriques["duree_moyenne_secondes"] = round(
        float(resultats["duree_secondes"].mean()),
        4,
    )

    resume: dict[str, Any] = {
        "nom_experience": nom_experience,
        "questions_evaluees": int(len(resultats)),
        "questions_en_erreur": int(
            resultats["erreur"].astype(bool).sum()
        ),
        "configuration": {
            "fichier_questions": str(chemin_questions),
            "colonne_question": colonne_question,
            "colonne_pertinents": colonne_pertinents,
            "mode_pertinents": mode_pertinents,
            "valeurs_k": list(valeurs_k),
            "top_k_retrieval": top_k_retrieval,
            "candidats": candidats,
            "reranker": utiliser_reranker,
            "seuil": appliquer_seuil,
        },
        "metriques": metriques,
    }

    dossier_sortie.mkdir(parents=True, exist_ok=True)

    chemin_csv = dossier_sortie / f"{nom_experience}_details.csv"
    chemin_json = dossier_sortie / f"{nom_experience}_resume.json"

    resultats.to_csv(
        chemin_csv,
        index=False,
        encoding="utf-8",
    )

    chemin_json.write_text(
        json.dumps(resume, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    resume["fichiers_generes"] = {
        "details": str(chemin_csv),
        "resume": str(chemin_json),
    }

    return resultats, resume


# ===========================================================================
# CLI
# ===========================================================================


def _parser_k(valeur: str) -> tuple[int, ...]:
    try:
        resultat = tuple(
            int(element.strip())
            for element in valeur.split(",")
            if element.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--k doit être une liste d'entiers, par exemple 1,3,5,10."
        ) from exc

    if not resultat or min(resultat) < 1:
        raise argparse.ArgumentTypeError(
            "Les valeurs de --k doivent être positives."
        )

    return resultat


def afficher_resume(resume: dict[str, Any]) -> None:
    print("\n" + "=" * 68)
    print(f"ÉVALUATION RETRIEVAL — {resume['nom_experience']}")
    print("=" * 68)

    print(f"Questions évaluées  : {resume['questions_evaluees']}")
    print(f"Questions en erreur : {resume['questions_en_erreur']}")

    print("\nMétriques")
    for nom, valeur in resume["metriques"].items():
        print(f"  {nom:28s}: {valeur}")

    print("\nFichiers")
    for nom, chemin in resume["fichiers_generes"].items():
        print(f"  {nom:28s}: {chemin}")

    print("=" * 68 + "\n")


def main() -> None:
    parseur = argparse.ArgumentParser(
        description="Évaluation générique du retrieval au niveau document."
    )

    parseur.add_argument(
        "--questions",
        type=Path,
        required=True,
        help="Chemin vers le CSV, JSON ou JSONL de questions.",
    )
    parseur.add_argument(
        "--question-column",
        required=True,
        help="Nom de la colonne contenant la question.",
    )
    parseur.add_argument(
        "--relevant-column",
        required=True,
        help="Colonne contenant les documents pertinents.",
    )
    parseur.add_argument(
        "--answer-column",
        default=None,
        help="Colonne optionnelle contenant la réponse attendue.",
    )
    parseur.add_argument(
        "--relevant-mode",
        choices=("filename", "index"),
        default="filename",
        help="Interprétation de la colonne pertinente.",
    )
    parseur.add_argument(
        "--relevant-separator",
        default="|",
        help="Séparateur lorsque plusieurs documents sont indiqués.",
    )
    parseur.add_argument("--document-prefix", default="")
    parseur.add_argument("--document-suffix", default="")
    parseur.add_argument("--document-width", type=int, default=0)

    parseur.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/logs/evaluation_retrieval"),
    )
    parseur.add_argument(
        "--name",
        default="retrieval_baseline",
        help="Nom de l'expérience et préfixe des fichiers générés.",
    )
    parseur.add_argument(
        "--k",
        type=_parser_k,
        default=(1, 3, 5, 10),
        help="Valeurs de k, par exemple 1,3,5,10.",
    )
    parseur.add_argument("--top-k", type=int, default=10)
    parseur.add_argument("--candidats", type=int, default=30)
    parseur.add_argument("--sans-reranker", action="store_true")
    parseur.add_argument("--sans-seuil", action="store_true")

    args = parseur.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    try:
        _, resume = evaluer_retrieval(
            chemin_questions=args.questions,
            colonne_question=args.question_column,
            colonne_pertinents=args.relevant_column,
            colonne_reponse=args.answer_column,
            dossier_sortie=args.output_dir,
            nom_experience=args.name,
            mode_pertinents=args.relevant_mode,
            separateur_pertinents=args.relevant_separator,
            prefixe_document=args.document_prefix,
            suffixe_document=args.document_suffix,
            largeur_document=args.document_width,
            valeurs_k=args.k,
            top_k_retrieval=args.top_k,
            candidats=args.candidats,
            utiliser_reranker=not args.sans_reranker,
            appliquer_seuil=not args.sans_seuil,
        )

        afficher_resume(resume)

    finally:
        fermer_client()


if __name__ == "__main__":
    main()