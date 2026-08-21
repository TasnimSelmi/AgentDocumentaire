"""
Préparation du benchmark d'évaluation.

Télécharge le benchmark, copie les documents sources dans un dossier
ingérable par le pipeline, et produit un JSONL normalisé identique quel que
soit le benchmark. Toute la logique spécifique au dataset vit ici et
nulle part ailleurs : les six autres scripts ne connaissent que le format
`Enregistrement`.

Benchmark principal : UDA (NeurIPS'24), sous-jeux paper_text et paper_tab.
    - vrais PDF -> le loader, l'OCR et le chunking du projet sont exercés
    - evidences héritées de QASPER, annotées au niveau paragraphe
    - questions sans réponse natives -> évaluation du refus
    - licence CC-BY-SA-4.0

Benchmark complémentaire : MultiHop-RAG (COLM'24), licence ODC-BY.
    Multi-document et null queries. Prévu pour la couche agentique, pas
    nécessaire au gel du socle RAG.

Exemples
--------
    # 1. Télécharger (une seule fois, ~1 Go pour paper_text + paper_tab)
    python -m evaluation.prepare_dataset --benchmark uda --telecharger

    # 2. Construire l'échantillon reproductible
    python -m evaluation.prepare_dataset --benchmark uda \\
        --taille-echantillon 400 --seed 20240601

    # 3. Smoke test (20 questions, 10 documents)
    python -m evaluation.prepare_dataset --benchmark uda \\
        --taille-echantillon 20 --nom smoke

Après cette étape, ingère les documents avec le pipeline EXISTANT :

    python -m src.rag.ingestion --dossier evaluation/data/corpus
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from evaluation.common import (
    DOSSIER_DONNEES,
    Enregistrement,
    charger_enregistrements,
    configurer_logs,
    ecrire_jsonl,
    fixer_seed,
    nom_document,
)

logger = logging.getLogger("evaluation.prepare")

DEPOT_UDA = "qinchuanhui/UDA-QA"
DEPOT_MULTIHOP = "yixuantt/MultiHopRAG"

SOUS_JEUX_UDA_PAR_DEFAUT = ("paper_text", "paper_tab")


# ===========================================================================
# Téléchargement
# ===========================================================================


def telecharger_uda(
    destination: Path,
    sous_jeux: Iterable[str],
) -> Path:
    """
    Télécharge les annotations UDA utiles et les documents sources nécessaires.

    Pour les sous-jeux ``paper_text`` et ``paper_tab``, les PDF sont distribués
    dans ``src_doc_files/paper_docs.zip``. L'archive est téléchargée puis extraite
    automatiquement sous ``src_doc_files/paper_docs``.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "huggingface_hub est requis pour le téléchargement.\n"
            "    pip install huggingface_hub"
        ) from exc

    sous_jeux = tuple(sous_jeux)
    motifs: list[str] = ["*.md"]

    for sous_jeu in sous_jeux:
        motifs.append(f"extended_qa_info/{sous_jeu}_qa.json")
        motifs.append(f"{sous_jeu}/*.parquet")

    if any(s in {"paper_text", "paper_tab"} for s in sous_jeux):
        motifs.append("src_doc_files/paper_docs.zip")

    logger.info("Téléchargement de %s vers %s", DEPOT_UDA, destination)
    logger.info("Motifs : %s", ", ".join(motifs))

    chemin = Path(
        snapshot_download(
            repo_id=DEPOT_UDA,
            repo_type="dataset",
            local_dir=str(destination),
            allow_patterns=motifs,
        )
    )

    archive = destination / "src_doc_files" / "paper_docs.zip"
    dossier_extrait = destination / "src_doc_files" / "paper_docs"

    if archive.exists():
        deja_extraits = (
            any(dossier_extrait.rglob("*.pdf"))
            if dossier_extrait.exists()
            else False
        )
        if not deja_extraits:
            logger.info("Extraction de %s vers %s", archive, dossier_extrait)
            dossier_extrait.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(dossier_extrait)
        else:
            logger.info("PDF UDA déjà extraits dans %s", dossier_extrait)

    return chemin


def telecharger_multihop(destination: Path) -> Path:
    """Télécharge MultiHop-RAG (corpus.json + MultiHopRAG.json, ~12 Mo)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "huggingface_hub est requis.\n    pip install huggingface_hub"
        ) from exc

    logger.info("Téléchargement de %s vers %s", DEPOT_MULTIHOP, destination)
    return Path(
        snapshot_download(
            repo_id=DEPOT_MULTIHOP,
            repo_type="dataset",
            local_dir=str(destination),
        )
    )


# ===========================================================================
# Lecture des annotations UDA
# ===========================================================================


def _lire_table(chemin: Path) -> list[dict[str, Any]]:
    """
    Lit un CSV, JSON, JSONL ou parquet en liste de dictionnaires.

    Les JSON ``extended_qa_info`` d'UDA sont indexés par ``doc_name`` et sont
    aplatis ici pour produire une ligne par question.
    """
    import pandas as pd

    suffixe = chemin.suffix.lower()

    if suffixe == ".csv":
        cadre = pd.read_csv(
            chemin, dtype=str, keep_default_na=False, na_values=[""]
        )
        return cadre.where(cadre.notna(), None).to_dict(orient="records")

    if suffixe == ".parquet":
        cadre = pd.read_parquet(chemin)
        return cadre.where(cadre.notna(), None).to_dict(orient="records")

    if suffixe == ".jsonl":
        cadre = pd.read_json(chemin, lines=True, dtype=False, convert_dates=False)
        return cadre.where(cadre.notna(), None).to_dict(orient="records")

    if suffixe == ".json":
        donnees = json.loads(chemin.read_text(encoding="utf-8"))

        if isinstance(donnees, dict) and all(
            isinstance(v, list) for v in donnees.values()
        ):
            lignes: list[dict[str, Any]] = []
            for doc_name, questions in donnees.items():
                for item in questions:
                    if not isinstance(item, dict):
                        continue
                    ligne = dict(item)
                    ligne["doc_name"] = str(doc_name)
                    lignes.append(ligne)
            return lignes

        if isinstance(donnees, list):
            return [item for item in donnees if isinstance(item, dict)]

        raise ValueError(f"Structure JSON non reconnue : {chemin}")

    raise ValueError(f"Format non géré : {chemin}")


def _trouver_annotations(racine: Path, sous_jeu: str) -> list[Path]:
    """Localise les annotations d'un sous-jeu UDA sans mélanger les sources."""
    enrichi = racine / "extended_qa_info" / f"{sous_jeu}_qa.json"
    if enrichi.exists():
        return [enrichi]

    candidats: list[Path] = []
    for extension in ("csv", "parquet", "jsonl", "json"):
        candidats.extend(racine.rglob(f"*{sous_jeu}*.{extension}"))

    return sorted(
        chemin
        for chemin in candidats
        if "src_doc_files" not in chemin.parts
    )


def _premier_non_vide(ligne: dict[str, Any], *cles: str) -> str:
    for cle in cles:
        valeur = ligne.get(cle)
        if valeur is None:
            continue
        texte = str(valeur).strip()
        if texte and texte.lower() not in {"nan", "none"}:
            return texte
    return ""


def _extraire_reponses(
    ligne: dict[str, Any],
) -> tuple[str, list[str], str]:
    """Retourne réponse principale, variantes et type(s) de réponse UDA."""
    answers = ligne.get("answers")

    if isinstance(answers, list):
        reponses: list[str] = []
        types: list[str] = []

        for element in answers:
            if isinstance(element, dict):
                texte = str(element.get("answer") or "").strip()
                type_reponse = str(element.get("type") or "").strip()
            else:
                texte = str(element).strip()
                type_reponse = ""

            if (
                texte
                and texte.lower() not in {"nan", "none"}
                and texte not in reponses
            ):
                reponses.append(texte)
            if type_reponse and type_reponse not in types:
                types.append(type_reponse)

        if reponses:
            return reponses[0], reponses[1:], ",".join(types)
        return "", [], ",".join(types)

    reponse = _premier_non_vide(
        ligne, "answer", "answer_1", "short_answer", "free_form_answer"
    )
    variantes = [
        v
        for v in (
            _premier_non_vide(ligne, "answer_2"),
            _premier_non_vide(ligne, "answer_3"),
            _premier_non_vide(ligne, "long_answer"),
        )
        if v and v != reponse
    ]
    return reponse, variantes, _premier_non_vide(
        ligne, "answer_type", "question_type"
    )


def _extraire_evidence(ligne: dict[str, Any]) -> str:
    """Extrait le texte de preuve UDA en privilégiant ``raw_evidence``."""
    valeur = ligne.get("evidence")

    if isinstance(valeur, str):
        texte = valeur.strip()
        if texte and texte.lower() not in {"nan", "none", "[]"}:
            return texte

    if isinstance(valeur, (list, tuple)):
        morceaux: list[str] = []

        for element in valeur:
            if isinstance(element, dict):
                candidats = (
                    element.get("raw_evidence")
                    or element.get("highlighted_evidence")
                    or []
                )
                if isinstance(candidats, str):
                    candidats = [candidats]
                if isinstance(candidats, (list, tuple)):
                    for candidat in candidats:
                        texte = str(candidat).strip()
                        if texte and texte not in morceaux:
                            morceaux.append(texte)
            else:
                texte = str(element).strip()
                if texte and texte not in morceaux:
                    morceaux.append(texte)

        if morceaux:
            return "\n\n".join(morceaux)

    for cle in (
        "evidences",
        "highlighted_evidence",
        "context",
        "gold_evidence",
    ):
        valeur = ligne.get(cle)
        if isinstance(valeur, str):
            texte = valeur.strip()
            if texte and texte.lower() not in {"nan", "none", "[]"}:
                return texte
        elif isinstance(valeur, (list, tuple)):
            morceaux = [str(v).strip() for v in valeur if str(v).strip()]
            if morceaux:
                return "\n\n".join(morceaux)

    return ""


def _est_sans_reponse(ligne: dict[str, Any], reponse: str) -> bool:
    """Détermine si la question est volontairement sans réponse."""
    drapeau = ligne.get("unanswerable")
    if isinstance(drapeau, bool):
        return drapeau
    if isinstance(drapeau, str):
        return drapeau.strip().lower() in {"true", "1", "yes"}

    return (not reponse) or reponse.strip().lower() in {
        "unanswerable",
        "n/a",
        "none",
    }


def normaliser_uda(
    racine: Path,
    sous_jeux: Iterable[str],
    *,
    suffixe_document: str = ".pdf",
) -> list[Enregistrement]:
    """Convertit les annotations UDA en enregistrements normalisés."""
    enregistrements: list[Enregistrement] = []

    for sous_jeu in sous_jeux:
        fichiers = _trouver_annotations(racine, sous_jeu)
        if not fichiers:
            logger.warning(
                "Aucun fichier d'annotations trouvé pour le sous-jeu %r sous %s. "
                "Vérifie le téléchargement (--telecharger).",
                sous_jeu,
                racine,
            )
            continue

        logger.info(
            "Sous-jeu %s : %d fichier(s) d'annotations.", sous_jeu, len(fichiers)
        )

        vus: set[str] = set()
        for fichier in fichiers:
            try:
                lignes = _lire_table(fichier)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Fichier ignoré (%s) : %s", fichier.name, exc)
                continue

            for ligne in lignes:
                question = _premier_non_vide(ligne, "question", "query")
                doc_name = _premier_non_vide(ligne, "doc_name", "document", "title")
                if not question or not doc_name:
                    continue

                q_uid = _premier_non_vide(ligne, "q_uid", "question_id", "id")
                identifiant = f"{sous_jeu}:{q_uid or f'{doc_name}:{len(vus)}'}"
                if identifiant in vus:
                    continue
                vus.add(identifiant)

                reponse, variantes, type_reponse = _extraire_reponses(ligne)

                page = ligne.get("page") or ligne.get("page_number")
                try:
                    page = int(page) if page is not None else None
                except (TypeError, ValueError):
                    page = None

                enregistrements.append(
                    Enregistrement(
                        id=identifiant,
                        question=question,
                        expected_answer=reponse,
                        expected_document=f"{doc_name}{suffixe_document}",
                        evidence_text=_extraire_evidence(ligne),
                        page=page,
                        answerable=not _est_sans_reponse(ligne, reponse),
                        subset=sous_jeu,
                        question_type=(
                            type_reponse
                            or _premier_non_vide(
                                ligne, "answer_type", "question_type"
                            )
                        ),
                        answer_variants=variantes,
                        metadata={
                            "doc_name": doc_name,
                            "q_uid": q_uid,
                            "benchmark": "uda",
                            "fichier_source": fichier.name,
                        },
                    )
                )

    return enregistrements


def normaliser_multihop(racine: Path) -> list[Enregistrement]:
    """
    Convertit MultiHop-RAG en enregistrements normalisés.

    Chaque requête possède des evidences réparties sur 2 à 4 documents. Le
    format normalisé ne portant qu'un document attendu, on retient le premier
    et on conserve la liste complète dans `metadata['documents_attendus']`,
    exploitée par l'évaluation multi-document de la couche agentique.
    """
    fichier_qa = racine / "MultiHopRAG.json"
    if not fichier_qa.exists():
        raise FileNotFoundError(
            f"{fichier_qa} introuvable. Lance --telecharger."
        )

    requetes = json.loads(fichier_qa.read_text(encoding="utf-8"))
    enregistrements: list[Enregistrement] = []

    for index, requete in enumerate(requetes):
        question = str(requete.get("query", "")).strip()
        if not question:
            continue

        faits = requete.get("evidence_list") or []
        documents: list[str] = []
        morceaux: list[str] = []
        for fait in faits:
            titre = str(fait.get("title", "")).strip()
            if titre and titre not in documents:
                documents.append(titre)
            extrait = str(fait.get("fact", "")).strip()
            if extrait:
                morceaux.append(extrait)

        type_requete = str(requete.get("question_type", "")).strip()
        reponse = str(requete.get("answer", "")).strip()

        enregistrements.append(
            Enregistrement(
                id=f"multihop:{index}",
                question=question,
                expected_answer=reponse,
                expected_document=documents[0] if documents else None,
                evidence_text="\n\n".join(morceaux),
                page=None,
                # Les null queries sont les cas sans réponse dérivable.
                answerable=type_requete != "null_query" and bool(documents),
                subset="multihop",
                question_type=type_requete,
                metadata={
                    "benchmark": "multihop-rag",
                    "documents_attendus": documents,
                    "nombre_documents": len(documents),
                },
            )
        )

    return enregistrements


# ===========================================================================
# Corpus ingérable
# ===========================================================================


def preparer_corpus_uda(
    racine: Path,
    enregistrements: list[Enregistrement],
    destination: Path,
) -> int:
    """
    Copie dans `destination` les seuls PDF référencés par l'échantillon.

    Ingérer 1 394 documents pour n'en interroger que quelques centaines
    serait inutilement coûteux. On ne copie donc que le nécessaire, ce qui
    rend aussi le corpus reproductible : même seed, même corpus.
    """
    destination.mkdir(parents=True, exist_ok=True)

    attendus = {
        nom_document(e.expected_document)
        for e in enregistrements
        if e.expected_document
    }

    disponibles: dict[str, Path] = {}
    for chemin in (racine / "src_doc_files").rglob("*"):
        if chemin.is_file():
            disponibles.setdefault(chemin.name, chemin)

    copies = 0
    manquants: list[str] = []

    for nom in sorted(attendus):
        source = disponibles.get(nom)
        if source is None:
            manquants.append(nom)
            continue
        cible = destination / nom
        if not cible.exists():
            shutil.copy2(source, cible)
        copies += 1

    if manquants:
        logger.warning(
            "%d document(s) attendu(s) absent(s) du téléchargement, "
            "par exemple : %s",
            len(manquants),
            ", ".join(manquants[:5]),
        )

    logger.info("Corpus prêt : %d document(s) dans %s", copies, destination)
    return copies


# ===========================================================================
# Échantillonnage stratifié
# ===========================================================================


def echantillonner(
    enregistrements: list[Enregistrement],
    taille: int,
    seed: int,
    *,
    part_sans_reponse: float = 0.20,
) -> list[Enregistrement]:
    """
    Échantillon stratifié et reproductible.

    Stratification par (sous-jeu, répondable), avec une part garantie de
    questions sans réponse : sans elles, le taux de refus n'est pas mesurable,
    et un système qui répond toujours obtiendrait un score flatteur.

    Le tri préalable par identifiant garantit que l'échantillon ne dépend pas
    de l'ordre de lecture des fichiers.
    """
    if taille <= 0 or taille >= len(enregistrements):
        return sorted(enregistrements, key=lambda e: e.id)

    alea = fixer_seed(seed)

    repondables = sorted(
        (e for e in enregistrements if e.answerable), key=lambda e: e.id
    )
    sans_reponse = sorted(
        (e for e in enregistrements if not e.answerable), key=lambda e: e.id
    )

    cible_sans = min(len(sans_reponse), int(round(taille * part_sans_reponse)))
    cible_avec = min(len(repondables), taille - cible_sans)

    # Si l'un des deux groupes est trop petit, l'autre compense.
    reste = taille - cible_avec - cible_sans
    if reste > 0:
        supplement = min(reste, len(repondables) - cible_avec)
        cible_avec += supplement

    def tirer(population: list[Enregistrement], nombre: int, par_strate: bool) -> list[Enregistrement]:
        if nombre <= 0 or not population:
            return []
        if not par_strate:
            return alea.sample(population, min(nombre, len(population)))

        groupes: dict[str, list[Enregistrement]] = defaultdict(list)
        for element in population:
            groupes[element.subset].append(element)

        selection: list[Enregistrement] = []
        noms = sorted(groupes)
        quota = max(1, nombre // len(noms))
        for nom in noms:
            selection.extend(
                alea.sample(groupes[nom], min(quota, len(groupes[nom])))
            )
        # Complément aléatoire si le quota entier n'a pas suffi.
        if len(selection) < nombre:
            restants = [e for e in population if e not in selection]
            selection.extend(
                alea.sample(restants, min(nombre - len(selection), len(restants)))
            )
        return selection[:nombre]

    echantillon = tirer(repondables, cible_avec, True) + tirer(
        sans_reponse, cible_sans, True
    )

    logger.info(
        "Échantillon : %d questions (%d répondables, %d sans réponse), seed=%d",
        len(echantillon),
        sum(1 for e in echantillon if e.answerable),
        sum(1 for e in echantillon if not e.answerable),
        seed,
    )
    return sorted(echantillon, key=lambda e: e.id)


# ===========================================================================
# CLI
# ===========================================================================


def construire_parseur() -> argparse.ArgumentParser:
    parseur = argparse.ArgumentParser(
        description="Prépare un benchmark d'évaluation du socle RAG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parseur.add_argument(
        "--benchmark",
        choices=("uda", "multihop"),
        default="uda",
        help="benchmark à préparer (défaut : uda)",
    )
    parseur.add_argument(
        "--sous-jeux",
        nargs="+",
        default=list(SOUS_JEUX_UDA_PAR_DEFAUT),
        help="sous-jeux UDA (défaut : paper_text paper_tab)",
    )
    parseur.add_argument(
        "--telecharger",
        action="store_true",
        help="télécharge le dataset avant normalisation",
    )
    parseur.add_argument(
        "--racine-dataset",
        type=Path,
        default=None,
        help="dossier du dataset téléchargé (défaut : evaluation/data/<benchmark>)",
    )
    parseur.add_argument(
        "--corpus",
        type=Path,
        default=DOSSIER_DONNEES / "corpus",
        help="dossier des documents à ingérer",
    )
    parseur.add_argument(
        "--sortie",
        type=Path,
        default=None,
        help="chemin du JSONL produit (défaut : evaluation/data/<nom>.jsonl)",
    )
    parseur.add_argument(
        "--nom",
        default=None,
        help="nom du jeu produit (défaut : <benchmark>_<taille>)",
    )
    parseur.add_argument(
        "--taille-echantillon",
        type=int,
        default=400,
        help="0 = tout le benchmark (défaut : 400)",
    )
    parseur.add_argument(
        "--part-sans-reponse",
        type=float,
        default=0.20,
        help="proportion visée de questions sans réponse (défaut : 0.20)",
    )
    parseur.add_argument("--seed", type=int, default=20240601)
    parseur.add_argument(
        "--sans-corpus",
        action="store_true",
        help="ne copie pas les documents (annotations seules)",
    )
    parseur.add_argument("--verbose", action="store_true")
    return parseur


def main(argv: list[str] | None = None) -> int:
    arguments = construire_parseur().parse_args(argv)
    configurer_logs(arguments.verbose)

    racine = arguments.racine_dataset or (DOSSIER_DONNEES / arguments.benchmark)

    # --- Téléchargement -----------------------------------------------------
    if arguments.telecharger:
        if arguments.benchmark == "uda":
            telecharger_uda(racine, arguments.sous_jeux)
        else:
            telecharger_multihop(racine)

    if not racine.exists():
        logger.error(
            "Dossier %s introuvable. Relance avec --telecharger.", racine
        )
        return 1

    # --- Normalisation ------------------------------------------------------
    if arguments.benchmark == "uda":
        enregistrements = normaliser_uda(racine, arguments.sous_jeux)
    else:
        enregistrements = normaliser_multihop(racine)

    if not enregistrements:
        logger.error(
            "Aucun enregistrement normalisé. Le téléchargement est-il complet ?"
        )
        return 1

    logger.info("Annotations normalisées : %d questions.", len(enregistrements))

    # --- Échantillonnage ----------------------------------------------------
    echantillon = echantillonner(
        enregistrements,
        arguments.taille_echantillon,
        arguments.seed,
        part_sans_reponse=arguments.part_sans_reponse,
    )

    nom = arguments.nom or (
        f"{arguments.benchmark}_{len(echantillon)}"
    )
    sortie = arguments.sortie or (DOSSIER_DONNEES / f"{nom}.jsonl")
    ecrire_jsonl(sortie, echantillon)
    logger.info("Jeu écrit : %s (%d questions)", sortie, len(echantillon))

    # --- Corpus -------------------------------------------------------------
    if not arguments.sans_corpus and arguments.benchmark == "uda":
        preparer_corpus_uda(racine, echantillon, arguments.corpus)
        logger.info(
            "Étape suivante — ingère le corpus avec le pipeline existant :\n"
            "    python -m src.rag.ingestion --dossier %s",
            arguments.corpus,
        )
    elif arguments.benchmark == "multihop":
        logger.info(
            "MultiHop-RAG : le corpus est un JSON d'articles. "
            "Conversion en fichiers .md non effectuée à ce stade "
            "(benchmark réservé à la couche agentique)."
        )

    # --- Récapitulatif ------------------------------------------------------
    documents = {
        nom_document(e.expected_document)
        for e in echantillon
        if e.expected_document
    }
    avec_evidence = sum(1 for e in echantillon if e.a_une_evidence)

    logger.info(
        "Récapitulatif : %d questions | %d documents | %d avec evidence | "
        "%d sans réponse",
        len(echantillon),
        len(documents),
        avec_evidence,
        sum(1 for e in echantillon if not e.answerable),
    )

    if avec_evidence == 0:
        logger.warning(
            "Aucune evidence exploitable : evaluate_retrieval_evidence.py "
            "ne pourra rien mesurer. Vérifie que extended_qa_info a bien été "
            "téléchargé."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())