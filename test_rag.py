"""
Test de bout en bout du pipeline RAG.

Ce script vérifie :
1. la disponibilité de la collection Qdrant ;
2. l'exécution du retrieval hybride ;
3. le reranking éventuel ;
4. la génération de la réponse ;
5. la validité des citations et le respect du périmètre documentaire ;
6. la conformité de la réponse à une vérité terrain, lorsqu'elle est fournie ;
7. l'enregistrement d'un rapport JSON.

Deux notions distinctes sont mesurées et ne doivent jamais être confondues :

    Succès technique   le pipeline a produit une réponse sourcée, avec des
                       citations valides et un contexte suffisant ;
    Réponse exacte     cette réponse correspond à la valeur attendue.

Un pipeline peut parfaitement réussir techniquement en donnant un chiffre
faux — c'est exactement ce qui se produit quand une valeur est reprise du
mauvais document. Le statut final n'est donc SUCCÈS que si les deux
conditions sont réunies.

Format du fichier de questions :
    - texte : une question par ligne (aucune vérité terrain) ;
    - JSON ou JSONL : objets {"question": ..., "reponse_attendue": ...,
      "document": ...}. Seul "question" est obligatoire.

Exemples :
    python test_rag.py "What is Bullet Kin?"
    python test_rag.py "Total electricity in 2020?" --attendu 105727.236
    python test_rag.py --questions tests/questions_rag.json
    python test_rag.py "Question" --profil generic --json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import get_profil, get_settings
from src.rag.generation import ErreurGeneration, ReponseRAG, generer_reponse
from src.rag.retrieval import ErreurRecherche
from src.rag.vectorstore import fermer_client, info_collection


logger = logging.getLogger(__name__)

# Tolérance relative par défaut pour la comparaison de valeurs numériques.
# Une vérité terrain est rarement reproduite au dernier chiffre près par un
# LLM qui reformule ; une tolérance relative reste néanmoins stricte sur les
# ordres de grandeur.
TOLERANCE_RELATIVE = 1e-6

# Nombre éventuellement muni de séparateurs de milliers et d'une décimale.
_RE_NOMBRE = re.compile(r"[-+]?\d[\d\s.,\u00a0']*\d|[-+]?\d")


# ===========================================================================
# Cas de test
# ===========================================================================


@dataclass
class CasDeTest:
    """Une question, éventuellement accompagnée de sa vérité terrain."""

    question: str
    reponse_attendue: str | None = None
    document_attendu: str | None = None

    @property
    def a_verite_terrain(self) -> bool:
        return self.reponse_attendue is not None and str(self.reponse_attendue).strip() != ""


def charger_questions(chemin: Path) -> list[CasDeTest]:
    """Charge des cas de test au format texte, JSON ou JSONL.

    Le format texte historique reste accepté tel quel : une question par
    ligne, sans vérité terrain. Les formats structurés permettent d'ajouter
    la réponse attendue et le document attendu sans changer d'outil.
    """
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier de questions introuvable : {chemin}")

    contenu = chemin.read_text(encoding="utf-8").strip()
    if not contenu:
        raise ValueError(f"Aucune question exploitable dans {chemin}")

    if chemin.suffix.lower() in {".json", ".jsonl"} or contenu[0] in "[{":
        return _charger_cas_structures(contenu, chemin)

    cas = [
        CasDeTest(question=ligne.strip())
        for ligne in contenu.splitlines()
        if ligne.strip() and not ligne.strip().startswith("#")
    ]
    if not cas:
        raise ValueError(f"Aucune question exploitable dans {chemin}")
    return cas


def _charger_cas_structures(contenu: str, chemin: Path) -> list[CasDeTest]:
    """Lit un tableau JSON ou une suite d'objets JSONL."""
    entrees: list[Any]
    try:
        charge = json.loads(contenu)
        entrees = charge if isinstance(charge, list) else [charge]
    except json.JSONDecodeError:
        entrees = []
        for ligne in contenu.splitlines():
            ligne = ligne.strip()
            if not ligne or ligne.startswith("#"):
                continue
            try:
                entrees.append(json.loads(ligne))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Ligne JSON invalide dans {chemin} : {ligne[:80]}"
                ) from exc

    cas: list[CasDeTest] = []
    for entree in entrees:
        if isinstance(entree, str):
            cas.append(CasDeTest(question=entree))
            continue
        if not isinstance(entree, dict):
            continue

        question = str(
            entree.get("question") or entree.get("query") or ""
        ).strip()
        if not question:
            continue

        attendu = (
            entree.get("reponse_attendue")
            or entree.get("expected_answer")
            or entree.get("attendu")
        )
        document = (
            entree.get("document")
            or entree.get("document_attendu")
            or entree.get("expected_document")
        )
        cas.append(
            CasDeTest(
                question=question,
                reponse_attendue=None if attendu is None else str(attendu),
                document_attendu=None if document is None else str(document),
            )
        )

    if not cas:
        raise ValueError(f"Aucune question exploitable dans {chemin}")
    return cas


# ===========================================================================
# Comparaison à la vérité terrain
# ===========================================================================


def _normaliser(texte: str) -> str:
    """Minuscule, sans accents, sans ponctuation, espaces normalisés."""
    brut = unicodedata.normalize("NFKD", str(texte))
    brut = "".join(car for car in brut if not unicodedata.combining(car))
    return " ".join(re.findall(r"[a-z0-9]+", brut.lower()))


def _en_nombre(fragment: str) -> float | None:
    """Convertit un fragment numérique, quel que soit le séparateur employé.

    Les corpus mélangent « 105 727,236 », « 105,727.236 » et « 105727.236 ».
    Les règles appliquées sont générales, sans hypothèse de locale :

    - deux types de séparateurs présents : le dernier est le séparateur
      décimal, l'autre marque les milliers ;
    - un seul séparateur, répété : il marque les milliers ;
    - un seul séparateur, unique : il marque les milliers uniquement s'il est
      suivi d'exactement trois chiffres *et* précédé d'au plus trois chiffres,
      seul cas où un groupement de milliers est bien formé. « 105727.236 » est
      donc décimal, tandis que « 1,234 » vaut mille deux cent trente-quatre.

    Reste une ambiguïté irréductible sur « 1.234 », lue comme un groupement
    de milliers. Aucune information dans la chaîne ne permet de trancher.
    """
    fragment = fragment.strip().replace("\u00a0", "").replace("'", "")
    fragment = re.sub(r"\s", "", fragment)
    if not fragment or not any(car.isdigit() for car in fragment):
        return None

    signe = -1.0 if fragment.startswith("-") else 1.0
    corps = fragment.lstrip("+-")

    positions = [(i, car) for i, car in enumerate(corps) if car in ".,"]
    if positions:
        indice, separateur = positions[-1]
        avant = corps[:indice]
        apres = corps[indice + 1 :]

        if not apres.isdigit():
            # Séparateur terminal (ponctuation de phrase) : on l'ignore.
            corps = re.sub(r"[.,]", "", avant)
        else:
            types = {car for _, car in positions}
            unique = len(positions) == 1
            groupement_valide = (
                len(apres) == 3 and len(re.sub(r"[.,]", "", avant)) <= 3
            )

            if len(types) > 1:
                decimal = True
            elif not unique:
                decimal = False
            else:
                decimal = not groupement_valide

            if decimal:
                corps = f"{re.sub(r'[.,]', '', avant)}.{apres}"
            else:
                corps = re.sub(r"[.,]", "", corps)

    try:
        return signe * float(corps)
    except ValueError:
        return None


def _nombres_dans(texte: str) -> list[float]:
    """Toutes les valeurs numériques trouvées dans un texte.

    Les marqueurs de citation sont retirés au préalable : l'indice de [S1]
    n'est pas une donnée et ne doit jamais être confondu avec une valeur
    attendue.
    """
    texte = re.sub(r"\[S\d+\]", " ", str(texte))
    valeurs: list[float] = []
    for fragment in _RE_NOMBRE.findall(texte):
        valeur = _en_nombre(fragment)
        if valeur is not None:
            valeurs.append(valeur)
    return valeurs


def _proches(obtenu: float, attendu: float, tolerance: float) -> bool:
    if attendu == 0.0:
        return abs(obtenu) <= tolerance
    return abs(obtenu - attendu) / abs(attendu) <= tolerance


def comparer_reponse(
    reponse: str,
    attendu: str,
    *,
    tolerance: float = TOLERANCE_RELATIVE,
) -> tuple[bool, str]:
    """Compare une réponse libre à une vérité terrain, sans règle métier.

    Deux stratégies, choisies d'après la nature de l'attendu :

    - attendu numérique : la réponse est correcte si l'une des valeurs
      qu'elle contient correspond, à la tolérance relative près. Comparer les
      nombres et non les chaînes évite de rejeter « 105 727,236 MWh » face à
      « 105727.236 » ;
    - attendu textuel : comparaison sur forme normalisée, égalité ou
      inclusion, ce qui tolère une phrase de contexte autour de la réponse.
    """
    reponse = str(reponse or "")
    attendu = str(attendu or "").strip()
    if not attendu:
        return False, "aucune vérité terrain fournie"

    valeur_attendue = _en_nombre(attendu)
    if valeur_attendue is not None:
        candidats = _nombres_dans(reponse)
        if not candidats:
            return False, "aucune valeur numérique trouvée dans la réponse"
        for candidat in candidats:
            if _proches(candidat, valeur_attendue, tolerance):
                return True, f"valeur {candidat} conforme à {valeur_attendue}"
        return (
            False,
            "valeurs trouvées : "
            + ", ".join(str(c) for c in candidats[:8])
            + f" — attendu {valeur_attendue}",
        )

    attendu_normalise = _normaliser(attendu)
    reponse_normalisee = _normaliser(reponse)
    if not attendu_normalise:
        return False, "vérité terrain vide après normalisation"
    if attendu_normalise in reponse_normalisee:
        return True, "réponse attendue présente dans la réponse produite"
    return False, "réponse attendue absente de la réponse produite"


def _document_conforme(resultat: ReponseRAG, document_attendu: str) -> tuple[bool, str]:
    """Vérifie que les sources citées proviennent du document attendu."""
    attendu = _normaliser(document_attendu)
    if not attendu:
        return True, "aucun document attendu"
    if not resultat.sources:
        return False, "aucune source citée"

    observes = []
    for source in resultat.sources:
        for valeur in (source.document, source.nom_fichier, source.source):
            if valeur:
                observes.append(valeur)

    conformes = [
        valeur
        for valeur in observes
        if attendu in _normaliser(valeur) or _normaliser(valeur) in attendu
    ]
    if conformes:
        return True, "sources citées cohérentes avec le document attendu"
    return False, "sources citées : " + ", ".join(sorted(set(observes))[:5])


# ===========================================================================
# Rapport
# ===========================================================================


def tronquer(texte: str, limite: int = 300) -> str:
    texte = " ".join(str(texte).split())
    if len(texte) <= limite:
        return texte
    return texte[: limite - 3] + "..."


def verifier_collection() -> dict[str, Any]:
    """Vérifie que la collection Qdrant existe et contient des points."""
    infos = info_collection()

    if not infos.get("existe"):
        raise RuntimeError(
            "La collection Qdrant n'existe pas. Lance d'abord l'ingestion."
        )

    if not infos.get("points", 0):
        raise RuntimeError(
            "La collection Qdrant est vide. Indexe d'abord des documents."
        )

    return infos


def resultat_vers_rapport(
    resultat: ReponseRAG,
    cas: CasDeTest,
    *,
    tolerance: float = TOLERANCE_RELATIVE,
) -> dict[str, Any]:
    """Transforme le résultat RAG en rapport de test lisible.

    Le champ ``succes`` est délibérément le statut final, et non le succès
    technique : c'est lui qui pilote le code de sortie, et une valeur fausse
    correctement citée doit faire échouer la campagne.
    """
    recherche = resultat.recherche

    passages = []
    if recherche is not None:
        for passage in recherche.passages:
            passages.append(
                {
                    "citation": passage.citation,
                    "rang": passage.rang,
                    "document": passage.libelle_document,
                    "doc_id": passage.doc_id,
                    "source": passage.source,
                    "nom_fichier": passage.nom_fichier,
                    "page": passage.page,
                    "chunk_index": passage.chunk_index,
                    "categorie": passage.categorie,
                    "score_recherche": passage.score_recherche,
                    "score_reranking": passage.score_reranking,
                    "score_final": passage.score_final,
                    "texte": passage.texte,
                }
            )

    perimetre = recherche.perimetre if recherche is not None else None

    succes_technique = bool(
        resultat.contexte_suffisant
        and resultat.citations_valides
        and not resultat.citations_hors_perimetre
        and resultat.reponse.strip()
    )

    # Vérité terrain : évaluée indépendamment du succès technique.
    if cas.a_verite_terrain:
        reponse_exacte, detail_exactitude = comparer_reponse(
            resultat.reponse,
            str(cas.reponse_attendue),
            tolerance=tolerance,
        )
    else:
        reponse_exacte, detail_exactitude = None, "aucune vérité terrain fournie"

    if cas.document_attendu:
        document_conforme, detail_document = _document_conforme(
            resultat, cas.document_attendu
        )
    else:
        document_conforme, detail_document = None, "aucun document attendu"

    statut_final = succes_technique
    if reponse_exacte is not None:
        statut_final = statut_final and reponse_exacte
    if document_conforme is not None:
        statut_final = statut_final and document_conforme

    return {
        "question": resultat.question,
        "reponse": resultat.reponse,
        "profil": resultat.profil,
        "succes": statut_final,
        "succes_technique": succes_technique,
        "verite_terrain": {
            "fournie": cas.a_verite_terrain,
            "reponse_attendue": cas.reponse_attendue,
            "reponse_exacte": reponse_exacte,
            "detail": detail_exactitude,
            "document_attendu": cas.document_attendu,
            "document_conforme": document_conforme,
            "detail_document": detail_document,
        },
        "diagnostic": {
            "contexte_suffisant": resultat.contexte_suffisant,
            "citations_valides": resultat.citations_valides,
            "citations_reparees": resultat.citations_reparees,
            "citations_hors_perimetre": resultat.citations_hors_perimetre,
            "duree_secondes": resultat.duree_secondes,
            "avertissements": resultat.avertissements,
        },
        "perimetre": (
            {
                "statut": perimetre.statut,
                "documents": list(perimetre.libelles),
                "identifiants": list(perimetre.valeurs_filtre),
                "origine": perimetre.origine,
                "score": perimetre.score,
                "annees_publication": list(perimetre.annees_publication),
                "annees_valeur": list(perimetre.annees_valeur),
                "raison": perimetre.raison,
            }
            if perimetre is not None
            else None
        ),
        "recherche": {
            "requete": recherche.requete if recherche else None,
            "candidats_recuperes": (
                recherche.candidats_recuperes if recherche else 0
            ),
            "passages_retenus": len(recherche.passages) if recherche else 0,
            "reranking_utilise": (
                recherche.reranking_utilise if recherche else False
            ),
            "seuil_applique": recherche.seuil_applique if recherche else None,
            "diversification_active": (
                recherche.diversification_active if recherche else None
            ),
            "motif_absence": recherche.motif_absence if recherche else None,
            "duree_secondes": recherche.duree_secondes if recherche else None,
            "passages": passages,
        },
        "sources_citees": [asdict(source) for source in resultat.sources],
    }


def _libelle_tri_etat(valeur: bool | None) -> str:
    if valeur is None:
        return "non évalué"
    return "oui" if valeur else "NON"


def afficher_resultat(index: int, total: int, rapport: dict[str, Any]) -> None:
    """Affiche un diagnostic humain pour une question."""
    diagnostic = rapport["diagnostic"]
    recherche = rapport["recherche"]
    verite = rapport["verite_terrain"]
    perimetre = rapport.get("perimetre")

    print("\n" + "=" * 88)
    print(f"TEST RAG {index}/{total}")
    print("=" * 88)
    print(f"Question : {rapport['question']}")
    print(f"Profil   : {rapport['profil']}")

    print("\nRÉPONSE")
    print(rapport["reponse"])

    print("\nRETRIEVAL")
    print(f"  Candidats Qdrant : {recherche['candidats_recuperes']}")
    print(f"  Passages retenus : {recherche['passages_retenus']}")
    print(f"  Reranking        : {recherche['reranking_utilise']}")
    print(f"  Seuil            : {recherche['seuil_applique']}")
    print(f"  Durée retrieval  : {recherche['duree_secondes']} s")
    if recherche.get("motif_absence"):
        print(f"  Motif d'absence  : {recherche['motif_absence']}")

    if perimetre and perimetre["statut"] != "aucun":
        print("\nPÉRIMÈTRE DOCUMENTAIRE")
        print(f"  Statut           : {perimetre['statut']} ({perimetre['origine']})")
        print(f"  Documents        : {', '.join(perimetre['documents']) or '—'}")
        if perimetre["annees_valeur"]:
            print(
                "  Année(s) valeur  : "
                + ", ".join(str(a) for a in perimetre["annees_valeur"])
            )
        if perimetre["annees_publication"]:
            print(
                "  Année(s) public. : "
                + ", ".join(str(a) for a in perimetre["annees_publication"])
            )

    if recherche["passages"]:
        print("\nPASSAGES RETENUS")
        for passage in recherche["passages"]:
            print(
                f"  [{passage['citation']}] "
                f"score={passage['score_final']:.4f} | "
                f"{passage['document'] or passage['nom_fichier'] or passage['source']}"
                + (
                    f", page {passage['page']}"
                    if passage["page"] is not None
                    else ""
                )
            )
            print(f"      {tronquer(passage['texte'])}")

    print("\nVALIDATION")
    print(f"  Contexte suffisant : {diagnostic['contexte_suffisant']}")
    print(f"  Citations valides  : {diagnostic['citations_valides']}")
    print(f"  Citations réparées : {diagnostic['citations_reparees']}")
    if diagnostic["citations_hors_perimetre"]:
        print(
            "  Hors périmètre     : "
            + ", ".join(f"[{c}]" for c in diagnostic["citations_hors_perimetre"])
        )
    print(f"  Durée totale       : {diagnostic['duree_secondes']} s")

    print("\nÉVALUATION")
    print(f"  Succès technique   : {rapport['succes_technique']}")
    if verite["fournie"]:
        print(f"  Réponse exacte     : {_libelle_tri_etat(verite['reponse_exacte'])}")
        print(f"  Réponse attendue   : {verite['reponse_attendue']}")
        print(f"  Détail comparaison : {verite['detail']}")
    else:
        print("  Réponse exacte     : non évaluée (aucune vérité terrain)")
    if verite["document_attendu"]:
        print(f"  Document attendu   : {verite['document_attendu']}")
        print(
            f"  Document conforme  : {_libelle_tri_etat(verite['document_conforme'])}"
            f"  ({verite['detail_document']})"
        )
    print(f"  STATUT FINAL       : {'SUCCÈS' if rapport['succes'] else 'ÉCHEC'}")

    if diagnostic["avertissements"]:
        print("\nAVERTISSEMENTS")
        for avertissement in diagnostic["avertissements"]:
            print(f"  - {avertissement}")


# ===========================================================================
# Exécution
# ===========================================================================


def executer_test(
    cas: CasDeTest,
    *,
    profil_nom: str | None,
    top_k: int | None,
    candidats: int | None,
    utiliser_reranker: bool,
    appliquer_seuil: bool,
    seuil: float | None,
    contexte: int,
    documents: list[str] | None = None,
    resolution_document: bool = True,
    tolerance: float = TOLERANCE_RELATIVE,
) -> dict[str, Any]:
    """Exécute une question à travers tout le pipeline RAG."""
    profil = get_profil(profil_nom)

    resultat = generer_reponse(
        question=cas.question,
        profil=profil,
        top_k=top_k,
        limite_candidats=candidats,
        utiliser_reranker=utiliser_reranker,
        appliquer_seuil=appliquer_seuil,
        seuil_pertinence=seuil,
        limite_contexte_caracteres=contexte,
        documents=documents,
        resolution_document=resolution_document,
    )

    return resultat_vers_rapport(resultat, cas, tolerance=tolerance)


def construire_parseur() -> argparse.ArgumentParser:
    parseur = argparse.ArgumentParser(
        description="Test de bout en bout du pipeline RAG."
    )
    parseur.add_argument(
        "question",
        nargs="?",
        help="question unique à envoyer au RAG",
    )
    parseur.add_argument(
        "--questions",
        type=Path,
        help="fichier de questions (texte, JSON ou JSONL)",
    )
    parseur.add_argument(
        "--attendu",
        default=None,
        help="réponse attendue pour la question unique (vérité terrain)",
    )
    parseur.add_argument(
        "--document-attendu",
        default=None,
        help="document dont la réponse doit provenir",
    )
    parseur.add_argument(
        "--tolerance",
        type=float,
        default=TOLERANCE_RELATIVE,
        help="tolérance relative pour la comparaison numérique",
    )
    parseur.add_argument("--profil", default=None)
    parseur.add_argument("--top-k", type=int, default=None)
    parseur.add_argument("--candidats", type=int, default=None)
    parseur.add_argument("--seuil", type=float, default=None)
    parseur.add_argument("--sans-reranker", action="store_true")
    parseur.add_argument("--sans-seuil", action="store_true")
    parseur.add_argument(
        "--document",
        action="append",
        default=None,
        dest="documents",
        help="restreint la recherche à un document (répétable)",
    )
    parseur.add_argument(
        "--sans-resolution",
        action="store_true",
        help="désactive la résolution automatique du périmètre documentaire",
    )
    parseur.add_argument("--contexte", type=int, default=16_000)
    parseur.add_argument(
        "--sortie",
        type=Path,
        default=None,
        help="chemin du rapport JSON",
    )
    parseur.add_argument(
        "--json",
        action="store_true",
        help="affiche aussi le rapport JSON complet",
    )
    parseur.add_argument("--verbose", action="store_true")
    return parseur


def main() -> int:
    args = construire_parseur().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    if not args.question and not args.questions:
        print(
            "Erreur : fournis une question ou utilise --questions fichier.txt",
            file=sys.stderr,
        )
        return 2

    cas_de_test: list[CasDeTest] = []
    if args.question:
        cas_de_test.append(
            CasDeTest(
                question=args.question,
                reponse_attendue=args.attendu,
                document_attendu=args.document_attendu,
            )
        )
    if args.questions:
        cas_de_test.extend(charger_questions(args.questions))

    rapports: list[dict[str, Any]] = []
    debut = time.perf_counter()

    try:
        infos_collection = verifier_collection()
        settings = get_settings()

        print("\n" + "=" * 88)
        print("PRÉVÉRIFICATION DU RAG")
        print("=" * 88)
        print(f"Collection Qdrant : {infos_collection}")
        print(f"LLM provider       : {settings.llm_provider}")
        print(f"LLM model          : {settings.llm_model}")
        print(f"Embedding model    : {settings.embedding_model}")
        print(f"Reranker model     : {settings.reranker_model}")
        print(f"Questions          : {len(cas_de_test)}")
        print(
            "Vérités terrain    : "
            f"{sum(1 for cas in cas_de_test if cas.a_verite_terrain)}"
        )

        for index, cas in enumerate(cas_de_test, start=1):
            try:
                rapport = executer_test(
                    cas,
                    profil_nom=args.profil,
                    top_k=args.top_k,
                    candidats=args.candidats,
                    utiliser_reranker=not args.sans_reranker,
                    appliquer_seuil=not args.sans_seuil,
                    seuil=args.seuil,
                    contexte=args.contexte,
                    documents=args.documents,
                    resolution_document=not args.sans_resolution,
                    tolerance=args.tolerance,
                )
            except (ErreurRecherche, ErreurGeneration) as exc:
                logger.exception("Échec du test pour la question : %s", cas.question)
                rapport = {
                    "question": cas.question,
                    "succes": False,
                    "succes_technique": False,
                    "erreur": repr(exc),
                }
            except Exception as exc:  # noqa: BLE001 — un cas ne doit pas arrêter la campagne
                logger.exception("Erreur inattendue sur la question : %s", cas.question)
                rapport = {
                    "question": cas.question,
                    "succes": False,
                    "succes_technique": False,
                    "erreur": repr(exc),
                }

            rapports.append(rapport)

            if "erreur" in rapport:
                print("\n" + "=" * 88)
                print(f"TEST RAG {index}/{len(cas_de_test)} — ÉCHEC")
                print("=" * 88)
                print(f"Question : {cas.question}")
                print(f"Erreur   : {rapport['erreur']}")
            else:
                afficher_resultat(index, len(cas_de_test), rapport)

    finally:
        fermer_client()

    duree_totale = round(time.perf_counter() - debut, 4)
    nb_succes = sum(1 for rapport in rapports if rapport.get("succes"))
    nb_echecs = len(rapports) - nb_succes
    nb_techniques = sum(1 for rapport in rapports if rapport.get("succes_technique"))
    nb_evalues = sum(
        1
        for rapport in rapports
        if rapport.get("verite_terrain", {}).get("fournie")
    )
    nb_exacts = sum(
        1
        for rapport in rapports
        if rapport.get("verite_terrain", {}).get("reponse_exacte")
    )
    # Cas le plus révélateur : le pipeline se déclare satisfait alors que la
    # valeur restituée est fausse. C'est ce compteur qu'il faut surveiller.
    nb_faux_positifs = sum(
        1
        for rapport in rapports
        if rapport.get("succes_technique")
        and rapport.get("verite_terrain", {}).get("reponse_exacte") is False
    )

    rapport_global = {
        "date_execution": datetime.now().isoformat(timespec="seconds"),
        "configuration": {
            "profil": args.profil,
            "top_k": args.top_k,
            "candidats": args.candidats,
            "utiliser_reranker": not args.sans_reranker,
            "appliquer_seuil": not args.sans_seuil,
            "seuil": args.seuil,
            "limite_contexte_caracteres": args.contexte,
            "documents": args.documents,
            "resolution_document": not args.sans_resolution,
            "tolerance": args.tolerance,
        },
        "resume": {
            "questions": len(rapports),
            "succes": nb_succes,
            "echecs": nb_echecs,
            "succes_techniques": nb_techniques,
            "questions_avec_verite_terrain": nb_evalues,
            "reponses_exactes": nb_exacts,
            "faux_positifs": nb_faux_positifs,
            "duree_totale_secondes": duree_totale,
        },
        "resultats": rapports,
    }

    sortie = args.sortie
    if sortie is None:
        sortie = Path("data/logs") / "rapport_test_rag.json"

    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(
        json.dumps(rapport_global, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("RÉSUMÉ GLOBAL")
    print("=" * 88)
    print(f"Questions testées      : {len(rapports)}")
    print(f"Succès techniques      : {nb_techniques}")
    print(f"Avec vérité terrain    : {nb_evalues}")
    print(f"Réponses exactes       : {nb_exacts}")
    if nb_faux_positifs:
        print(
            f"Faux positifs          : {nb_faux_positifs} "
            "(pipeline satisfait, réponse fausse)"
        )
    print(f"Statut final — succès  : {nb_succes}")
    print(f"Statut final — échecs  : {nb_echecs}")
    print(f"Durée totale           : {duree_totale} s")
    print(f"Rapport JSON           : {sortie}")
    print("=" * 88 + "\n")

    if args.json:
        print(json.dumps(rapport_global, ensure_ascii=False, indent=2))

    return 0 if nb_echecs == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())