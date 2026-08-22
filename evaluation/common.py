"""
Socle partagé du harnais d'évaluation.

Ce module existe pour une seule raison : sept scripts d'évaluation doivent
partager exactement la même normalisation lexicale, le même appariement de
documents et le même format d'entrée/sortie. Dupliquer ces fonctions
fausserait silencieusement toute comparaison entre expériences — une
correction appliquée à un seul script suffirait à rendre deux runs
incomparables.

Rien ici n'est spécifique au pipeline : `evaluation/` appelle `src/rag/`,
jamais l'inverse. Aucune fonction de ce module n'est importée par src/.

Vocabulaire employé dans tout le harnais
----------------------------------------
enregistrement   une question normalisée du benchmark (voir Enregistrement)
plafond          part de l'evidence gold réellement présente dans le corpus
                 indexé, compte tenu du parsing. Borne haute atteignable.
atteint          part de l'evidence gold présente dans les passages récupérés
couverture_relative = atteint / plafond
                 la métrique d'evidence à lire. Neutralise les pertes de
                 parsing pour ne mesurer que le retrieval.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

# --- Racine du projet, pour que les scripts soient lançables depuis VS Code
# sans configuration de PYTHONPATH.
RACINE_PROJET = Path(__file__).resolve().parent.parent
if str(RACINE_PROJET) not in sys.path:
    sys.path.insert(0, str(RACINE_PROJET))

DOSSIER_EVALUATION = RACINE_PROJET / "evaluation"
DOSSIER_DONNEES = DOSSIER_EVALUATION / "data"
DOSSIER_RAPPORTS = DOSSIER_EVALUATION / "reports"


def _amorcer_paquet_outils() -> None:
    """
    Contournement d'un import circulaire présent dans le dépôt.

    Le cycle, tel qu'il existe au commit 9e919f9 :

        src/rag/generation.py:48   from src.llm.common import texte_message
          -> src/llm/common.py:21  from src.tools.base import nettoyer_reflexion
               -> déclenche src/tools/__init__.py:12
                    -> src/tools/classify.py:40
                         from src.llm.common import bloc_profil_domaine
                         -> src.llm.common est encore en cours
                            d'initialisation -> ImportError

    Conséquence : importer `src.rag.generation` en premier échoue. C'est
    aussi la raison pour laquelle `import test_rag` échoue actuellement.

    Importer `src.tools` d'abord amorce le paquet complètement et casse le
    cycle. Le contournement est placé ici, dans le harnais, plutôt que dans
    `src/` : corriger le dépôt relève d'une décision explicite, pas d'un
    effet de bord de l'évaluation.

    Correction propre suggérée (non appliquée) : rendre l'import de
    `nettoyer_reflexion` paresseux dans `src/llm/common.py`, c'est-à-dire le
    déplacer à l'intérieur des quatre fonctions qui l'utilisent.
    """
    try:
        import src.tools  # noqa: F401  (amorçage volontaire)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("evaluation").debug(
            "Amorçage de src.tools impossible : %s", exc
        )


_amorcer_paquet_outils()

SEED_PAR_DEFAUT = 20240601

logger = logging.getLogger("evaluation")


# ===========================================================================
# Journalisation et reproductibilité
# ===========================================================================


def configurer_logs(verbeux: bool = False) -> None:
    """Journalisation lisible, identique dans tous les scripts."""
    logging.basicConfig(
        level=logging.DEBUG if verbeux else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Le pipeline est bavard en DEBUG ; on ne le laisse parler qu'en --verbose.
    if not verbeux:
        for nom in ("src", "httpx", "urllib3", "qdrant_client"):
            logging.getLogger(nom).setLevel(logging.WARNING)


def fixer_seed(seed: int = SEED_PAR_DEFAUT) -> random.Random:
    """
    Fixe les générateurs aléatoires et retourne un Random dédié.

    Le Random dédié est préférable au module global : deux scripts lancés
    dans le même processus ne peuvent pas se voler mutuellement des tirages.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # numpy est présent via pandas, mais on ne l'exige pas.
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    return random.Random(seed)


# ===========================================================================
# Format intermédiaire générique
# ===========================================================================


@dataclass
class Enregistrement:
    """
    Une question de benchmark, normalisée.

    Ce format est le contrat entre `prepare_dataset.py` et tous les scripts
    d'évaluation. Changer de benchmark ne doit affecter QUE la préparation :
    c'est ce qui garde le harnais générique.

    Attributes:
        id: identifiant stable, préfixé par le sous-jeu d'origine.
        question: la question telle qu'elle sera envoyée au retriever.
        expected_answer: réponse de référence (chaîne, même si numérique).
        expected_document: nom de fichier attendu dans le corpus indexé,
            tel qu'il apparaîtra dans le payload Qdrant `nom_fichier`.
            None lorsque la question est sans réponse.
        evidence_text: passages gold concaténés. Peut être vide.
        page: page gold si le benchmark la fournit, sinon None.
        answerable: False pour les questions volontairement sans réponse.
            Le système doit alors refuser.
        subset: sous-jeu d'origine (traçabilité, stratification).
        question_type: type déclaré par le benchmark, si disponible.
        answer_variants: réponses alternatives acceptables.
        metadata: tout le reste, conservé sans interprétation.
    """

    id: str
    question: str
    expected_answer: str = ""
    expected_document: str | None = None
    evidence_text: str = ""
    page: int | None = None
    answerable: bool = True
    subset: str = ""
    question_type: str = ""
    answer_variants: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def documents_pertinents(self) -> set[str]:
        """Documents gold, sous la forme attendue par les métriques."""
        if not self.expected_document:
            return set()
        return {Path(str(self.expected_document)).name}

    @property
    def a_une_evidence(self) -> bool:
        return bool(self.evidence_text and self.evidence_text.strip())

    def vers_dict(self) -> dict[str, Any]:
        return asdict(self)


def ecrire_jsonl(chemin: Path, enregistrements: Iterable[Any]) -> int:
    """Écrit des dataclasses ou des dicts en JSONL UTF-8. Retourne le compte."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    compte = 0
    with chemin.open("w", encoding="utf-8") as flux:
        for element in enregistrements:
            donnees = (
                element.vers_dict()
                if hasattr(element, "vers_dict")
                else (asdict(element) if hasattr(element, "__dataclass_fields__") else element)
            )
            flux.write(json.dumps(donnees, ensure_ascii=False) + "\n")
            compte += 1
    return compte


def lire_jsonl(chemin: Path) -> Iterator[dict[str, Any]]:
    """Lit un JSONL en ignorant les lignes vides."""
    if not chemin.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {chemin}. "
            "Lance d'abord evaluation/prepare_dataset.py."
        )
    with chemin.open("r", encoding="utf-8") as flux:
        for numero, ligne in enumerate(flux, start=1):
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                yield json.loads(ligne)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Ligne JSON invalide dans {chemin} (ligne {numero})."
                ) from exc


def charger_enregistrements(
    chemin: Path,
    *,
    limite: int | None = None,
    uniquement_avec_evidence: bool = False,
    uniquement_answerable: bool | None = None,
) -> list[Enregistrement]:
    """
    Charge un JSONL normalisé en objets `Enregistrement`.

    Args:
        limite: tronque après N enregistrements (smoke test).
        uniquement_avec_evidence: pour l'évaluation evidence-level.
        uniquement_answerable: True (questions répondables), False
            (questions sans réponse), None (les deux).
    """
    champs_connus = set(Enregistrement.__dataclass_fields__)
    resultats: list[Enregistrement] = []

    for donnees in lire_jsonl(chemin):
        extra = {k: v for k, v in donnees.items() if k not in champs_connus}
        connus = {k: v for k, v in donnees.items() if k in champs_connus}
        metadata = dict(connus.pop("metadata", {}) or {})
        metadata.update(extra)

        enregistrement = Enregistrement(**connus, metadata=metadata)

        if uniquement_answerable is not None:
            if enregistrement.answerable is not uniquement_answerable:
                continue
        if uniquement_avec_evidence and not enregistrement.a_une_evidence:
            continue

        resultats.append(enregistrement)
        if limite is not None and len(resultats) >= limite:
            break

    return resultats


# ===========================================================================
# Normalisation lexicale
# ===========================================================================

# Séparateurs de milliers : "1 445", "1,445" et "1445" doivent correspondre.
_MILLIERS = re.compile(r"(?<=\d)[\s\u00a0,](?=\d{3}\b)")
_JETON = re.compile(r"[a-z0-9]+")


def normaliser_texte(valeur: Any) -> str:
    """
    Minuscule, sans accents, séparateurs de milliers retirés.

    Normalisation de MESURE uniquement. Elle n'est jamais appliquée au
    pipeline, seulement à la comparaison entre evidence gold et texte des
    chunks. Volontairement légère : une normalisation agressive (racinisation,
    mots vides) gonflerait artificiellement les recouvrements.
    """
    texte = str(valeur)
    texte = unicodedata.normalize("NFKD", texte)
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    texte = texte.lower()
    texte = _MILLIERS.sub("", texte)
    return texte


def jetons(valeur: Any) -> list[str]:
    """Jetons alphanumériques, dans l'ordre d'apparition."""
    return _JETON.findall(normaliser_texte(valeur))


def ensemble_jetons(valeur: Any) -> set[str]:
    return set(jetons(valeur))


def ngrammes(valeur: Any, n: int = 3) -> set[tuple[str, ...]]:
    """
    N-grammes de jetons.

    Plus exigeants que les jetons isolés : deux textes peuvent partager
    beaucoup de mots fréquents sans partager la moindre formulation. Les
    n-grammes servent donc de contrôle contre les recouvrements fortuits.
    """
    suite = jetons(valeur)
    if len(suite) < n:
        return {tuple(suite)} if suite else set()
    return {tuple(suite[i : i + n]) for i in range(len(suite) - n + 1)}


# ===========================================================================
# Appariement de documents
# ===========================================================================


def nom_document(valeur: Any) -> str:
    """Nom de fichier nu, sans dossier. Casse préservée."""
    texte = str(valeur).strip()
    return Path(texte).name if texte else ""


# Extensions réellement retirées lors de l'appariement. La liste est
# explicite parce que Path().stem est dangereux ici : le doc_name d'UDA
# "1912.01214" (identifiant arXiv) deviendrait "1912", et l'appariement
# confondrait tous les articles publiés le même mois.
EXTENSIONS_DOCUMENT = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".tsv",
    ".pptx", ".ppt", ".html", ".htm", ".txt", ".md", ".json",
}


def cle_document(valeur: Any) -> str:
    """
    Clé d'appariement tolérante : sans extension, sans ponctuation, minuscule.

    Nécessaire parce que le nom gold du benchmark ("1912.01214") et le nom du
    fichier indexé ("1912.01214.pdf") ne coïncident pas exactement.

    Seules les extensions documentaires connues sont retirées, afin que les
    identifiants pointés ("1912.01214") restent intacts.
    """
    base = nom_document(valeur)
    suffixe = Path(base).suffix.lower()
    if suffixe in EXTENSIONS_DOCUMENT:
        base = base[: -len(suffixe)]
    return re.sub(r"[^a-z0-9]+", "", base.lower())


def documents_correspondent(attendu: Any, obtenu: Any) -> bool:
    """Vrai si deux désignations pointent le même document."""
    gauche, droite = cle_document(attendu), cle_document(obtenu)
    return bool(gauche) and gauche == droite


# ===========================================================================
# Couverture d'evidence
# ===========================================================================


@dataclass
class CouvertureEvidence:
    """
    Résultat d'une mesure de couverture d'evidence.

    Attributes:
        plafond: part des jetons gold présents dans le corpus indexé pour le
            document attendu. Borne haute imposée par le parsing.
        atteint: part des jetons gold présents dans les passages récupérés.
        couverture_relative: atteint / plafond. Vaut 1.0 quand le retrieval a
            ramené tout ce qu'il était possible de ramener.
        rang_premier_utile: rang du premier passage contribuant à l'evidence,
            au sens de `Passage.rang`. None si aucun.
        passages_utiles: nombre de passages contribuant effectivement.
        passages_pour_80pct: nombre de passages nécessaires pour atteindre
            80 % du plafond. Mesure directe de l'adéquation du chunking.
        recouvrement_ngrammes: contrôle anti-coïncidence sur les n-grammes.
    """

    plafond: float = 0.0
    atteint: float = 0.0
    couverture_relative: float = 0.0
    rang_premier_utile: int | None = None
    passages_utiles: int = 0
    passages_pour_80pct: int | None = None
    recouvrement_ngrammes: float = 0.0
    jetons_gold: int = 0

    def vers_dict(self) -> dict[str, Any]:
        return asdict(self)


def _proportion(gold: set[str], candidat: set[str]) -> float:
    if not gold:
        return 0.0
    return len(gold & candidat) / len(gold)


def mesurer_couverture(
    evidence: str,
    passages: Sequence[Any],
    *,
    textes_du_document: Sequence[str] = (),
    taille_ngramme: int = 3,
) -> CouvertureEvidence:
    """
    Mesure la couverture d'une evidence gold par des passages récupérés.

    Aucune égalité exacte n'est utilisée : l'evidence gold et les chunks du
    projet proviennent de découpages différents, et une comparaison stricte
    ne mesurerait que cette divergence.

    Args:
        evidence: texte gold.
        passages: objets `Passage` du projet (attributs `texte` et `rang`).
        textes_du_document: tous les textes de chunks du document attendu.
            Sert à calculer le plafond. Si vide, plafond = 1.0 et la
            couverture relative dégénère en couverture absolue.
        taille_ngramme: taille des n-grammes de contrôle.

    Returns:
        Une `CouvertureEvidence`.
    """
    gold = ensemble_jetons(evidence)
    resultat = CouvertureEvidence(jetons_gold=len(gold))

    if not gold:
        return resultat

    # --- Plafond : ce que le corpus indexé contient réellement.
    if textes_du_document:
        union_document: set[str] = set()
        for texte in textes_du_document:
            union_document |= ensemble_jetons(texte)
        resultat.plafond = _proportion(gold, union_document)
    else:
        resultat.plafond = 1.0

    # --- Atteint : contribution cumulée des passages, dans l'ordre du rang.
    ordonnes = sorted(
        passages,
        key=lambda p: getattr(p, "rang", 0) or 0,
    )

    cumul: set[str] = set()
    seuil_80 = 0.8 * resultat.plafond if resultat.plafond > 0 else None

    for passage in ordonnes:
        jetons_passage = ensemble_jetons(getattr(passage, "texte", ""))
        apport = (gold & jetons_passage) - cumul

        if apport:
            resultat.passages_utiles += 1
            if resultat.rang_premier_utile is None:
                resultat.rang_premier_utile = getattr(passage, "rang", None)
            cumul |= apport

            if (
                seuil_80 is not None
                and resultat.passages_pour_80pct is None
                and _proportion(gold, cumul) >= seuil_80
            ):
                resultat.passages_pour_80pct = resultat.passages_utiles

    resultat.atteint = _proportion(gold, cumul)

    # --- Couverture relative : la métrique à lire.
    if resultat.plafond > 0:
        resultat.couverture_relative = min(
            1.0, resultat.atteint / resultat.plafond
        )

    # --- Contrôle n-grammes : recouvrement fortuit de mots fréquents ?
    gold_ngrammes = ngrammes(evidence, taille_ngramme)
    if gold_ngrammes:
        union_ngrammes: set[tuple[str, ...]] = set()
        for passage in ordonnes:
            union_ngrammes |= ngrammes(
                getattr(passage, "texte", ""), taille_ngramme
            )
        resultat.recouvrement_ngrammes = len(
            gold_ngrammes & union_ngrammes
        ) / len(gold_ngrammes)

    return resultat


# ===========================================================================
# Agrégation et écriture des rapports
# ===========================================================================


def moyenne(valeurs: Sequence[float]) -> float:
    valeurs = [v for v in valeurs if v is not None]
    return sum(valeurs) / len(valeurs) if valeurs else 0.0


def percentile(valeurs: Sequence[float], q: float) -> float:
    """Percentile simple, sans dépendance. q dans [0, 1]."""
    propres = sorted(v for v in valeurs if v is not None)
    if not propres:
        return 0.0
    index = min(len(propres) - 1, max(0, int(round(q * (len(propres) - 1)))))
    return propres[index]


def ecrire_rapport(
    dossier: Path,
    nom: str,
    *,
    resume: dict[str, Any],
    details: Sequence[dict[str, Any]],
) -> tuple[Path, Path]:
    """
    Écrit un rapport en JSON (résumé) et CSV (détail question par question).

    Returns:
        Les chemins (json, csv).
    """
    dossier.mkdir(parents=True, exist_ok=True)

    chemin_json = dossier / f"{nom}.json"
    chemin_csv = dossier / f"{nom}.csv"

    chemin_json.write_text(
        json.dumps(
            {"resume": resume, "details": list(details)},
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    if details:
        import pandas as pd  # déjà utilisé par src/rag/evaluate_retrieval.py

        pd.DataFrame(list(details)).to_csv(
            chemin_csv, index=False, encoding="utf-8"
        )
    else:
        chemin_csv.write_text("", encoding="utf-8")

    logger.info("Rapport écrit : %s", chemin_json)
    logger.info("Détail écrit  : %s", chemin_csv)
    return chemin_json, chemin_csv


def horodatage() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


# ===========================================================================
# Accès au corpus indexé (lecture seule)
# ===========================================================================


def attendre_client_qdrant(
    tentatives: int = 15,
    delai_secondes: float = 2.0,
) -> None:
    """
    Acquiert le client Qdrant du projet, avec de nouvelles tentatives en cas
    de verrou transitoire.

    En mode local, Qdrant pose un verrou fichier exclusif (voir
    `src.rag.vectorstore.get_client`) : un seul processus à la fois. Si un
    run d'évaluation démarre pendant qu'un autre processus (une session
    interactive, un run précédent pas encore terminé) tient encore ce
    verrou, `RuntimeError("... already accessed by another instance ...")`
    est levée.

    Sans cette attente, chaque question posée avant la libération du verrou
    échoue individuellement et est comptée comme une erreur de retrieval —
    ce qui a déjà corrompu un run (15 erreurs consécutives, toutes en tête
    de rapport, le temps que l'autre processus se termine). Cette fonction
    absorbe cette attente UNE fois, avant la boucle chronométrée, plutôt que
    de la laisser polluer les métriques question par question.

    Une contention qui persiste au-delà de `tentatives` est une vraie panne
    externe (processus bloqué, deadlock) : elle est alors relevée telle
    quelle, sans être masquée.
    """
    from src.rag.vectorstore import get_client

    derniere_erreur: Exception | None = None
    for tentative in range(1, tentatives + 1):
        try:
            get_client()
            return
        except RuntimeError as exc:
            derniere_erreur = exc
            if "already accessed" not in str(exc):
                raise
            logger.warning(
                "Verrou Qdrant local occupé (tentative %d/%d) : %s",
                tentative,
                tentatives,
                exc,
            )
            time.sleep(delai_secondes)

    raise RuntimeError(
        f"Verrou Qdrant local toujours occupé après {tentatives} tentatives "
        f"({tentatives * delai_secondes:.0f}s). Un autre processus retient "
        "le stockage local — ferme-le avant de relancer l'évaluation."
    ) from derniere_erreur


def charger_textes_par_document(
    limite_chunks: int = 100_000,
) -> dict[str, list[str]]:
    """
    Lit tous les chunks indexés et les regroupe par document.

    Utilisé uniquement pour calculer le plafond d'evidence. Lecture seule :
    aucune écriture dans Qdrant. Réutilise le client du projet plutôt que
    d'en ouvrir un second, ce qui provoquerait un verrou sur le stockage local.
    """
    from src.config import get_config_technique
    from src.rag.vectorstore import get_client

    cfg = get_config_technique()
    client = get_client()
    collection = cfg.qdrant.nom_collection

    textes: dict[str, list[str]] = {}
    decalage = None
    lus = 0

    while lus < limite_chunks:
        points, decalage = client.scroll(
            collection_name=collection,
            limit=min(1000, limite_chunks - lus),
            offset=decalage,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        for point in points:
            charge = point.payload or {}
            cle = cle_document(charge.get("nom_fichier") or charge.get("source") or "")
            if cle:
                textes.setdefault(cle, []).append(charge.get("texte") or "")
            lus += 1

        if decalage is None:
            break

    logger.info(
        "Corpus indexé : %d chunks répartis sur %d documents.",
        lus,
        len(textes),
    )
    return textes


__all__ = [
    "RACINE_PROJET",
    "DOSSIER_DONNEES",
    "DOSSIER_RAPPORTS",
    "SEED_PAR_DEFAUT",
    "Enregistrement",
    "CouvertureEvidence",
    "configurer_logs",
    "fixer_seed",
    "ecrire_jsonl",
    "lire_jsonl",
    "charger_enregistrements",
    "normaliser_texte",
    "jetons",
    "ensemble_jetons",
    "ngrammes",
    "nom_document",
    "cle_document",
    "documents_correspondent",
    "mesurer_couverture",
    "moyenne",
    "percentile",
    "ecrire_rapport",
    "horodatage",
    "attendre_client_qdrant",
    "charger_textes_par_document",
]