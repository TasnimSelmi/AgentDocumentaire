"""
Génération assistée d'un profil de domaine, par consensus multi-lots.

Principe : plutôt qu'un seul appel LLM sur un gros échantillon, le corpus
est échantillonné puis découpé en lots analysés indépendamment. Les
propositions sont fusionnées et chaque élément porte le nombre de lots
réussis qui l'ont proposé.

Ce compte est le vrai apport : un champ vu dans 4 lots réussis sur 4 est
structurel, un champ vu dans 1 lot réussi sur 4 est probablement du bruit.
La relecture humaine devient une décision informée plutôt qu'une
appréciation à l'aveugle.

Utilisation :
    python -m src.rag.profiling --nom mondomaine
    python -m src.rag.profiling --nom mondomaine --echantillon 40 --min-lots 2
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from rapidfuzz import fuzz
from unidecode import unidecode

from src.config import (
    DOSSIER_SCHEMAS,
    Profil,
    get_config_technique,
    get_settings,
)
from src.rag.ingestion import construire_llm
from src.rag.loaders import ErreurChargement, charger_document

logger = logging.getLogger(__name__)

NOMS_RESERVES = {
    "doc_id",
    "chunk_index",
    "texte",
    "source",
    "nom_fichier",
    "categorie",
    "page",
    "ocr",
}

TYPES_VALIDES = {
    "texte",
    "nombre",
    "entier",
    "booleen",
    "date",
    "liste[texte]",
    "liste[nombre]",
    "liste[date]",
}
TYPES_TEXTUELS = {"texte", "liste[texte]"}

# Deux noms au-dessus de ce seuil de similarité désignent la même chose.
SEUIL_FUSION = 0.85

# Au-delà, le prompt devient long et la qualité des propositions baisse.
DOCS_PAR_LOT = 12


# ===========================================================================
# 1. Schémas de proposition
# ===========================================================================


class CategorieProposee(BaseModel):
    nom: str = Field(
        ...,
        description="Identifiant court, minuscules, sans accent ni espace.",
    )
    description: str = Field(
        ...,
        description="Ce que contient cette catégorie, en français.",
    )


class ChampPropose(BaseModel):
    nom: str = Field(
        ...,
        description="Identifiant court, minuscules, sans accent ni espace.",
    )
    type: Literal[
        "texte",
        "nombre",
        "entier",
        "booleen",
        "date",
        "liste[texte]",
        "liste[nombre]",
        "liste[date]",
    ]
    description: str = Field(
        ...,
        description=(
            "Instruction destinée au LLM d'extraction : ce qu'il faut chercher, "
            "sous quel format, et quand laisser vide. Deux à quatre phrases."
        ),
    )
    filtrable: bool = Field(
        default=False,
        description=(
            "Vrai si un utilisateur voudrait restreindre une recherche sur ce champ."
        ),
    )
    resoudre_entites: bool = Field(
        default=False,
        description="Vrai pour les noms propres à orthographe variable uniquement.",
    )


class ProfilPropose(BaseModel):
    description_corpus: str = Field(
        ...,
        description="Ce que contient ce corpus, en une ou deux phrases.",
    )
    categories: list[CategorieProposee] = Field(
        ...,
        description="Entre 3 et 7 catégories.",
    )
    champs_metadonnees: list[ChampPropose] = Field(
        ...,
        description="Entre 3 et 7 champs.",
    )
    champs_extraction: list[ChampPropose] = Field(
        ...,
        description="Entre 4 et 8 champs.",
    )


# ===========================================================================
# 2. Échantillonnage adaptatif
# ===========================================================================


def taille_adaptative(
    nb_fichiers: int,
    plancher: int = 15,
    plafond: int = 60,
) -> int:
    """
    Croissance logarithmique, bornée.

    Un corpus dix fois plus gros n'est pas dix fois plus divers : au-delà
    d'une soixantaine de documents, chaque ajout n'apporte quasiment plus
    de type nouveau, seulement du coût et de la redondance.

        50 fichiers   -> 21
        200 fichiers  -> 31
        2 000         -> 48
        10 000        -> 59
    """
    if nb_fichiers <= plancher:
        return nb_fichiers

    estimation = int(plancher + 5 * math.log2(nb_fichiers / 20))
    return max(plancher, min(estimation, plafond))


def echantillonner(
    dossier: Path,
    extensions: list[str],
    taille: int | None,
    graine: int,
) -> tuple[list[Path], int]:
    """
    Sélection stratifiée par extension.

    Un tirage uniforme sur un corpus à 90 % de PDF ne montrerait presque
    jamais les tableurs — dont la structure est pourtant la plus
    différente. Le tour de table garantit la présence de chaque format.
    """
    fichiers = [
        chemin
        for chemin in dossier.rglob("*")
        if chemin.is_file()
        and chemin.suffix.lower() in extensions
        and not chemin.name.startswith("~$")
    ]

    if not fichiers:
        return [], 0

    taille = taille or taille_adaptative(len(fichiers))
    taille = min(taille, len(fichiers))

    rng = random.Random(graine)
    par_extension: dict[str, list[Path]] = defaultdict(list)

    for chemin in fichiers:
        par_extension[chemin.suffix.lower()].append(chemin)

    for liste in par_extension.values():
        rng.shuffle(liste)

    selection: list[Path] = []
    groupes = list(par_extension.values())

    while len(selection) < taille:
        groupes = [groupe for groupe in groupes if groupe]
        if not groupes:
            break

        for groupe in groupes:
            if len(selection) >= taille:
                break
            selection.append(groupe.pop())

    # Évite que les lots soient ordonnés par format.
    rng.shuffle(selection)
    return selection, len(fichiers)


def extraire_apercu(texte: str, limite: int) -> str:
    """
    Extrait un aperçu réparti entre le début, le milieu et la fin.

    Le budget de caractères reste proche de ``limite``, mais le profiler
    observe plusieurs zones du document. Cela réduit le risque de manquer
    les montants, échéances, tableaux, signatures ou conclusions situés
    après l'introduction.
    """
    texte = texte.strip()

    if not texte or limite <= 0:
        return ""

    if len(texte) <= limite:
        return texte

    taille_debut = limite // 3
    taille_milieu = limite // 3
    taille_fin = limite - taille_debut - taille_milieu

    centre = len(texte) // 2
    debut_milieu = max(0, centre - taille_milieu // 2)
    fin_milieu = min(len(texte), debut_milieu + taille_milieu)

    # Si la borne droite a été limitée, on décale la fenêtre vers la gauche
    # afin de conserver autant que possible la taille demandée.
    debut_milieu = max(0, fin_milieu - taille_milieu)

    debut = texte[:taille_debut]
    milieu = texte[debut_milieu:fin_milieu]
    fin = texte[-taille_fin:]

    return (
        "[DÉBUT DU DOCUMENT]\n"
        f"{debut}\n\n"
        "[MILIEU DU DOCUMENT]\n"
        f"{milieu}\n\n"
        "[FIN DU DOCUMENT]\n"
        f"{fin}"
    )


def _lots(
    fichiers: list[Path],
    chars_par_doc: int,
) -> list[list[tuple[str, str]]]:
    """Charge les fichiers et les répartit en lots de taille homogène."""
    extraits: list[tuple[str, str]] = []

    for chemin in fichiers:
        try:
            doc = charger_document(chemin)
        except ErreurChargement as exc:
            logger.warning("Ignoré : %s", exc)
            continue

        if doc.est_vide:
            logger.warning("Ignoré (vide) : %s", chemin.name)
            continue

        apercu = extraire_apercu(
            doc.texte_complet,
            chars_par_doc,
        )

        if not apercu:
            logger.warning("Ignoré (aucun aperçu exploitable) : %s", chemin.name)
            continue

        extraits.append((chemin.name, apercu))

    if not extraits:
        return []

    nb_lots = max(1, math.ceil(len(extraits) / DOCS_PAR_LOT))
    taille_lot = math.ceil(len(extraits) / nb_lots)

    return [
        extraits[i : i + taille_lot]
        for i in range(0, len(extraits), taille_lot)
    ]


# ===========================================================================
# 3. Prompts
# ===========================================================================


_REGLES = f"""CONTRAINTES DE NOMMAGE
- Les `nom` sont des identifiants techniques : minuscules, sans accent,
  sans espace, mots séparés par des underscores. Exemple : date_emission.
- Interdits car réservés : {', '.join(sorted(NOMS_RESERVES))}.
- Les `description` sont en français, destinées à être lues par un LLM.

CHAMPS DE MÉTADONNÉES
- Décrivent le document dans son ensemble, pas son contenu détaillé.
- `filtrable: true` uniquement pour ce sur quoi un utilisateur voudrait
  restreindre une recherche (période, émetteur, référence...).
- `resoudre_entites: true` pour les noms propres à orthographe variable.
  Jamais sur les dates, nombres ou codes normalisés.

CHAMPS D'EXTRACTION
- Entités présentes dans le contenu : montants, échéances, parties,
  obligations, identifiants...

QUALITÉ DES DESCRIPTIONS — le point le plus important
Chaque description est l'instruction que recevra le LLM d'extraction.
Une description vague produit un champ vide dans 60 % des documents.
Précise : ce qu'il faut chercher, le format attendu, les formulations
qui l'introduisent dans le texte, et quand laisser vide.

Mauvais  : "Date du document."
Bon      : "Date d'émission ou de signature, au format AAAA-MM-JJ.
            Généralement en en-tête ou près de la signature. Ne pas
            confondre avec une date d'échéance. Laisser vide si absente."
"""


def _prompt_lot(
    extraits: list[tuple[str, str]],
    index: int,
    total: int,
) -> str:
    corpus = "\n\n".join(
        f"### Document {i} — {nom}\n{texte}"
        for i, (nom, texte) in enumerate(extraits, start=1)
    )

    return f"""Tu conçois le schéma d'indexation d'une base documentaire.
Voici le lot {index}/{total} d'un échantillon du corpus. Propose une
taxonomie et des champs adaptés à CES documents précisément.

Ne propose que ce que ce lot justifie. Si un type n'apparaît qu'une fois,
ne crée pas de catégorie pour lui.

{_REGLES}

DOCUMENTS
{corpus}"""


def _prompt_consolidation(
    squelette: str,
    description_corpus: str,
) -> str:
    return f"""Plusieurs analyses indépendantes d'un même corpus ont produit
les éléments ci-dessous. Le compte entre crochets indique dans combien de
lots réussis chaque élément a été proposé, et les descriptions listées sont
les variantes issues de ces analyses.

Ta tâche : produire la version finale.
- Conserve EXACTEMENT les noms et les types indiqués. N'en ajoute ni
  n'en retire aucun.
- Pour chaque élément, rédige une description unique qui synthétise les
  variantes, en appliquant les règles de qualité ci-dessous.
- Un élément proposé par peu de lots est probablement marginal : rédige
  sa description de façon plus restrictive, en indiquant clairement quand
  laisser le champ vide.

{_REGLES}

DESCRIPTION DU CORPUS
{description_corpus}

ÉLÉMENTS À CONSOLIDER
{squelette}"""


# ===========================================================================
# 4. Fusion des propositions
# ===========================================================================


def _identifiant(brut: str) -> str:
    propre = unidecode(brut).lower().strip()
    propre = re.sub(r"[^a-z0-9]+", "_", propre).strip("_")
    return propre or "champ"


@dataclass
class Element:
    """
    Candidat fusionné, avec une seule voix par lot.

    Les champs ``types_par_lot``, ``filtrable_par_lot`` et
    ``resoudre_par_lot`` empêchent un même lot de compter plusieurs fois si
    le LLM propose deux noms proches ensuite fusionnés par l'accumulateur.
    """

    nom: str
    lots: set[int] = field(default_factory=set)
    descriptions: list[str] = field(default_factory=list)
    types_par_lot: dict[int, str] = field(default_factory=dict)
    filtrable_par_lot: dict[int, bool] = field(default_factory=dict)
    resoudre_par_lot: dict[int, bool] = field(default_factory=dict)

    @property
    def occurrences(self) -> int:
        return len(self.lots)

    @property
    def votes_filtrable(self) -> int:
        return sum(self.filtrable_par_lot.values())

    @property
    def votes_resoudre(self) -> int:
        return sum(self.resoudre_par_lot.values())

    def type_majoritaire(self) -> str:
        if not self.types_par_lot:
            return "texte"

        compteur = Counter(self.types_par_lot.values())
        return compteur.most_common(1)[0][0]

    def enregistrer_type(self, lot: int, type_champ: str) -> None:
        """
        Enregistre au plus un vote de type par lot.

        Si deux propositions fusionnées du même lot ont des types
        différents, le premier type est conservé et le conflit est signalé
        pour ne pas donner deux voix au même lot.
        """
        precedent = self.types_par_lot.get(lot)

        if precedent is None:
            self.types_par_lot[lot] = type_champ
            return

        if precedent != type_champ:
            logger.debug(
                "Conflit de type dans le lot %d pour %s : %s / %s ; "
                "premier type conservé.",
                lot,
                self.nom,
                precedent,
                type_champ,
            )

    def enregistrer_filtrable(self, lot: int, valeur: bool) -> None:
        """Un lot vote une seule fois ; en cas de doublon, ``True`` prévaut."""
        self.filtrable_par_lot[lot] = (
            self.filtrable_par_lot.get(lot, False) or valeur
        )

    def enregistrer_resolution(self, lot: int, valeur: bool) -> None:
        """Un lot vote une seule fois ; en cas de doublon, ``True`` prévaut."""
        self.resoudre_par_lot[lot] = (
            self.resoudre_par_lot.get(lot, False) or valeur
        )


class Accumulateur:
    """
    Fusionne les propositions successives.

    Deux noms au-dessus du seuil de similarité sont considérés comme
    désignant la même chose : « date_emission » et « date_d_emission »
    ne doivent pas produire deux champs.
    """

    def __init__(self, seuil: float = SEUIL_FUSION) -> None:
        self.seuil = seuil
        self._elements: dict[str, Element] = {}

    def _cle(self, nom: str) -> str:
        for existant in self._elements:
            if fuzz.ratio(nom, existant) / 100.0 >= self.seuil:
                return existant
        return nom

    def ajouter(
        self,
        nom: str,
        description: str,
        lot: int,
        type_champ: str | None = None,
        filtrable: bool = False,
        resoudre: bool = False,
    ) -> None:
        cle = self._cle(_identifiant(nom))
        element = self._elements.setdefault(cle, Element(nom=cle))

        element.lots.add(lot)

        description = description.strip()
        if description and description not in element.descriptions:
            element.descriptions.append(description)

        if type_champ:
            element.enregistrer_type(lot, type_champ)

        element.enregistrer_filtrable(lot, filtrable)
        element.enregistrer_resolution(lot, resoudre)

    def elements(self, min_lots: int = 1) -> list[Element]:
        retenus = [
            element
            for element in self._elements.values()
            if element.occurrences >= min_lots
        ]
        return sorted(
            retenus,
            key=lambda element: (-element.occurrences, element.nom),
        )

    def ecartes(self, min_lots: int) -> list[Element]:
        return [
            element
            for element in self._elements.values()
            if element.occurrences < min_lots
        ]


# ===========================================================================
# 5. Assainissement
# ===========================================================================


def _element_vers_champ(element: Element) -> dict[str, Any]:
    nom = element.nom

    if nom in NOMS_RESERVES:
        nom = f"{nom}_doc"
        logger.info("Nom réservé renommé : %s", nom)

    type_champ = element.type_majoritaire()
    if type_champ not in TYPES_VALIDES:
        type_champ = "texte"

    # Vote majoritaire sur les lots ayant réellement vu le champ.
    seuil = element.occurrences / 2

    return {
        "nom": nom,
        "type": type_champ,
        "description": (
            max(element.descriptions, key=len)
            if element.descriptions
            else ""
        ),
        "obligatoire": False,
        "filtrable": element.votes_filtrable > seuil,
        "normaliser": True,
        # Contrainte de config.Champ : réservé aux types textuels.
        "resoudre_entites": (
            element.votes_resoudre > seuil
            and type_champ in TYPES_TEXTUELS
        ),
    }


def _element_vers_categorie(element: Element) -> dict[str, str]:
    return {
        "nom": element.nom,
        "description": (
            max(element.descriptions, key=len)
            if element.descriptions
            else ""
        ),
    }


# ===========================================================================
# 6. Assemblage
# ===========================================================================


def _assembler(
    nom_profil: str,
    description_corpus: str,
    categories: list[dict[str, Any]],
    metadonnees: list[dict[str, Any]],
    extraction: list[dict[str, Any]],
) -> dict[str, Any]:
    noms_categories = {categorie["nom"] for categorie in categories}

    if "autre" not in noms_categories:
        categories.append(
            {
                "nom": "autre",
                "description": (
                    "Tout document ne correspondant à aucune autre catégorie."
                ),
            }
        )

    champs_extraction = [
        {
            cle: valeur
            for cle, valeur in champ.items()
            if cle in {"nom", "type", "description", "obligatoire"}
        }
        for champ in extraction
    ]

    return {
        "profile_name": nom_profil,
        "description": description_corpus.strip(),
        "langue": {
            "langue_sortie": "fr",
            "conserver_langue_source": False,
        },
        "classification": {
            "active": True,
            "multi_etiquette": False,
            "categories": categories,
            "categorie_defaut": "autre",
            "seuil_confiance": 0.55,
        },
        "champs_metadonnees": metadonnees,
        "schema_extraction": {
            "nom": f"Extraction{nom_profil.capitalize()}",
            "description": (
                f"Entités extraites d'un document du corpus « {nom_profil} »."
            ),
            "champs": champs_extraction,
        },
        "resume": {
            "style_defaut": "puces",
            "nb_points_max": 7,
        },
    }


_ENTETE = """# BROUILLON généré automatiquement — à relire avant utilisation.
#
# Le commentaire au-dessus de chaque élément indique dans combien de lots
# d'analyse réussis il a été proposé. Un élément vu dans tous les lots
# réussis est structurel ; un élément vu dans un seul lot est à examiner :
# soit il est marginal et se supprime, soit il est réel mais rare et sa
# description doit préciser quand laisser le champ vide.
#
# À vérifier en priorité :
#   1. Les champs `filtrable: true` correspondent-ils à ce que tu veux
#      pouvoir interroger ? Ce sont eux qui deviennent des index Qdrant.
#   2. Les descriptions sont-elles assez précises ? Elles deviennent
#      littéralement les instructions du LLM d'extraction.
#   3. Manque-t-il une catégorie que l'échantillon n'a pas montrée ?
#
# Renomme ce fichier en <nom>.yaml puis règle ACTIVE_PROFILE dans .env.
"""


def _injecter_frequences(
    texte_yaml: str,
    frequences: dict[str, dict[str, str]],
) -> str:
    """
    Insère un commentaire de confiance au-dessus de chaque ``- nom:``.

    Les sections sont suivies pendant le parcours car un même nom peut
    exister à la fois en métadonnée et en champ d'extraction.
    """
    section = ""
    sortie: list[str] = []

    for ligne in texte_yaml.splitlines():
        if ligne.startswith("classification:"):
            section = "categories"
        elif ligne.startswith("champs_metadonnees:"):
            section = "metadonnees"
        elif ligne.startswith("schema_extraction:"):
            section = "extraction"

        correspondance = re.match(
            r"^(\s*)-\s+nom:\s+['\"]?([^'\"\s]+)['\"]?\s*$",
            ligne,
        )

        if correspondance:
            indentation, nom = correspondance.groups()
            note = frequences.get(section, {}).get(nom)
            if note:
                sortie.append(f"{indentation}# {note}")

        sortie.append(ligne)

    return "\n".join(sortie) + "\n"


# ===========================================================================
# 7. Point d'entrée
# ===========================================================================


def profiler(
    nom_profil: str,
    taille_echantillon: int | None = None,
    chars_par_doc: int = 1500,
    graine: int = 42,
    min_lots: int = 1,
    consolider: bool = True,
) -> tuple[Path, dict[str, Any]]:
    settings = get_settings()
    technique = get_config_technique()

    fichiers, total_corpus = echantillonner(
        settings.documents_dir,
        technique.ingestion.extensions_supportees,
        taille_echantillon,
        graine,
    )

    if not fichiers:
        raise RuntimeError(
            f"Aucun fichier exploitable dans {settings.documents_dir}"
        )

    lots = _lots(fichiers, chars_par_doc)

    if not lots:
        raise RuntimeError(
            "Aucun fichier de l'échantillon n'a pu être lu."
        )

    nb_lots_prepares = len(lots)

    logger.info(
        "Corpus : %d fichiers | échantillon : %d | lots préparés : %d",
        total_corpus,
        len(fichiers),
        nb_lots_prepares,
    )

    llm = construire_llm()

    try:
        llm = llm.bind(max_tokens=4096)
    except Exception:  # noqa: BLE001 — providers sans .bind
        logger.debug("max_tokens non ajustable sur ce provider.")

    acc_categories = Accumulateur()
    acc_metadonnees = Accumulateur()
    acc_extraction = Accumulateur()
    descriptions_corpus: list[str] = []
    lots_reussis: set[int] = set()

    # Un appel par lot préparé.
    for index, lot in enumerate(lots, start=1):
        logger.info(
            "Lot %d/%d (%d documents)…",
            index,
            nb_lots_prepares,
            len(lot),
        )

        try:
            proposition = llm.with_structured_output(ProfilPropose).invoke(
                _prompt_lot(lot, index, nb_lots_prepares)
            )
        except Exception as exc:  # noqa: BLE001 — un lot raté n'annule pas les autres
            logger.warning("Lot %d échoué : %s", index, exc)
            continue

        lots_reussis.add(index)
        descriptions_corpus.append(proposition.description_corpus)

        for categorie in proposition.categories:
            acc_categories.ajouter(
                categorie.nom,
                categorie.description,
                index,
            )

        for champ in proposition.champs_metadonnees:
            acc_metadonnees.ajouter(
                champ.nom,
                champ.description,
                index,
                champ.type,
                champ.filtrable,
                champ.resoudre_entites,
            )

        for champ in proposition.champs_extraction:
            acc_extraction.ajouter(
                champ.nom,
                champ.description,
                index,
                champ.type,
            )

    nb_lots_reussis = len(lots_reussis)
    nb_lots_echoues = nb_lots_prepares - nb_lots_reussis

    if nb_lots_reussis == 0:
        raise RuntimeError(
            "Tous les lots ont échoué ; aucun consensus ne peut être calculé."
        )

    if min_lots > nb_lots_reussis:
        raise RuntimeError(
            f"--min-lots={min_lots} dépasse le nombre de lots réussis "
            f"({nb_lots_reussis})."
        )

    categories = [
        _element_vers_categorie(element)
        for element in acc_categories.elements(min_lots)
    ]
    metadonnees = [
        _element_vers_champ(element)
        for element in acc_metadonnees.elements(min_lots)
    ]
    extraction = [
        _element_vers_champ(element)
        for element in acc_extraction.elements(min_lots)
    ]

    if not categories or not extraction:
        raise RuntimeError(
            "Fusion vide — abaisse --min-lots ou augmente l'échantillon."
        )

    description_corpus = (
        max(descriptions_corpus, key=len)
        if descriptions_corpus
        else ""
    )

    # Consolidation des descriptions.
    consolidation_tentee = False

    if consolider and nb_lots_reussis > 1:
        consolidation_tentee = True
        logger.info(
            "Consolidation des descriptions à partir de %d lots réussis…",
            nb_lots_reussis,
        )

        squelette = _squelette(
            acc_categories,
            acc_metadonnees,
            acc_extraction,
            min_lots,
        )

        try:
            finale = llm.with_structured_output(ProfilPropose).invoke(
                _prompt_consolidation(
                    squelette,
                    description_corpus,
                )
            )

            categories = _reappliquer(
                categories,
                finale.categories,
            )
            metadonnees = _reappliquer(
                metadonnees,
                finale.champs_metadonnees,
            )
            extraction = _reappliquer(
                extraction,
                finale.champs_extraction,
            )
            description_corpus = (
                finale.description_corpus
                or description_corpus
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Consolidation échouée, descriptions brutes conservées : %s",
                exc,
            )

    donnees = _assembler(
        nom_profil,
        description_corpus,
        categories,
        metadonnees,
        extraction,
    )

    # Validation avant écriture : un brouillon invalide n'est jamais produit.
    Profil(**donnees)

    frequences = {
        "categories": _notes(
            acc_categories,
            nb_lots_reussis,
            min_lots,
        ),
        "metadonnees": _notes(
            acc_metadonnees,
            nb_lots_reussis,
            min_lots,
        ),
        "extraction": _notes(
            acc_extraction,
            nb_lots_reussis,
            min_lots,
        ),
    }

    chemin = DOSSIER_SCHEMAS / f"{nom_profil}_brouillon.yaml"
    chemin.parent.mkdir(parents=True, exist_ok=True)

    corps = yaml.safe_dump(
        donnees,
        allow_unicode=True,
        sort_keys=False,
        width=88,
    )

    chemin.write_text(
        _ENTETE
        + "\n"
        + _injecter_frequences(corps, frequences),
        encoding="utf-8",
    )

    resume = {
        "corpus": total_corpus,
        "echantillon": len(fichiers),
        # Clés explicites.
        "lots_prepares": nb_lots_prepares,
        "lots_reussis": nb_lots_reussis,
        "lots_echoues": nb_lots_echoues,
        # Compatibilité avec l'ancien affichage : le dénominateur du
        # consensus est désormais le nombre de lots réussis.
        "lots": nb_lots_reussis,
        "appels_llm": nb_lots_prepares + int(consolidation_tentee),
        "ecartes": {
            "categories": [
                element.nom
                for element in acc_categories.ecartes(min_lots)
            ],
            "metadonnees": [
                element.nom
                for element in acc_metadonnees.ecartes(min_lots)
            ],
            "extraction": [
                element.nom
                for element in acc_extraction.ecartes(min_lots)
            ],
        },
        "consensus": {
            "categories": {
                element.nom: element.occurrences
                for element in acc_categories.elements(min_lots)
            },
            "metadonnees": {
                element.nom: element.occurrences
                for element in acc_metadonnees.elements(min_lots)
            },
        },
    }

    return chemin, resume


def _squelette(
    cat: Accumulateur,
    meta: Accumulateur,
    extr: Accumulateur,
    min_lots: int,
) -> str:
    """Rend les éléments fusionnés sous forme lisible pour la consolidation."""
    blocs: list[str] = []

    for titre, accumulateur, avec_type in (
        ("CATÉGORIES", cat, False),
        ("CHAMPS DE MÉTADONNÉES", meta, True),
        ("CHAMPS D'EXTRACTION", extr, True),
    ):
        lignes = [titre]

        for element in accumulateur.elements(min_lots):
            suffixe = (
                f" [{element.type_majoritaire()}]"
                if avec_type
                else ""
            )
            lignes.append(
                f"- {element.nom}{suffixe} "
                f"[vu dans {element.occurrences} lot(s) réussi(s)]"
            )

            for description in element.descriptions:
                lignes.append(f"    · {description}")

        blocs.append("\n".join(lignes))

    return "\n\n".join(blocs)


def _reappliquer(
    base: list[dict[str, Any]],
    finale: list[Any],
) -> list[dict[str, Any]]:
    """
    Remplace les descriptions par celles de la consolidation.

    Le reste (type, filtrable, résolution d'entités) provient du vote et
    n'est pas réécrit : la consolidation ne doit pas défaire le consensus.
    """
    par_nom = {
        _identifiant(item.nom): item.description.strip()
        for item in finale
    }

    for entree in base:
        nouvelle = par_nom.get(entree["nom"])
        if nouvelle:
            entree["description"] = nouvelle

    return base


def _notes(
    accumulateur: Accumulateur,
    nb_lots_reussis: int,
    min_lots: int,
) -> dict[str, str]:
    notes: dict[str, str] = {}

    for element in accumulateur.elements(min_lots):
        marque = (
            "  <-- à examiner"
            if element.occurrences == 1 and nb_lots_reussis > 1
            else ""
        )
        notes[element.nom] = (
            f"vu dans {element.occurrences}/{nb_lots_reussis} "
            f"lots réussis{marque}"
        )

    return notes


# ===========================================================================
# 8. Interface en ligne de commande
# ===========================================================================


def main() -> None:
    parseur = argparse.ArgumentParser(
        description="Génération assistée de profil"
    )
    parseur.add_argument(
        "--nom",
        required=True,
    )
    parseur.add_argument(
        "--echantillon",
        type=int,
        default=None,
        help="taille forcée ; par défaut, adaptée au corpus",
    )
    parseur.add_argument(
        "--chars",
        type=int,
        default=1500,
        help=(
            "budget de caractères par document, réparti entre début, "
            "milieu et fin"
        ),
    )
    parseur.add_argument(
        "--graine",
        type=int,
        default=42,
    )
    parseur.add_argument(
        "--min-lots",
        type=int,
        default=1,
        help="nombre minimal de lots réussis pour retenir un élément",
    )
    parseur.add_argument(
        "--no-consolidation",
        action="store_true",
        help="ne pas réécrire les descriptions en fin de traitement",
    )
    parseur.add_argument(
        "--verbose",
        action="store_true",
    )
    args = parseur.parse_args()

    logging.basicConfig(
        level=(
            logging.DEBUG
            if args.verbose
            else logging.INFO
        ),
        format="%(levelname)s | %(message)s",
    )

    chemin, resume = profiler(
        nom_profil=args.nom,
        taille_echantillon=args.echantillon,
        chars_par_doc=args.chars,
        graine=args.graine,
        min_lots=args.min_lots,
        consolider=not args.no_consolidation,
    )

    print(f"\n  Corpus          : {resume['corpus']} fichiers")
    print(f"  Échantillon     : {resume['echantillon']} fichiers")
    print(f"  Lots préparés   : {resume['lots_prepares']}")
    print(f"  Lots réussis    : {resume['lots_reussis']}")
    print(f"  Lots échoués    : {resume['lots_echoues']}")
    print(f"  Appels LLM      : {resume['appels_llm']}")

    print("\n  Consensus — catégories")
    for nom, nombre in resume["consensus"]["categories"].items():
        print(
            f"    {nom:<26} "
            f"{nombre}/{resume['lots_reussis']}"
        )

    print("\n  Consensus — métadonnées")
    for nom, nombre in resume["consensus"]["metadonnees"].items():
        print(
            f"    {nom:<26} "
            f"{nombre}/{resume['lots_reussis']}"
        )

    ecartes = {
        cle: valeur
        for cle, valeur in resume["ecartes"].items()
        if valeur
    }

    if ecartes:
        print(
            f"\n  Écartés (< {args.min_lots} lots réussis)"
        )
        for section, noms in ecartes.items():
            print(f"    {section:<26} {noms}")

    print(f"\n  Brouillon : {chemin}")
    print(
        "  Relis-le, renomme-le, puis règle ACTIVE_PROFILE dans .env.\n"
    )


if __name__ == "__main__":
    main()