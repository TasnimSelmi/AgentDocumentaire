"""
Normalisation des valeurs inférées et résolution d'entités.

Le LLM peut renvoyer des données correctes mais hétérogènes :
"Groupe A", "GRP-A" et "groupe a" peuvent désigner la même chose.
Sans traitement, les filtres structurés risquent de manquer une partie
importante des documents pertinents.

Deux couches :
  1. Normalisation -> mise en forme déterministe par type
                     (texte, date, nombre, entier, booléen)
  2. Résolution    -> fusion des variantes textuelles vers une forme
                     canonique stable et persistée

Aucune dépendance directe au LLM ni à Qdrant : ce module reste testable
unitairement et peut être réutilisé avec n'importe quel corpus.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import re
import tempfile
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Sequence

from dateutil import parser as date_parser
from rapidfuzz import fuzz

from src.config import (
    Champ,
    ConfigNormalisation,
    ConfigResolutionEntites,
    Profil,
    get_config_technique,
    get_profil,
    get_settings,
)

logger = logging.getLogger(__name__)

# Type d'une fonction d'embedding injectée (voir embeddings.py).
# L'injection évite un cycle d'import et permet de tester ce module sans
# charger un modèle d'embedding volumineux.
FonctionEmbedding = Callable[[list[str]], list[list[float]]]


# ===========================================================================
# 1. Normalisation par type
# ===========================================================================

_RE_ESPACES = re.compile(r"\s+")
_RE_PONCTUATION = re.compile(r"[^\w\s-]", flags=re.UNICODE)
_RE_TIRETS = re.compile(r"[-_]+")
_RE_TOKEN_NOMBRE = re.compile(
    r"[+-]?\s*\(?\s*\d(?:[\d\s\u00a0\u202f.,'’]*\d)?"
    r"(?:[eE][+-]?\d+)?\s*\)?"
)
_RE_DATE_MOIS_JOUR_ANNEE = re.compile(
    r"^\s*(\d{1,2})\s+([^\W\d_]+)\.?\s+(\d{4})\s*$",
    flags=re.UNICODE,
)
_RE_DATE_MOIS_ANGLAISE = re.compile(
    r"^\s*([^\W\d_]+)\.?\s+(\d{1,2})(?:er|st|nd|rd|th)?[,]?\s+(\d{4})\s*$",
    flags=re.IGNORECASE | re.UNICODE,
)

_VALEURS_NULLES = {
    "",
    "n/a",
    "na",
    "n.d.",
    "nd",
    "null",
    "none",
    "inconnu",
    "inconnue",
    "non renseigne",
    "non renseignee",
    "non disponible",
    "-",
    "--",
}

_VALEURS_VRAIES = {
    "1",
    "true",
    "vrai",
    "vraie",
    "oui",
    "yes",
    "y",
    "on",
}

_VALEURS_FAUSSES = {
    "0",
    "false",
    "faux",
    "fausse",
    "non",
    "no",
    "n",
    "off",
}

# Les clés sont comparées après normalisation Unicode et suppression des
# diacritiques. Les mois français, anglais et arabes les plus courants sont
# pris en charge sans dépendre de la locale installée sur Windows/Linux.
_MOIS: dict[str, int] = {
    # Français
    "janvier": 1,
    "janv": 1,
    "fevrier": 2,
    "fevr": 2,
    "mars": 3,
    "avril": 4,
    "avr": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "juil": 7,
    "aout": 8,
    "septembre": 9,
    "sept": 9,
    "octobre": 10,
    "oct": 10,
    "novembre": 11,
    "nov": 11,
    "decembre": 12,
    "dec": 12,
    # Anglais
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    # Arabe
    "يناير": 1,
    "فبراير": 2,
    "مارس": 3,
    "أبريل": 4,
    "ابريل": 4,
    "مايو": 5,
    "يونيو": 6,
    "يوليو": 7,
    "أغسطس": 8,
    "اغسطس": 8,
    "سبتمبر": 9,
    "أكتوبر": 10,
    "اكتوبر": 10,
    "نوفمبر": 11,
    "ديسمبر": 12,
}


def _normaliser_unicode(texte: str) -> str:
    """Uniformise les variantes Unicode sans translittérer les alphabets."""
    return unicodedata.normalize("NFKC", texte)


def _retirer_diacritiques(texte: str) -> str:
    """
    Retire les marques diacritiques tout en conservant l'alphabet d'origine.

    Contrairement à une translittération globale, cette méthode transforme
    « Hôtel » en « Hotel » sans convertir un texte arabe en caractères latins.
    """
    decompose = unicodedata.normalize("NFKD", texte)
    sans_diacritiques = "".join(
        caractere
        for caractere in decompose
        if not unicodedata.combining(caractere)
    )
    return unicodedata.normalize("NFC", sans_diacritiques)


def _cle_comparaison(texte: str) -> str:
    """Clé interne insensible à la casse et aux diacritiques."""
    texte = _normaliser_unicode(texte)
    texte = _retirer_diacritiques(texte)
    return texte.casefold().strip()


def _est_valeur_nulle(valeur: Any) -> bool:
    if valeur is None:
        return True
    if not isinstance(valeur, str):
        return False
    return _cle_comparaison(valeur) in _VALEURS_NULLES


def normaliser_texte(valeur: str, cfg: ConfigNormalisation) -> str:
    """
    Produit une forme déterministe d'une chaîne.

    Opérations configurables :
      - normalisation Unicode ;
      - suppression des diacritiques sans translittération des alphabets ;
      - conversion en minuscules ;
      - unification des tirets et underscores ;
      - retrait de la ponctuation ;
      - réduction des espaces.

    Exemple : « Hôtel Al-Safwa  » -> « hotel al safwa ».
    """
    if not valeur:
        return ""

    texte = _normaliser_unicode(str(valeur)).strip()

    if cfg.retirer_accents:
        texte = _retirer_diacritiques(texte)

    if cfg.minuscules:
        texte = texte.casefold()

    texte = _RE_TIRETS.sub(" ", texte)
    texte = _RE_PONCTUATION.sub(" ", texte)

    if cfg.reduire_espaces:
        texte = _RE_ESPACES.sub(" ", texte)

    return texte.strip()


def _jour_avant_mois(locale_date: str) -> bool:
    """Déduit la convention de date depuis la locale configurée."""
    locale = (locale_date or "").replace("_", "-").casefold()

    # Locale américaine : mois/jour/année.
    if locale in {"en-us", "en-us-posix", "us"}:
        return False

    # Par défaut, convention jour/mois adaptée au français, à l'arabe et à
    # la majorité des corpus européens/internationaux du projet.
    return True


def _date_avec_mois_textuel(texte: str) -> dt.date | None:
    """Interprète une date complète contenant un mois écrit en lettres."""
    correspondance = _RE_DATE_MOIS_JOUR_ANNEE.fullmatch(texte)
    if correspondance:
        jour_texte, mois_texte, annee_texte = correspondance.groups()
        mois = _MOIS.get(_cle_comparaison(mois_texte))
        if mois is None:
            return None
        try:
            return dt.date(int(annee_texte), mois, int(jour_texte))
        except ValueError:
            return None

    correspondance = _RE_DATE_MOIS_ANGLAISE.fullmatch(texte)
    if correspondance:
        mois_texte, jour_texte, annee_texte = correspondance.groups()
        mois = _MOIS.get(_cle_comparaison(mois_texte))
        if mois is None:
            return None
        try:
            return dt.date(int(annee_texte), mois, int(jour_texte))
        except ValueError:
            return None

    return None


def normaliser_date(valeur: Any, cfg: ConfigNormalisation) -> str | None:
    """
    Convertit une date complète vers ISO 8601 (AAAA-MM-JJ).

    Ordre de traitement :
      1. objets date/datetime natifs ;
      2. formats explicitement déclarés dans le YAML ;
      3. mois textuels français, anglais ou arabes ;
      4. dateutil en mode strict (fuzzy=False).

    Une chaîne incomplète ou ambiguë non reconnue retourne None au lieu de
    fabriquer une date plausible mais incorrecte.
    """
    if valeur is None:
        return None

    if isinstance(valeur, dt.datetime):
        return valeur.date().isoformat()

    if isinstance(valeur, dt.date):
        return valeur.isoformat()

    texte = _normaliser_unicode(str(valeur)).strip()
    if _est_valeur_nulle(texte):
        return None

    for format_date in cfg.formats_date_essayes:
        try:
            return dt.datetime.strptime(texte, format_date).date().isoformat()
        except (ValueError, TypeError):
            continue

    date_textuelle = _date_avec_mois_textuel(texte)
    if date_textuelle is not None:
        return date_textuelle.isoformat()

    # Évite que dateutil complète silencieusement « 2026 » ou « mars 2026 »
    # avec le mois/jour courants.
    groupes_numeriques = re.findall(r"\d+", texte)
    contient_mois_connu = any(
        _cle_comparaison(mot) in _MOIS
        for mot in re.findall(r"[^\W\d_]+", texte, flags=re.UNICODE)
    )
    if len(groupes_numeriques) < 3 and not (
        contient_mois_connu and len(groupes_numeriques) >= 2
    ):
        logger.debug("Date incomplète ou non reconnue : %r", texte)
        return None

    try:
        resultat = date_parser.parse(
            texte,
            dayfirst=_jour_avant_mois(cfg.locale_date),
            fuzzy=False,
            default=dt.datetime(1900, 1, 1),
        )
        return resultat.date().isoformat()
    except (ValueError, OverflowError, TypeError):
        logger.debug("Date non reconnue : %r", texte)
        return None


def _normaliser_separateurs_nombre(mantisse: str) -> str:
    """
    Convertit une mantisse utilisant des séparateurs français ou anglais
    vers une représentation compatible avec Decimal.
    """
    mantisse = mantisse.replace("\u202f", "")
    mantisse = mantisse.replace("\xa0", "")
    mantisse = mantisse.replace(" ", "")
    mantisse = mantisse.replace("'", "").replace("’", "")

    signe = ""
    if mantisse.startswith(("+", "-")):
        signe, mantisse = mantisse[0], mantisse[1:]

    contient_virgule = "," in mantisse
    contient_point = "." in mantisse

    if contient_virgule and contient_point:
        # Lorsque les deux existent, le dernier séparateur est généralement
        # le séparateur décimal : 1.234,56 ou 1,234.56.
        separateur_decimal = (
            "," if mantisse.rfind(",") > mantisse.rfind(".") else "."
        )
        separateur_milliers = "." if separateur_decimal == "," else ","
        mantisse = mantisse.replace(separateur_milliers, "")
        gauche, droite = mantisse.rsplit(separateur_decimal, 1)
        mantisse = f"{gauche.replace(separateur_decimal, '')}.{droite}"

    elif contient_virgule or contient_point:
        separateur = "," if contient_virgule else "."
        parties = mantisse.split(separateur)

        if len(parties) == 2:
            gauche, droite = parties

            # Un seul séparateur suivi de 1 ou 2 chiffres est décimal.
            # Avec 3 chiffres, il est généralement un séparateur de milliers,
            # sauf pour les valeurs comprises entre -1 et 1 (0,125).
            if len(droite) in {1, 2} or (
                gauche.lstrip("0") == "" and len(droite) >= 1
            ):
                mantisse = f"{gauche}.{droite}"
            elif len(droite) == 3:
                mantisse = f"{gauche}{droite}"
            else:
                # Plus de trois décimales : vraisemblablement une mesure
                # scientifique plutôt qu'un groupement de milliers.
                mantisse = f"{gauche}.{droite}"

        else:
            groupes_apres_premier = parties[1:]

            # Groupements occidentaux ou indiens : 1,234,567 / 12,34,567.
            ressemble_groupement = (
                len(parties[-1]) == 3
                and all(1 <= len(partie) <= 3 for partie in groupes_apres_premier)
            )

            if ressemble_groupement:
                mantisse = "".join(parties)
            else:
                # Dernier séparateur considéré comme décimal, les précédents
                # étant des séparateurs de milliers.
                mantisse = f"{''.join(parties[:-1])}.{parties[-1]}"

    return f"{signe}{mantisse}"


def _decimal_depuis_valeur(valeur: Any) -> Decimal | None:
    """Interprète une valeur numérique sans perte avant conversion finale."""
    if valeur is None or isinstance(valeur, bool):
        return None

    if isinstance(valeur, Decimal):
        return valeur if valeur.is_finite() else None

    if isinstance(valeur, int):
        return Decimal(valeur)

    if isinstance(valeur, float):
        if not math.isfinite(valeur):
            return None
        return Decimal(str(valeur))

    texte = _normaliser_unicode(str(valeur)).strip()
    if _est_valeur_nulle(texte):
        return None

    # Normalise les signes moins Unicode courants.
    texte = texte.replace("−", "-").replace("–", "-").replace("—", "-")

    correspondance = _RE_TOKEN_NOMBRE.search(texte)
    if correspondance is None:
        return None

    token = correspondance.group().strip()
    negatif_parentheses = token.startswith("(") and token.endswith(")")
    token = token.strip("() ")

    # Sépare éventuellement la notation scientifique.
    expo = ""
    correspondance_expo = re.search(r"([eE][+-]?\d+)$", token)
    if correspondance_expo:
        expo = correspondance_expo.group(1)
        token = token[: correspondance_expo.start()]

    token = _normaliser_separateurs_nombre(token)

    if negatif_parentheses and not token.startswith("-"):
        token = f"-{token}"

    try:
        resultat = Decimal(f"{token}{expo}")
    except (InvalidOperation, ValueError):
        return None

    return resultat if resultat.is_finite() else None


def normaliser_nombre(valeur: Any) -> float | None:
    """
    Convertit une valeur numérique vers float.

    Exemples pris en charge :
      - « 4 500,00 € » -> 4500.0
      - « 4.500,00 »   -> 4500.0
      - « 4,500.00 »   -> 4500.0
      - « (1 250,5) »  -> -1250.5
    """
    resultat = _decimal_depuis_valeur(valeur)
    return float(resultat) if resultat is not None else None


def normaliser_entier(valeur: Any) -> int | None:
    """
    Convertit une valeur vers int uniquement lorsqu'elle représente un
    entier exact. Une valeur décimale n'est jamais tronquée silencieusement.
    """
    resultat = _decimal_depuis_valeur(valeur)
    if resultat is None:
        return None

    if resultat != resultat.to_integral_value():
        logger.debug("Valeur non entière refusée : %r", valeur)
        return None

    return int(resultat)


def normaliser_booleen(valeur: Any) -> bool | None:
    """
    Convertit une représentation booléenne explicite en bool.

    Une valeur inconnue retourne None au lieu d'être transformée
    arbitrairement en True parce qu'elle est non vide.
    """
    if valeur is None:
        return None

    if isinstance(valeur, bool):
        return valeur

    if isinstance(valeur, (int, float, Decimal)):
        if valeur == 1:
            return True
        if valeur == 0:
            return False
        return None

    texte = _cle_comparaison(str(valeur))

    if texte in _VALEURS_VRAIES:
        return True

    if texte in _VALEURS_FAUSSES:
        return False

    logger.debug("Valeur booléenne non reconnue : %r", valeur)
    return None


def _type_scalaire(type_champ: str) -> str:
    """Extrait le type élémentaire d'un type simple ou liste[type]."""
    if type_champ.startswith("liste[") and type_champ.endswith("]"):
        return type_champ[6:-1]
    return type_champ


def _dedoublonner(valeurs: list[Any]) -> list[Any]:
    """Dédoublonne en conservant l'ordre, y compris pour des listes imbriquées."""
    resultat: list[Any] = []
    cles_vues: set[str] = set()

    for valeur in valeurs:
        try:
            cle = json.dumps(
                valeur,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        except TypeError:
            cle = repr(valeur)

        if cle not in cles_vues:
            cles_vues.add(cle)
            resultat.append(valeur)

    return resultat


def normaliser_valeur(
    valeur: Any,
    champ: Champ,
    cfg: ConfigNormalisation,
) -> Any:
    """
    Aiguille la valeur vers le normaliseur correspondant au type YAML.

    Les listes sont normalisées élément par élément, débarrassées des
    valeurs vides puis dédoublonnées en conservant l'ordre d'apparition.
    """
    if valeur is None:
        return None

    if champ.est_liste():
        elements = valeur if isinstance(valeur, (list, tuple, set)) else [valeur]
        type_element = _type_scalaire(champ.type)
        sortie: list[Any] = []

        for element in elements:
            normalise = _normaliser_scalaire(element, type_element, cfg)
            if normalise not in (None, ""):
                sortie.append(normalise)

        sortie = _dedoublonner(sortie)
        return sortie or None

    return _normaliser_scalaire(valeur, champ.type, cfg)


def _normaliser_scalaire(
    valeur: Any,
    type_champ: str,
    cfg: ConfigNormalisation,
) -> Any:
    type_element = _type_scalaire(type_champ)

    if type_element == "date":
        return normaliser_date(valeur, cfg)

    if type_element == "nombre":
        return normaliser_nombre(valeur)

    if type_element == "entier":
        return normaliser_entier(valeur)

    if type_element == "booleen":
        return normaliser_booleen(valeur)

    if type_element == "texte":
        return normaliser_texte(str(valeur), cfg)

    raise ValueError(f"Type de champ non pris en charge : {type_champ!r}")


# ===========================================================================
# 2. Résolution d'entités
# ===========================================================================


class RegistreEntites:
    """
    Mémorise, par champ, une forme canonique stable et ses alias.

    Structure persistée dans data/vectordb/entities.json :
        {
          "fournisseur": {
            "hotel al safwa": {
              "alias": ["al safwa hotel", "safwa"],
              "freq": 42
            }
          }
        }

    La première forme normalisée enregistrée devient la clé canonique.
    Cette clé reste stable afin de garantir la stabilité des filtres Qdrant.
    Le champ « freq » compte les occurrences totales du groupe canonique,
    toutes variantes confondues.
    """

    def __init__(
        self,
        chemin: Path,
        cfg: ConfigResolutionEntites,
        fonction_embedding: FonctionEmbedding | None = None,
    ) -> None:
        self.chemin = chemin
        self.cfg = cfg
        self.fonction_embedding = fonction_embedding
        self._donnees: dict[str, dict[str, dict[str, Any]]] = {}
        self._cache_vecteurs: dict[str, list[float]] = {}
        self.charger()

    # -- Persistance --------------------------------------------------------

    @staticmethod
    def _valider_structure(
        donnees: Any,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        if not isinstance(donnees, dict):
            raise ValueError("Le registre d'entités doit être un objet JSON.")

        validees: dict[str, dict[str, dict[str, Any]]] = {}

        for champ, valeurs in donnees.items():
            if not isinstance(champ, str) or not isinstance(valeurs, dict):
                raise ValueError("Structure de registre d'entités invalide.")

            champ_valide: dict[str, dict[str, Any]] = {}

            for canonique, info in valeurs.items():
                if not isinstance(canonique, str) or not isinstance(info, dict):
                    raise ValueError(
                        f"Entrée invalide dans le champ {champ!r}."
                    )

                alias_bruts = info.get("alias", [])
                frequence = info.get("freq", 1)

                if not isinstance(alias_bruts, list):
                    raise ValueError(
                        f"La liste d'alias de {canonique!r} est invalide."
                    )

                if (
                    isinstance(frequence, bool)
                    or not isinstance(frequence, int)
                    or frequence < 1
                ):
                    raise ValueError(
                        f"La fréquence de {canonique!r} est invalide."
                    )

                alias = _dedoublonner(
                    [
                        str(valeur).strip()
                        for valeur in alias_bruts
                        if str(valeur).strip()
                        and str(valeur).strip() != canonique
                    ]
                )

                champ_valide[canonique] = {
                    "alias": alias,
                    "freq": frequence,
                }

            validees[champ] = champ_valide

        return validees

    def charger(self) -> None:
        self._cache_vecteurs.clear()

        if not self.chemin.exists():
            self._donnees = {}
            return

        try:
            contenu = self.chemin.read_text(encoding="utf-8").strip()
            if not contenu:
                self._donnees = {}
                return

            donnees = json.loads(contenu)
            self._donnees = self._valider_structure(donnees)

        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(
                f"Impossible de charger le registre d'entités : {self.chemin}"
            ) from exc

    def sauvegarder(self) -> None:
        """Sauvegarde atomique : l'ancien JSON reste intact en cas d'échec."""
        self.chemin.parent.mkdir(parents=True, exist_ok=True)
        contenu = json.dumps(
            self._donnees,
            ensure_ascii=False,
            indent=2,
        )

        chemin_temporaire: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.chemin.parent,
                prefix=f".{self.chemin.name}.",
                suffix=".tmp",
                delete=False,
            ) as fichier_temporaire:
                fichier_temporaire.write(contenu)
                fichier_temporaire.flush()
                os.fsync(fichier_temporaire.fileno())
                chemin_temporaire = Path(fichier_temporaire.name)

            os.replace(chemin_temporaire, self.chemin)

        finally:
            if chemin_temporaire is not None and chemin_temporaire.exists():
                chemin_temporaire.unlink(missing_ok=True)

    # -- Résolution ---------------------------------------------------------

    def resoudre(self, champ: str, valeur: str) -> str:
        """
        Renvoie la forme canonique correspondant à ``valeur``.

        Ordre de recherche :
          1. correspondance exacte avec une canonique ;
          2. correspondance exacte avec un alias ;
          3. similarité Levenshtein sur canoniques et alias ;
          4. similarité d'embedding si elle est activée ;
          5. création d'une nouvelle canonique.
        """
        if valeur is None:
            return valeur

        valeur = str(valeur).strip()
        champ = str(champ).strip()

        if not valeur or not self.cfg.active:
            return valeur

        if len(valeur) < self.cfg.longueur_min_valeur:
            return valeur

        connues = self._donnees.setdefault(champ, {})

        if valeur in connues:
            connues[valeur]["freq"] += 1
            return valeur

        canonique = self._chercher_alias(connues, valeur)
        if canonique is not None:
            self._ajouter_alias(connues, canonique, valeur)
            return canonique

        candidat = self._plus_proche_levenshtein(connues, valeur)

        if candidat is None and self._embedding_disponible():
            candidat = self._plus_proche_embedding(connues, valeur)

        if candidat is not None:
            self._ajouter_alias(connues, candidat, valeur)
            return candidat

        connues[valeur] = {"alias": [], "freq": 1}
        return valeur

    # -- Recherche de correspondance ---------------------------------------

    @staticmethod
    def _chercher_alias(
        connues: dict[str, dict[str, Any]],
        valeur: str,
    ) -> str | None:
        for canonique, info in connues.items():
            if valeur in info.get("alias", []):
                return canonique
        return None

    @staticmethod
    def _variantes(
        connues: dict[str, dict[str, Any]],
    ) -> list[tuple[str, str]]:
        """Retourne des couples (variante, canonique), sans doublon."""
        resultat: list[tuple[str, str]] = []
        deja_vues: set[tuple[str, str]] = set()

        for canonique, info in connues.items():
            for variante in [canonique, *info.get("alias", [])]:
                couple = (variante, canonique)
                if couple not in deja_vues:
                    deja_vues.add(couple)
                    resultat.append(couple)

        return resultat

    def _plus_proche_levenshtein(
        self,
        connues: dict[str, dict[str, Any]],
        valeur: str,
    ) -> str | None:
        meilleur: str | None = None
        meilleur_score = 0.0

        for variante, canonique in self._variantes(connues):
            score = fuzz.ratio(valeur, variante) / 100.0
            if score > meilleur_score:
                meilleur = canonique
                meilleur_score = score

        return (
            meilleur
            if meilleur_score >= self.cfg.seuil_levenshtein
            else None
        )

    def _embedding_disponible(self) -> bool:
        return (
            self.cfg.utiliser_embedding
            and self.fonction_embedding is not None
        )

    def _vecteurs_manquants(self, textes: list[str]) -> None:
        manquants = [
            texte
            for texte in dict.fromkeys(textes)
            if texte not in self._cache_vecteurs
        ]

        if not manquants or self.fonction_embedding is None:
            return

        vecteurs = self.fonction_embedding(manquants)

        if len(vecteurs) != len(manquants):
            raise ValueError(
                "La fonction d'embedding n'a pas renvoyé un vecteur "
                "pour chaque texte demandé."
            )

        for texte, vecteur in zip(manquants, vecteurs, strict=True):
            self._cache_vecteurs[texte] = [float(x) for x in vecteur]

    def _plus_proche_embedding(
        self,
        connues: dict[str, dict[str, Any]],
        valeur: str,
    ) -> str | None:
        variantes = self._variantes(connues)
        if not variantes:
            return None

        textes_cibles = [variante for variante, _ in variantes]

        try:
            self._vecteurs_manquants([*textes_cibles, valeur])
        except Exception as exc:  # l'embedding d'entités reste un enrichissement
            logger.warning(
                "Résolution d'entités par embedding indisponible pour %r : %s",
                valeur,
                exc,
            )
            return None

        vecteur_source = self._cache_vecteurs[valeur]
        meilleur: str | None = None
        meilleur_score = 0.0

        for variante, canonique in variantes:
            score = _cosinus(
                vecteur_source,
                self._cache_vecteurs[variante],
            )
            if score > meilleur_score:
                meilleur = canonique
                meilleur_score = score

        return (
            meilleur
            if meilleur_score >= self.cfg.seuil_embedding
            else None
        )

    # -- Mise à jour --------------------------------------------------------

    @staticmethod
    def _ajouter_alias(
        connues: dict[str, dict[str, Any]],
        canonique: str,
        alias: str,
    ) -> None:
        info = connues[canonique]
        alias = alias.strip()

        if alias and alias != canonique and alias not in info["alias"]:
            info["alias"].append(alias)

        info["freq"] += 1

    # -- Diagnostic ---------------------------------------------------------

    def statistiques(self) -> dict[str, dict[str, int]]:
        return {
            champ: {
                "canoniques": len(valeurs),
                "variantes": sum(
                    len(info.get("alias", []))
                    for info in valeurs.values()
                ),
                "occurrences": sum(
                    int(info.get("freq", 0))
                    for info in valeurs.values()
                ),
            }
            for champ, valeurs in self._donnees.items()
        }

    def valeurs_rares(self, seuil: int = 3) -> dict[str, list[str]]:
        """Canoniques peu fréquentes, candidates à une fusion manquée."""
        if seuil < 1:
            raise ValueError("Le seuil doit être supérieur ou égal à 1.")

        return {
            champ: [
                valeur
                for valeur, info in valeurs.items()
                if int(info.get("freq", 0)) < seuil
            ]
            for champ, valeurs in self._donnees.items()
        }


def _cosinus(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0

    produit = sum(x * y for x, y in zip(a, b, strict=True))
    norme_a = math.sqrt(sum(x * x for x in a))
    norme_b = math.sqrt(sum(y * y for y in b))

    return produit / (norme_a * norme_b) if norme_a and norme_b else 0.0


# ===========================================================================
# 3. Point d'entrée : traitement d'un jeu de métadonnées
# ===========================================================================


def normaliser_metadonnees(
    brutes: dict[str, Any],
    profil: Profil | None = None,
    registre: RegistreEntites | None = None,
) -> dict[str, Any]:
    """
    Applique normalisation puis résolution à un dictionnaire de métadonnées.

    La sortie contient :
      - ``<champ>``      : valeur normalisée utilisée pour les filtres Qdrant ;
      - ``<champ>_brut`` : valeur d'origine conservée pour l'affichage.
    """
    profil = profil or get_profil()
    cfg = get_config_technique().normalisation

    sortie: dict[str, Any] = {}

    for champ in profil.champs_metadonnees:
        valeur = brutes.get(champ.nom)
        if valeur is None:
            continue

        sortie[f"{champ.nom}_brut"] = valeur

        if not champ.normaliser:
            sortie[champ.nom] = valeur
            continue

        normalisee = normaliser_valeur(valeur, champ, cfg)

        if champ.resoudre_entites and registre is not None and normalisee:
            if champ.est_liste():
                resolues = [
                    registre.resoudre(champ.nom, str(element))
                    for element in normalisee
                ]
                normalisee = _dedoublonner(resolues) or None
            else:
                normalisee = registre.resoudre(
                    champ.nom,
                    str(normalisee),
                )

        sortie[champ.nom] = normalisee

    return sortie


def creer_registre(
    fonction_embedding: FonctionEmbedding | None = None,
) -> RegistreEntites:
    """Fabrique le registre d'entités depuis la configuration courante."""
    return RegistreEntites(
        chemin=get_settings().chemin_entites,
        cfg=get_config_technique().resolution_entites,
        fonction_embedding=fonction_embedding,
    )


# ===========================================================================
# 4. Vérification manuelle : python -m src.rag.normalization
# ===========================================================================


if __name__ == "__main__":
    cfg_normalisation = get_config_technique().normalisation

    print("--- Normalisation texte ---")
    for exemple in [
        "Hôtel Al-Safwa  ",
        "GRP_A",
        "Groupe   A",
        "AIR FRANCE",
        "وثيقة رسمية",
    ]:
        print(
            f"  {exemple!r:<24} -> "
            f"{normaliser_texte(exemple, cfg_normalisation)!r}"
        )

    print("\n--- Normalisation dates ---")
    for exemple in [
        "10/03/2026",
        "10 mars 2026",
        "March 10, 2026",
        "2026-03-10",
        "mars 2026",
        "n/a",
    ]:
        print(
            f"  {exemple!r:<24} -> "
            f"{normaliser_date(exemple, cfg_normalisation)!r}"
        )

    print("\n--- Normalisation nombres ---")
    for exemple in [
        "4 500,00 €",
        "4.500,00",
        "4,500.00",
        "1230.50",
        "USD 89",
        "(1 250,5)",
        "aucun",
    ]:
        print(f"  {exemple!r:<24} -> {normaliser_nombre(exemple)!r}")

    print("\n--- Normalisation entiers ---")
    for exemple in ["12", "12,0", "12.5", 8, "aucun"]:
        print(f"  {exemple!r:<24} -> {normaliser_entier(exemple)!r}")

    print("\n--- Normalisation booléens ---")
    for exemple in [True, False, "oui", "non", "true", "false", "1", "0", "?"]:
        print(f"  {exemple!r:<24} -> {normaliser_booleen(exemple)!r}")

    print("\n--- Résolution d'entités (registre temporaire) ---")
    with tempfile.TemporaryDirectory() as dossier_temporaire:
        registre = RegistreEntites(
            chemin=Path(dossier_temporaire) / "entities.json",
            cfg=get_config_technique().resolution_entites,
            fonction_embedding=None,
        )

        for brut in [
            "Hôtel Al Safwa",
            "hotel al-safwa",
            "Hotel Al Safwa ",
            "Hôtel Anwar",
        ]:
            propre = normaliser_texte(brut, cfg_normalisation)
            print(
                f"  {brut!r:<24} -> "
                f"{registre.resoudre('fournisseur', propre)!r}"
            )

        print("\n--- Statistiques ---")
        print(
            json.dumps(
                registre.statistiques(),
                ensure_ascii=False,
                indent=2,
            )
        )