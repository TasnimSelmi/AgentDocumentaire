"""
Modèle du profil de domaine.

Un profil de domaine décrit **le champ métier** dans lequel le RAG travaille :
son intitulé, une description générique et les concepts importants du domaine.

Il ne décrit jamais un corpus, un document, une catégorie documentaire ni un
schéma d'extraction. Aucun document n'est lu pour le construire : il est
saisi par un administrateur, éventuellement pré-rempli par un LLM.

Il sera utilisé plus tard comme contexte de vocabulaire dans le prompt de
génération, jamais comme source de faits.
"""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Un nom de profil sert aussi de nom de fichier : identifiant technique,
# minuscules non accentuées, chiffres, tiret et underscore uniquement.
MOTIF_NOM_PROFIL = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}[a-z0-9]$|^[a-z0-9]$")

# Fragments interdits : toute forme de chemin doit être refusée avant
# normalisation, afin d'empêcher une traversée de répertoire.
FRAGMENTS_INTERDITS = ("/", "\\", "..", "\0", ":")

NB_MOTS_CLES_MIN = 3
NB_MOTS_CLES_MAX = 30
LONGUEUR_NOM_MAX = 64

# Étiquette de langue générique (fr, en, ar, fr-tn, pt-br...).
MOTIF_LANGUE = re.compile(r"^[a-z]{2,3}(-[a-z0-9]{2,8})*$")


def normaliser_nom_profil(valeur: str) -> str:
    """
    Transforme une saisie libre en identifiant technique utilisable comme
    nom de fichier, ou lève `ValueError` si la valeur reste invalide.

    Les accents sont translittérés et les espaces deviennent des underscores.
    En revanche, aucun caractère de chemin n'est corrigé silencieusement :
    `../finance` ou `finance/test` sont refusés, jamais réparés.
    """
    if not isinstance(valeur, str):
        raise ValueError("Le nom de profil doit être une chaîne de caractères.")

    brut = valeur.strip()
    if not brut:
        raise ValueError("Le nom de profil ne peut pas être vide.")

    for fragment in FRAGMENTS_INTERDITS:
        if fragment in brut:
            raise ValueError(
                f"Nom de profil invalide : {valeur!r}. "
                "Un nom de profil ne doit contenir aucun élément de chemin "
                "('/', '\\', '..', ':')."
            )
    if brut in {".", ".."}:
        raise ValueError(f"Nom de profil invalide : {valeur!r}.")

    # Translittération des accents : « juridique_tunisién » -> « juridique_tunisien ».
    sans_accents = "".join(
        caractere
        for caractere in unicodedata.normalize("NFKD", brut)
        if not unicodedata.combining(caractere)
    )
    normalise = re.sub(r"\s+", "_", sans_accents).lower()

    if len(normalise) > LONGUEUR_NOM_MAX:
        raise ValueError(
            f"Nom de profil trop long ({len(normalise)} caractères, "
            f"maximum {LONGUEUR_NOM_MAX})."
        )
    if not MOTIF_NOM_PROFIL.match(normalise):
        raise ValueError(
            f"Nom de profil invalide : {valeur!r}. "
            "Caractères autorisés : lettres minuscules non accentuées, "
            "chiffres, tiret et underscore ; le nom doit commencer et finir "
            "par une lettre ou un chiffre."
        )
    return normalise


def normaliser_mots_cles(valeurs: list[str]) -> list[str]:
    """
    Nettoie une liste de mots-clés : suppression des vides, déduplication
    insensible à la casse et aux accents, ordre d'apparition conservé.
    """
    nettoyes: list[str] = []
    vus: set[str] = set()

    for valeur in valeurs:
        if not isinstance(valeur, str):
            raise ValueError("Chaque mot-clé doit être une chaîne de caractères.")
        mot = re.sub(r"\s+", " ", valeur).strip()
        if not mot:
            continue
        empreinte = "".join(
            caractere
            for caractere in unicodedata.normalize("NFKD", mot.casefold())
            if not unicodedata.combining(caractere)
        )
        if empreinte in vus:
            continue
        vus.add(empreinte)
        nettoyes.append(mot)

    return nettoyes


class DomainProfile(BaseModel):
    """
    Profil de domaine validé.

    Attributes:
        profile_name: identifiant technique court, aussi utilisé comme nom de fichier.
        domain: intitulé lisible du domaine métier.
        description: description générique du champ couvert.
        keywords: concepts importants du domaine (vocabulaire, pas des phrases).
        output_language: langue attendue des réponses ("fr" par défaut).
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    profile_name: str = Field(
        ...,
        description="Identifiant technique court, minuscules, sans accent.",
    )
    domain: str = Field(..., description="Intitulé du domaine métier.")
    description: str = Field(..., description="Description générique du domaine.")
    keywords: list[str] = Field(
        ...,
        description="Concepts importants du domaine.",
        min_length=NB_MOTS_CLES_MIN,
        max_length=NB_MOTS_CLES_MAX,
    )
    output_language: str = Field(
        default="fr",
        description="Code de langue des réponses.",
    )

    @field_validator("profile_name", mode="before")
    @classmethod
    def _valider_nom(cls, valeur: object) -> str:
        return normaliser_nom_profil(valeur)  # type: ignore[arg-type]

    @field_validator("domain", "description", mode="after")
    @classmethod
    def _texte_non_vide(cls, valeur: str) -> str:
        texte = re.sub(r"[ \t]+", " ", valeur).strip()
        if not texte:
            raise ValueError("Ce champ est obligatoire et ne peut pas être vide.")
        return texte

    @field_validator("keywords", mode="before")
    @classmethod
    def _nettoyer_mots_cles(cls, valeurs: object) -> object:
        if isinstance(valeurs, list):
            return normaliser_mots_cles(valeurs)  # type: ignore[arg-type]
        return valeurs

    @field_validator("output_language", mode="before")
    @classmethod
    def _valider_langue(cls, valeur: object) -> str:
        if not isinstance(valeur, str):
            raise ValueError("La langue de sortie doit être une chaîne.")
        langue = valeur.strip().lower().replace("_", "-")
        if not langue:
            raise ValueError("La langue de sortie ne peut pas être vide.")
        if not MOTIF_LANGUE.match(langue):
            raise ValueError(
                f"Code de langue invalide : {valeur!r}. "
                "Attendu : un code court tel que 'fr', 'en', 'ar', 'fr-tn'."
            )
        return langue

    def bloc_contexte_domaine(self) -> str:
        """
        Rend le profil sous forme de bloc texte destiné au prompt de génération.

        Le nom est volontairement explicite : ce bloc n'est qu'un fragment de
        contexte métier, à ne pas confondre avec le prompt complet du RAG ni
        avec la méthode homonyme de `ConfigClassification` (src/config.py),
        qui appartient au profil technique.

        Fournit uniquement du contexte de vocabulaire : la méthode ne produit
        aucune affirmation factuelle sur le contenu du corpus.
        """
        concepts = ", ".join(self.keywords)
        return (
            "CONTEXTE DU DOMAINE\n"
            f"Domaine : {self.domain}\n"
            f"Description : {self.description}\n"
            f"Concepts importants : {concepts}"
        )


__all__ = [
    "DomainProfile",
    "normaliser_nom_profil",
    "normaliser_mots_cles",
    "NB_MOTS_CLES_MIN",
    "NB_MOTS_CLES_MAX",
]
