"""
Configuration centrale de l'agent documentaire.

Trois sources, volontairement séparées :
  - Settings (.env)         : secrets, chemins, modèles      -> varie par machine
  - Technique (default.yaml) : chunking, OCR, seuils, Qdrant -> varie par réglage
  - Profil (schemas/*.yaml)  : taxonomie, champs, extraction -> varie par domaine

Aucune logique métier n'est codée en dur ici. Changer de domaine =
déposer un nouveau YAML dans config/schemas/ et ajuster ACTIVE_PROFILE.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import tempfile
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, create_model, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RACINE_PROJET = Path(__file__).resolve().parent.parent
DOSSIER_CONFIG = RACINE_PROJET / "config"
DOSSIER_SCHEMAS = DOSSIER_CONFIG / "schemas"


# ===========================================================================
# 1. Variables d'environnement (.env)
# ===========================================================================

class Settings(BaseSettings):
    """Réglages dépendant de la machine et de l'environnement."""

    model_config = SettingsConfigDict(
        env_file=RACINE_PROJET / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM (Ollama uniquement) ---
    llm_provider: Literal["ollama"] = "ollama"
    llm_model: str = "qwen3:8b"
    llm_base_url: str | None = None
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    llm_num_ctx: int = 16384

    # --- Embeddings / reranking ---
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: Literal["cpu", "cuda"] = "cuda"
    embedding_batch_size: int = 8
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_enabled: bool = True

    # --- Qdrant : environnement et connexion ---
    qdrant_mode: Literal["local", "server"] = "local"
    qdrant_path: Path = Path("data/vectordb")
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # --- Installation OCR ---
    ocr_enabled: bool = True
    ocr_languages: str = "fra+ara+eng"
    tesseract_cmd: str = ""

    # --- Chemins ---
    documents_dir: Path = Path("data/documents")
    logs_dir: Path = Path("data/logs")

    # --- Stockage géré par corpus (upload / import URL) ---
    # `data/corpora/<corpus_id>/documents/` — jamais src/ ni config/. Chemin
    # toujours dérivé côté serveur (src.rag.corpus), jamais fourni par un
    # client. Distinct de `documents_dir` (dossier local partagé historique).
    corpora_dir: Path = Path("data/corpora")

    # --- Profil technique actif (config/schemas/<nom>.yaml) ---
    # Pilote la taxonomie, les métadonnées et le schéma d'extraction utilisés
    # par l'ingestion et le retrieval. Ne pas confondre avec le profil de
    # domaine ci-dessous, qui ne sert qu'au vocabulaire métier.
    active_profile: str = "generic"

    # --- Profil de domaine (src/profiling) ---
    domain_profiles_dir: Path = Path("profiles/domains")
    active_domain_profile: str | None = None
    domain_profile_output_language: str = "fr"

    # --- Journalisation ---
    log_level: str = "INFO"

    # --- Observabilité (P2.4) ---
    # Couche de traçage transverse (corrélation HTTP + événements agent /
    # ingestion). Générique, sans effet sur le comportement métier.
    observability_enabled: bool = True
    observability_emit_start: bool = True

    @field_validator(
        "documents_dir",
        "logs_dir",
        "qdrant_path",
        "domain_profiles_dir",
        "corpora_dir",
        mode="after",
    )
    @classmethod
    def _chemin_absolu(cls, v: Path) -> Path:
        """Transforme les chemins relatifs en chemins absolus."""
        return v if v.is_absolute() else RACINE_PROJET / v

    def creer_dossiers(self) -> None:
        """Crée les dossiers nécessaires au fonctionnement local."""
        dossiers = {
            self.documents_dir,
            self.logs_dir,
            self.domain_profiles_dir,
        }

        if self.qdrant_mode == "local":
            dossiers.add(self.qdrant_path)

        for dossier in dossiers:
            dossier.mkdir(parents=True, exist_ok=True)

    @property
    def chemin_registre(self) -> Path:
        """Registre des fichiers déjà indexés."""
        return self.qdrant_path / "registry.json"

    @property
    def chemin_entites(self) -> Path:
        """Registre des entités canoniques et de leurs alias."""
        return self.qdrant_path / "entities.json"


# ===========================================================================
# 2. Paramètres techniques (config/default.yaml)
# ===========================================================================

class ConfigTables(BaseModel):
    actif: bool = True
    conserver_entete: bool = True
    lignes_par_chunk: int = Field(default=12, ge=1)
    recouvrement_lignes: int = Field(default=2, ge=0)
    lignes_min: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def _recouvrement_lignes_coherent(self) -> "ConfigTables":
        if self.recouvrement_lignes >= self.lignes_par_chunk:
            raise ValueError(
                "recouvrement_lignes doit être inférieur à lignes_par_chunk."
            )
        return self


class ConfigParentChild(BaseModel):
    actif: bool = True
    taille_parent: int = Field(default=5000, ge=1)


class ConfigVoisins(BaseModel):
    actif: bool = True
    rayon: int = Field(default=1, ge=0)
    max_chunks_ajoutes: int = Field(default=6, ge=0)
    taille_max_contexte: int = Field(default=12000, ge=1)

class ConfigDecoupage(BaseModel):
    strategie: Literal[
        "recursive",
        "semantic",
        "structure_aware",
    ] = "recursive"

    taille_chunk: int = 1000
    recouvrement: int = 150

    separateurs: list[str] = Field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )

    tables: ConfigTables = Field(default_factory=ConfigTables)
    parent_child: ConfigParentChild = Field(default_factory=ConfigParentChild)
    voisins: ConfigVoisins = Field(default_factory=ConfigVoisins)

    @model_validator(mode="after")
    def _recouvrement_coherent(self) -> ConfigDecoupage:
        if self.recouvrement >= self.taille_chunk:
            raise ValueError(
                "Le recouvrement doit être inférieur à la taille du chunk."
            )
        return self


class ConfigIngestion(BaseModel):
    extensions_supportees: list[str] = Field(
        default_factory=lambda: [".pdf", ".docx", ".txt", ".md", ".xlsx", ".csv"]
    )
    taille_max_mo: int = 50
    inferer_metadonnees: bool = True
    algo_hash: str = "sha256"
    chars_pour_inference: int = 3000
    seuil_texte_vide: int = 100


class ConfigOCR(BaseModel):
    active: bool = True
    langues: str = "fra+ara+eng"
    dpi: int = 300
    pages_max: int = 20


class ConfigNormalisation(BaseModel):
    minuscules: bool = True
    retirer_accents: bool = True
    reduire_espaces: bool = True
    formats_date_essayes: list[str] = Field(
        default_factory=lambda: ["%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %B %Y"]
    )
    locale_date: str = "fr"


class ConfigResolutionEntites(BaseModel):
    active: bool = True
    seuil_levenshtein: float = Field(default=0.88, ge=0.0, le=1.0)
    seuil_embedding: float = Field(default=0.92, ge=0.0, le=1.0)
    utiliser_embedding: bool = True
    longueur_min_valeur: int = 3


class ConfigRecherche(BaseModel):
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_final: int = 6
    fusion: Literal["rrf", "alpha"] = "rrf"
    rrf_k: int = 60
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    score_min: float = 0.30


class ConfigQdrant(BaseModel):
    nom_collection: str = "documents"
    distance: Literal["cosine", "euclid", "dot"] = "cosine"
    taille_vecteur_dense: int = 1024        # BGE-M3
    sparse_active: bool = True
    taille_lot_upsert: int = 64


class ConfigAgent(BaseModel):
    max_iterations: int = 6
    timeout_secondes: int = 120
    citations_obligatoires: bool = True
    refuser_sans_source: bool = True


class ConfigTechnique(BaseModel):
    """Agrège tout le contenu de config/default.yaml."""

    decoupage: ConfigDecoupage = Field(default_factory=ConfigDecoupage)
    ingestion: ConfigIngestion = Field(default_factory=ConfigIngestion)
    ocr: ConfigOCR = Field(default_factory=ConfigOCR)
    normalisation: ConfigNormalisation = Field(default_factory=ConfigNormalisation)
    resolution_entites: ConfigResolutionEntites = Field(
        default_factory=ConfigResolutionEntites
    )
    recherche: ConfigRecherche = Field(default_factory=ConfigRecherche)
    qdrant: ConfigQdrant = Field(default_factory=ConfigQdrant)
    agent: ConfigAgent = Field(default_factory=ConfigAgent)


# ===========================================================================
# 3. Profil de domaine (config/schemas/<profil>.yaml)
# ===========================================================================

TypeChamp = Literal[
    "texte", "nombre", "entier", "booleen", "date",
    "liste[texte]", "liste[nombre]", "liste[date]",
]

# Correspondance type YAML (français) -> type Python
_TYPES_PYTHON: dict[str, Any] = {
    "texte": str,
    "nombre": float,
    "entier": int,
    "booleen": bool,
    "date": dt.date,
    "liste[texte]": list[str],
    "liste[nombre]": list[float],
    "liste[date]": list[dt.date],
}

# Types considérés comme textuels : seuls ceux-ci passent par la
# normalisation de chaînes et la résolution d'entités.
TYPES_TEXTUELS = {"texte", "liste[texte]"}


class ConfigLangue(BaseModel):
    langue_sortie: str = "fr"
    conserver_langue_source: bool = False


class Categorie(BaseModel):
    nom: str
    description: str


class ConfigClassification(BaseModel):
    active: bool = True
    multi_etiquette: bool = False
    categories: list[Categorie]
    categorie_defaut: str = "autre"
    seuil_confiance: float = Field(default=0.55, ge=0.0, le=1.0)

    @field_validator("categories")
    @classmethod
    def _non_vide(cls, v: list[Categorie]) -> list[Categorie]:
        if not v:
            raise ValueError("Au moins une catégorie est requise.")
        return v

    @model_validator(mode="after")
    def _defaut_existe(self) -> ConfigClassification:
        if self.categorie_defaut not in {c.nom for c in self.categories}:
            raise ValueError(
                f"categorie_defaut '{self.categorie_defaut}' absente de la liste des catégories."
            )
        return self

    def noms(self) -> list[str]:
        return [c.nom for c in self.categories]

    def en_enum(self) -> type[Enum]:
        """Enum dynamique : empêche le LLM d'inventer une catégorie."""
        membres = {c.nom.upper(): c.nom for c in self.categories}
        return Enum("CategorieEnum", membres, type=str)

    def bloc_prompt(self) -> str:
        """Catégories formatées pour insertion dans un prompt."""
        return "\n".join(f"- {c.nom} : {c.description}" for c in self.categories)


class Champ(BaseModel):
    """Une ligne du YAML : décrit un champ à inférer ou à extraire."""

    nom: str
    type: TypeChamp
    description: str
    obligatoire: bool = False
    filtrable: bool = False
    valeurs_autorisees: list[str] | None = None
    normaliser: bool = True
    resoudre_entites: bool = False

    @model_validator(mode="after")
    def _coherence(self) -> Champ:
        if self.valeurs_autorisees and self.type not in TYPES_TEXTUELS:
            raise ValueError(
                f"Champ '{self.nom}' : valeurs_autorisees réservé aux types texte."
            )
        if self.resoudre_entites and self.type not in TYPES_TEXTUELS:
            raise ValueError(
                f"Champ '{self.nom}' : resoudre_entites réservé aux types texte."
            )
        return self

    def type_python(self) -> Any:
        """
        Type Python correspondant.
        Avec valeurs_autorisees, renvoie un Literal : le LLM ne peut
        alors produire que ces valeurs, aucune normalisation nécessaire.
        """
        if self.valeurs_autorisees:
            literal = Literal[tuple(self.valeurs_autorisees)]  # type: ignore[valid-type]
            return list[literal] if self.type.startswith("liste") else literal
        return _TYPES_PYTHON[self.type]

    def est_textuel(self) -> bool:
        return self.type in TYPES_TEXTUELS

    def est_liste(self) -> bool:
        return self.type.startswith("liste")


class ConfigSchemaExtraction(BaseModel):
    nom: str = "ExtractionGenerique"
    description: str = ""
    champs: list[Champ]


class ConfigResume(BaseModel):
    style_defaut: Literal["puces", "paragraphe", "synthese"] = "puces"
    nb_points_max: int = 7


class Profil(BaseModel):
    """Définition complète d'un domaine. Interchangeable sans toucher au code."""

    profile_name: str
    description: str = ""
    langue: ConfigLangue = Field(default_factory=ConfigLangue)
    classification: ConfigClassification
    champs_metadonnees: list[Champ] = Field(default_factory=list)
    schema_extraction: ConfigSchemaExtraction
    resume: ConfigResume = Field(default_factory=ConfigResume)

    # -- Modèles Pydantic générés à la volée --------------------------------

    def modele_metadonnees(self) -> type[BaseModel]:
        """Schéma des métadonnées inférées à l'ingestion. Lu par ingestion.py."""
        return _modele_depuis_champs(
            nom=f"Metadonnees{self.profile_name.capitalize()}",
            description="Métadonnées inférées automatiquement depuis le document.",
            champs=self.champs_metadonnees,
        )

    def modele_extraction(self) -> type[BaseModel]:
        """Schéma de function calling de l'outil extract. Lu par tools/extract.py."""
        return _modele_depuis_champs(
            nom=self.schema_extraction.nom,
            description=self.schema_extraction.description,
            champs=self.schema_extraction.champs,
        )

    # -- Accès par catégorie de champ ---------------------------------------

    def champs_filtrables(self) -> list[str]:
        return [c.nom for c in self.champs_metadonnees if c.filtrable]

    def champs_a_normaliser(self) -> list[Champ]:
        return [c for c in self.champs_metadonnees if c.normaliser]

    def champs_a_resoudre(self) -> list[Champ]:
        return [c for c in self.champs_metadonnees if c.resoudre_entites]

    def champ_metadonnee(self, nom: str) -> Champ | None:
        return next((c for c in self.champs_metadonnees if c.nom == nom), None)

    def bloc_prompt_metadonnees(self) -> str:
        """Champs formatés pour insertion dans le prompt d'inférence."""
        lignes = []
        for c in self.champs_metadonnees:
            suffixe = ""
            if c.valeurs_autorisees:
                suffixe = f" (valeurs possibles : {', '.join(c.valeurs_autorisees)})"
            lignes.append(f"- {c.nom} ({c.type}) : {c.description}{suffixe}")
        return "\n".join(lignes)


def _modele_depuis_champs(
    nom: str, description: str, champs: list[Champ]
) -> type[BaseModel]:
    """
    Traduit une liste de champs YAML en classe Pydantic exécutable.

    Les sorties du LLM sont légèrement normalisées avant validation :
    - "" pour un champ optionnel -> None
    - nombre reçu pour un champ texte -> str
    - None / "" pour une liste optionnelle -> []
    """

    definitions: dict[str, tuple[Any, Any]] = {}

    # Types déclarés dans le YAML, utilisés par le validator dynamique.
    types_champs = {c.nom: c.type for c in champs}
    obligatoires = {c.nom: c.obligatoire for c in champs}

    for c in champs:
        type_py = c.type_python()

        if c.obligatoire:
            definitions[c.nom] = (
                type_py,
                Field(..., description=c.description),
            )
        else:
            definitions[c.nom] = (
                type_py | None,
                Field(default=None, description=c.description),
            )

    @model_validator(mode="before")
    @classmethod
    def _nettoyer_sortie_llm(cls, valeurs: Any) -> Any:
        if not isinstance(valeurs, dict):
            return valeurs

        valeurs = dict(valeurs)

        for champ, type_yaml in types_champs.items():
            if champ not in valeurs:
                continue

            valeur = valeurs[champ]

            # -----------------------------------------------------------
            # Valeurs vides
            # -----------------------------------------------------------
            if valeur == "":
                if type_yaml.startswith("liste["):
                    valeurs[champ] = []
                elif not obligatoires[champ]:
                    valeurs[champ] = None

                continue

            # -----------------------------------------------------------
            # Champs texte :
            # le LLM peut renvoyer un nombre au lieu d'une chaîne.
            # Exemple : reference = 1605.07683
            # -----------------------------------------------------------
            if type_yaml == "texte":
                if valeur is not None and not isinstance(valeur, str):
                    valeurs[champ] = str(valeur)

                continue

            # -----------------------------------------------------------
            # Listes :
            # None -> [] pour éviter des variations inutiles du LLM.
            # -----------------------------------------------------------
            if type_yaml.startswith("liste["):
                if valeur is None:
                    valeurs[champ] = []

                continue

            # -----------------------------------------------------------
            # Dates :
            # "" est déjà converti en None ci-dessus. Une date syntaxiquement
            # invalide (ex. "2016-09-00", jour inconnu recopié tel quel par
            # le LLM depuis une citation qui ne précise que le mois) ne doit
            # pas faire échouer tout le document sur un champ optionnel :
            # Pydantic la validerait strictement et lèverait, ce qui
            # remontait jusqu'à l'ingestion et écartait des documents par
            # ailleurs exploitables. Elle est neutralisée en None, exactement
            # comme "" ci-dessus. Un champ de date obligatoire, lui, doit
            # continuer à échouer : on ne le neutralise pas.
            #
            # Même problème avec une année nue (ex. 1993, sur des documents
            # à contenu historique où le LLM omet le format ISO complet) :
            # reçue comme int/float, Pydantic l'interprète comme un
            # timestamp/ordinal et lève `date_from_datetime_inexact`, une
            # exception qui écartait tout le document. Neutralisée en None
            # selon la même règle, sans jamais inventer de date (ex. year-01-01).
            # -----------------------------------------------------------
            if type_yaml == "date":
                if valeur is None:
                    continue
                if isinstance(valeur, str):
                    try:
                        dt.date.fromisoformat(valeur)
                    except ValueError:
                        if not obligatoires[champ]:
                            valeurs[champ] = None
                elif isinstance(valeur, (int, float)) and not isinstance(valeur, bool):
                    if not obligatoires[champ]:
                        valeurs[champ] = None

        return valeurs

    modele = create_model(
        nom,
        __validators__={
            "_nettoyer_sortie_llm": _nettoyer_sortie_llm,
        },
        **definitions,
    )  # type: ignore[call-overload]

    modele.__doc__ = description or nom
    return modele


# ===========================================================================
# 4. Chargeurs
# ===========================================================================

def _lire_yaml(chemin: Path) -> dict[str, Any]:
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {chemin}")
    with chemin.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.creer_dossiers()
    return s


@lru_cache(maxsize=1)
def get_config_technique() -> ConfigTechnique:
    return ConfigTechnique(**_lire_yaml(DOSSIER_CONFIG / "default.yaml"))


@lru_cache(maxsize=8)
def get_profil(nom: str | None = None) -> Profil:
    nom_profil = nom or get_settings().active_profile
    return Profil(**_lire_yaml(DOSSIER_SCHEMAS / f"{nom_profil}.yaml"))


def lister_profils() -> list[str]:
    return sorted(p.stem for p in DOSSIER_SCHEMAS.glob("*.yaml"))


# ===========================================================================
# 4bis. Registre des corpus (config/corpus.yaml) — isolation multi-corpus
# ===========================================================================
#
# Un corpus logique = 1 collection Qdrant + 1 registre de fichiers + 1 profil,
# strictement isolés (voir `src.rag.corpus`, qui dérive collection/registre
# depuis `corpus_id` et n'accepte jamais un chemin/nom fourni par le client).
# `corpus_id` est le SEUL identifiant qu'un appelant externe peut fournir.

_MOTIF_CORPUS_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class ErreurCorpusInvalide(ValueError):
    """`corpus_id` ne respecte pas le format attendu.

    Format volontairement strict (minuscules, chiffres, `-`, `_`, 1 à 64
    caractères) : exclut par construction tout séparateur de chemin (`/`,
    `\\`), tout `..`, tout caractère de contrôle. `corpus_id` ne doit jamais
    pouvoir être utilisé tel quel comme chemin filesystem non nettoyé.
    """


class CorpusInconnu(ValueError):
    """`corpus_id` au format valide mais absent de `config/corpus.yaml`."""


class CorpusDejaExistant(ValueError):
    """`corpus_id` déjà déclaré — `POST /corpora` ne réécrit jamais un corpus
    existant silencieusement."""


class CorpusProtege(ValueError):
    """`corpus_id` ne peut pas être supprimé (le corpus « default » reste la
    cible implicite de tout appel qui ne fournit pas `corpus_id` — le
    supprimer casserait la rétrocompatibilité pour tout appelant existant)."""


def valider_corpus_id(corpus_id: str) -> str:
    """Valide et normalise (strip) un `corpus_id`. Lève `ErreurCorpusInvalide`
    sinon — jamais un chemin ou un nom de collection acceptés tels quels."""
    valeur = str(corpus_id or "").strip()
    if not _MOTIF_CORPUS_ID.match(valeur):
        raise ErreurCorpusInvalide(
            f"corpus_id invalide : {corpus_id!r}. Attendu : minuscules, "
            "chiffres, '-', '_', 1 à 64 caractères, ex. « finance »."
        )
    return valeur


class DerniereIngestion(BaseModel):
    """Résumé persistant du dernier `RapportIngestion` connu pour ce corpus —
    seul état dynamique conservé, pour dériver un statut fonctionnel
    (`src.rag.corpus.statut_corpus`) sans rejouer l'ingestion ni interroger
    un historique complet (aucune base de données)."""

    statut: Literal["succes", "partiel", "echec"]
    fichiers_trouves: int = 0
    fichiers_traites: int = 0
    fichiers_en_echec: int = 0
    fichiers_ignores_inchanges: int = 0
    chunks_indexes: int = 0
    duree_secondes: float = 0.0
    a: str = ""  # ISO 8601 UTC


class ConfigCorpus(BaseModel):
    """
    Une entrée du registre des corpus.

    `profil` / `profil_domaine` à `None` (cas du corpus « default ») se
    replient sur `Settings.active_profile` / `active_domain_profile` —
    exactement le comportement d'avant l'introduction du multi-corpus.
    Un corpus déclaré explicitement (finance, rh, ...) doit fixer ses
    propres valeurs : jamais de repli implicite sur un réglage global pour
    un corpus autre que « default ».

    `nom` / `cree_le` / `derniere_ingestion` : le seul état persistant
    propre au multi-corpus (au-delà du profil), écrit atomiquement par
    `ecrire_registre_corpus`. Pas de machine à états complexe : le statut
    fonctionnel exposé à l'API (`src.rag.corpus.statut_corpus`) est
    RECALCULÉ à chaque lecture depuis ces quelques champs + l'état réel de
    la collection Qdrant, jamais stocké tel quel.
    """

    profil: str | None = None
    profil_domaine: str | None = None
    source: str = "local"
    nom: str | None = None
    cree_le: str | None = None  # ISO 8601 UTC
    derniere_ingestion: DerniereIngestion | None = None


@lru_cache(maxsize=1)
def get_registre_corpus() -> dict[str, ConfigCorpus]:
    """
    Charge `config/corpus.yaml`. Absent -> registre réduit à `default` avec
    ses valeurs par défaut (aucune régression pour une installation qui
    n'a pas encore ce fichier).
    """
    chemin = DOSSIER_CONFIG / "corpus.yaml"
    brut = _lire_yaml(chemin) if chemin.exists() else {"default": {}}
    return {
        valider_corpus_id(nom): ConfigCorpus(**(entree or {}))
        for nom, entree in brut.items()
    }


def get_corpus(corpus_id: str | None = None) -> ConfigCorpus:
    """Résout une entrée du registre des corpus. Lève `CorpusInconnu` si
    `corpus_id` n'y est pas déclaré."""
    corpus_id = valider_corpus_id(corpus_id or "default")
    entree = get_registre_corpus().get(corpus_id)
    if entree is None:
        raise CorpusInconnu(f"Corpus inconnu : {corpus_id!r}.")
    return entree


def lister_corpus() -> list[str]:
    return sorted(get_registre_corpus())


def ecrire_registre_corpus(registre: dict[str, ConfigCorpus]) -> None:
    """
    Réécrit `config/corpus.yaml` en entier, atomiquement (fichier temporaire
    + renommage), puis invalide le cache. Seule fonction du module qui
    écrit ce fichier — toute création/mise à jour de corpus passe par elle,
    jamais par une écriture directe éparpillée ailleurs.
    """
    chemin = DOSSIER_CONFIG / "corpus.yaml"
    brut = {
        corpus_id: entree.model_dump(exclude_none=True, mode="json")
        for corpus_id, entree in registre.items()
    }
    contenu = yaml.safe_dump(brut, allow_unicode=True, sort_keys=True)

    fd, chemin_temp = tempfile.mkstemp(
        dir=str(chemin.parent), prefix=".corpus-", suffix=".yaml.tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(contenu)
        os.replace(chemin_temp, chemin)
    finally:
        if os.path.exists(chemin_temp):
            os.remove(chemin_temp)

    get_registre_corpus.cache_clear()


def creer_corpus(
    corpus_id: str,
    *,
    nom: str | None = None,
    source: str = "local",
    profil: str | None = None,
) -> ConfigCorpus:
    """
    Déclare un nouveau corpus logique dans le registre.

    Ne crée AUCUNE collection Qdrant ni registre de fichiers : ceux-ci sont
    dérivés à la demande, de façon déterministe, par `src.rag.corpus` — la
    première ingestion les fait exister (`vectorstore.creer_collection`,
    déjà idempotent). Un corpus fraîchement déclaré est donc immédiatement
    dans l'état fonctionnel « indexation requise », sans écriture Qdrant
    inutile.

    Lève `ErreurCorpusInvalide` (format), `CorpusDejaExistant` (doublon).
    """
    corpus_id = valider_corpus_id(corpus_id)
    registre = dict(get_registre_corpus())
    if corpus_id in registre:
        raise CorpusDejaExistant(f"Corpus déjà déclaré : {corpus_id!r}.")

    entree = ConfigCorpus(
        profil=profil,
        source=source,
        nom=(nom or corpus_id).strip() or corpus_id,
        cree_le=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    )
    registre[corpus_id] = entree
    ecrire_registre_corpus(registre)
    return entree


def mettre_a_jour_corpus(corpus_id: str, **champs: Any) -> ConfigCorpus:
    """
    Met à jour certains champs d'un corpus déjà déclaré (ex.
    `profil_domaine` après validation d'un profil, `derniere_ingestion`
    après une synchronisation). Lève `CorpusInconnu` sinon.
    """
    corpus_id = valider_corpus_id(corpus_id)
    registre = dict(get_registre_corpus())
    actuelle = registre.get(corpus_id)
    if actuelle is None:
        raise CorpusInconnu(f"Corpus inconnu : {corpus_id!r}.")

    mise_a_jour = actuelle.model_copy(update=champs)
    registre[corpus_id] = mise_a_jour
    ecrire_registre_corpus(registre)
    return mise_a_jour


def supprimer_corpus_config(corpus_id: str) -> None:
    """
    Retire l'entrée `corpus_id` du registre (`config/corpus.yaml`).

    Ne touche à rien d'autre : ni la collection Qdrant, ni le registre de
    fichiers, ni un profil de domaine persisté — orchestré par
    `src.rag.corpus.supprimer_corpus`, qui appelle cette fonction en
    dernier, une fois le reste du périmètre du corpus effectivement
    nettoyé. Lève `CorpusInconnu` si `corpus_id` n'est pas déclaré.
    """
    corpus_id = valider_corpus_id(corpus_id)
    registre = dict(get_registre_corpus())
    if corpus_id not in registre:
        raise CorpusInconnu(f"Corpus inconnu : {corpus_id!r}.")
    del registre[corpus_id]
    ecrire_registre_corpus(registre)


def recharger_config() -> None:
    """Vide les caches. Utile en dev et pour basculer de profil à chaud."""
    get_settings.cache_clear()
    get_config_technique.cache_clear()
    get_profil.cache_clear()
    get_registre_corpus.cache_clear()


# ===========================================================================
# 5. Vérification manuelle : python -m src.config
# ===========================================================================

if __name__ == "__main__":
    s = get_settings()
    t = get_config_technique()
    p = get_profil()

    print("=" * 62)
    print(f"Profil actif        : {p.profile_name}")
    print(f"Profils disponibles : {lister_profils()}")
    print(f"Langue de sortie    : {p.langue.langue_sortie}")
    print("-" * 62)
    print(f"Documents           : {s.documents_dir}")
    print(f"Qdrant              : {s.qdrant_mode} -> {s.qdrant_path}")
    print(
    f"Collection          : {t.qdrant.nom_collection} "
    f"({t.qdrant.taille_vecteur_dense}d)"
)
    print(f"LLM                 : {s.llm_provider}/{s.llm_model}")
    print(f"Embeddings          : {s.embedding_model} ({s.embedding_device})")
    print(
    f"OCR                 : "
    f"{'actif' if t.ocr.active else 'inactif'} "
    f"[{t.ocr.langues}]"
)
    print("-" * 62)
    print(f"Découpage           : {t.decoupage.taille_chunk} / recouvrement {t.decoupage.recouvrement}")
    print(f"Fusion hybride      : {t.recherche.fusion} (k={t.recherche.rrf_k})")
    print(f"Seuil Levenshtein   : {t.resolution_entites.seuil_levenshtein}")
    print("-" * 62)
    print(f"Catégories          : {p.classification.noms()}")
    print(f"Champs filtrables   : {p.champs_filtrables()}")
    print(f"Champs à normaliser : {[c.nom for c in p.champs_a_normaliser()]}")
    print(f"Champs à résoudre   : {[c.nom for c in p.champs_a_resoudre()]}")
    print("-" * 62)

    ModeleMeta = p.modele_metadonnees()
    ModeleExtr = p.modele_extraction()
    print(f"Modèle métadonnées  : {ModeleMeta.__name__}")
    for nom_champ, info in ModeleMeta.model_fields.items():
        print(f"    {nom_champ:<20} {info.annotation}")
    print(f"Modèle extraction   : {ModeleExtr.__name__}")
    for nom_champ, info in ModeleExtr.model_fields.items():
        print(f"    {nom_champ:<20} {info.annotation}")
    print("=" * 62)