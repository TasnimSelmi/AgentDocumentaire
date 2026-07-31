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
    """Réglages machine. Rien de spécifique à un domaine."""

    model_config = SettingsConfigDict(
        env_file=RACINE_PROJET / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    llm_provider: Literal["openai", "anthropic", "ollama"] = "openai"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048

    # --- Embeddings / reranking ---
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: Literal["cpu", "cuda"] = "cpu"
    embedding_batch_size: int = 8
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_enabled: bool = True

    # --- Qdrant ---
    qdrant_mode: Literal["local", "server"] = "local"
    qdrant_path: Path = Path("data/vectordb")
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "documents"

    # --- OCR ---
    ocr_enabled: bool = True
    ocr_languages: str = "fra+ara+eng"
    tesseract_cmd: str = ""          # vide = binaire trouvé via le PATH

    # --- Chemins ---
    documents_dir: Path = Path("data/documents")
    vectordb_dir: Path = Path("data/vectordb")
    logs_dir: Path = Path("data/logs")

    # --- Profil de domaine actif ---
    active_profile: str = "generic"

    log_level: str = "INFO"

    @field_validator(
        "documents_dir", "vectordb_dir", "logs_dir", "qdrant_path", mode="after"
    )
    @classmethod
    def _chemin_absolu(cls, v: Path) -> Path:
        """Rend les chemins absolus : le projet devient lançable de n'importe où."""
        return v if v.is_absolute() else RACINE_PROJET / v

    def creer_dossiers(self) -> None:
        for p in (self.documents_dir, self.vectordb_dir, self.logs_dir):
            p.mkdir(parents=True, exist_ok=True)

    @property
    def chemin_registre(self) -> Path:
        """Registre des fichiers déjà indexés (hash -> doc_id)."""
        return self.vectordb_dir / "registry.json"

    @property
    def chemin_entites(self) -> Path:
        """Registre des entités canoniques et de leurs alias."""
        return self.vectordb_dir / "entities.json"


# ===========================================================================
# 2. Paramètres techniques (config/default.yaml)
# ===========================================================================

class ConfigDecoupage(BaseModel):
    strategie: Literal["recursive", "semantic"] = "recursive"
    taille_chunk: int = 1000
    recouvrement: int = 150
    separateurs: list[str] = Field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )

    @model_validator(mode="after")
    def _recouvrement_coherent(self) -> ConfigDecoupage:
        if self.recouvrement >= self.taille_chunk:
            raise ValueError("Le recouvrement doit être inférieur à la taille du chunk.")
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

    C'est le mécanisme central de la généricité : la classe produite sert
    de schéma de function calling, forçant le LLM à respecter exactement
    les champs et les types déclarés dans le YAML.
    """
    definitions: dict[str, tuple[Any, Any]] = {}
    for c in champs:
        type_py = c.type_python()
        if c.obligatoire:
            definitions[c.nom] = (type_py, Field(..., description=c.description))
        else:
            definitions[c.nom] = (
                type_py | None,
                Field(default=None, description=c.description),
            )

    modele = create_model(nom, **definitions)  # type: ignore[call-overload]
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


def recharger_config() -> None:
    """Vide les caches. Utile en dev et pour basculer de profil à chaud."""
    get_settings.cache_clear()
    get_config_technique.cache_clear()
    get_profil.cache_clear()


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
    print(f"Collection          : {t.qdrant.nom_collection} ({t.qdrant.taille_vecteur_dense}d)")
    print(f"LLM                 : {s.llm_provider}/{s.llm_model}")
    print(f"Embeddings          : {s.embedding_model} ({s.embedding_device})")
    print(f"OCR                 : {'actif' if s.ocr_enabled else 'inactif'} [{s.ocr_languages}]")
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