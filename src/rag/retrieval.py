"""
Couche de récupération du RAG : question -> passages pertinents.

Ce module reste volontairement déterministe et sans LLM. Il orchestre :
    1. validation et normalisation des filtres ;
    2. résolution générique du périmètre documentaire visé par la question ;
    3. encodage dense + sparse de la requête avec BGE-M3 ;
    4. recherche hybride dans Qdrant, cloisonnée au périmètre le cas échéant ;
    5. reranking des candidats ;
    6. déduplication et diversification des passages ;
    7. attribution d'identifiants de citation stables pour la génération.

Il constitue la frontière entre la base documentaire et la génération.
Plus tard, l'agent appellera cette couche comme un outil, sans dupliquer
la logique de recherche.

Résolution documentaire (générique, sans aucun nom codé en dur) :
    Un catalogue est dérivé à l'exécution des métadonnées déjà présentes dans
    les payloads Qdrant : identifiant, nom de fichier, titre, organisation,
    type, année de publication, alias. Aucun de ces champs n'est obligatoire ;
    le catalogue se contente de ce que le corpus expose.

    Le vocabulaire discriminant est pondéré par sa rareté (IDF) dans le
    catalogue : un jeton partagé par de nombreux documents pèse presque rien,
    un jeton propre à un seul document domine. Le module fonctionne donc à
    l'identique sur n'importe quel corpus, sans règle métier.

    La résolution renvoie l'un de quatre états : exact, compatible (plusieurs
    documents également valables), ambigu, ou aucun. Un filtre Qdrant strict
    n'est posé que dans les deux premiers cas.

Utilisation manuelle :
    python -m src.rag.retrieval "Comment installer le produit ?"
    python -m src.rag.retrieval "Quels rapports datent de 2026 ?" \
        --filtres '{"categorie": "rapport", "date_document": {"gte": "2026-01-01"}}'
    python -m src.rag.retrieval "Total consumed in 2020 in the 2022 report"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Sequence

from src.config import Champ, Profil, get_config_technique, get_profil, get_settings
from src.rag.embeddings import encoder_requete, reranker
from src.rag.normalization import normaliser_booleen, normaliser_entier, normaliser_valeur
from src.rag.vectorstore import (
    Resultat,
    construire_filtre,
    fermer_client,
    info_collection,
    rechercher,
    recuperer_contexte,
)

# Le catalogue documentaire est optionnel : si vectorstore.py n'expose pas
# encore lister_documents(), la résolution est simplement désactivée et le
# module conserve son comportement historique. Aucun import dur, donc aucune
# régression de compilation.
try:  # pragma: no cover - dépend de la version de vectorstore.py
    from src.rag.vectorstore import lister_documents as _lister_documents
except ImportError:  # pragma: no cover
    _lister_documents = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Champs techniques présents dans le payload de tous les chunks. Ils ne
# dépendent pas du profil métier et peuvent donc toujours être filtrés.
_CHAMPS_TECHNIQUES = {
    "doc_id",
    "source",
    "nom_fichier",
    "categorie",
    "ocr",
    "page",
    "chunk_index",
    "hash_contenu",
}

# Champs candidats pour cloisonner une recherche sur un document, par ordre
# de fiabilité décroissante. Le premier qui est renseigné pour tout le
# périmètre et filtrable selon le profil est retenu.
_CHAMPS_PORTEE_DOCUMENT = ("doc_id", "nom_fichier", "source")


# ===========================================================================
# Exceptions
# ===========================================================================


class ErreurRecherche(RuntimeError):
    """Erreur de haut niveau de la couche de récupération."""


class FiltreInvalide(ErreurRecherche):
    """Un filtre ne correspond pas au profil actif ou à un type attendu."""


class CollectionIndisponible(ErreurRecherche):
    """La collection Qdrant n'existe pas encore ou ne contient aucun point."""


class DocumentInconnu(ErreurRecherche):
    """Le document explicitement demandé n'existe pas dans le corpus indexé."""


# ===========================================================================
# Structures publiques
# ===========================================================================


@dataclass
class Passage:
    """Passage final transmis à la génération."""

    citation: str
    rang: int
    point_id: str
    doc_id: str
    chunk_index: int | None
    texte: str
    source: str
    nom_fichier: str
    page: int | None
    categorie: str
    score_recherche: float
    score_reranking: float | None
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def score_final(self) -> float:
        """Score utilisé pour l'affichage et le diagnostic."""
        return (
            self.score_reranking
            if self.score_reranking is not None
            else self.score_recherche
        )

    @property
    def localisation(self) -> str:
        """Libellé humain de la source, sans inventer de numéro de page."""
        nom = self.nom_fichier or self.source or "source inconnue"
        return f"{nom}, page {self.page}" if self.page is not None else nom

    @property
    def libelle_document(self) -> str:
        """Nom du document, sans numéro de page.

        La génération doit annoncer la provenance de chaque extrait et la
        validation doit pouvoir comparer ce libellé au périmètre demandé.
        """
        return self.nom_fichier or self.doc_id or self.source or "source inconnue"

    @property
    def identifiants(self) -> tuple[str, ...]:
        """Tous les identifiants de document portés par ce passage."""
        valeurs = [self.doc_id, self.nom_fichier, self.source]
        return tuple(valeur for valeur in valeurs if valeur)


@dataclass(frozen=True)
class FicheDocument:
    """Entrée du catalogue : identité d'un document, telle qu'indexée.

    Tous les champs sauf ``document_id`` sont facultatifs. Un corpus qui
    n'expose qu'un nom de fichier reste parfaitement exploitable ; les
    facettes absentes ne participent simplement pas à la résolution.
    """

    document_id: str
    champ_id: str
    nom_fichier: str = ""
    source: str = ""
    titre: str = ""
    organisation: str = ""
    type_document: str = ""
    annees: tuple[int, ...] = ()
    alias: tuple[str, ...] = ()
    jetons: frozenset[str] = frozenset()
    signatures: tuple[frozenset[str], ...] = ()

    @property
    def libelle(self) -> str:
        """Libellé lisible, dans l'ordre de ce qu'un humain reconnaîtrait."""
        return (
            self.titre
            or self.nom_fichier
            or self.organisation
            or self.document_id
            or self.source
        )

    def identifiant_pour(self, champ: str) -> str:
        if champ in {"doc_id", "document_id", self.champ_id}:
            return self.document_id
        return str(getattr(self, champ, "") or "")


@dataclass(frozen=True)
class PerimetreDocumentaire:
    """Périmètre documentaire imposé à une recherche.

    ``statut`` vaut :
        - ``exact``      : un seul document désigné sans ambiguïté ;
        - ``compatible`` : plusieurs documents également valables, tous
          retenus plutôt qu'un choix arbitraire ;
        - ``ambigu``     : des candidats trop proches, aucun filtre posé ;
        - ``aucun``      : la question ne désigne aucun document.

    Un filtre Qdrant strict n'est appliqué que pour ``exact`` et
    ``compatible`` : un filtre erroné masquerait la bonne réponse au lieu de
    la protéger.
    """

    statut: str
    libelles: tuple[str, ...] = ()
    champ_filtre: str = ""
    valeurs_filtre: tuple[str, ...] = ()
    score: float = 0.0
    jetons_reconnus: tuple[str, ...] = ()
    origine: str = "detection"  # "explicite" | "detection"
    concurrent: str | None = None
    score_concurrent: float = 0.0
    annees_publication: tuple[int, ...] = ()
    annees_valeur: tuple[int, ...] = ()
    raison: str = ""

    @property
    def contraignant(self) -> bool:
        """Vrai si ce périmètre doit se traduire par un filtre Qdrant."""
        return self.statut in {"exact", "compatible"} and bool(self.valeurs_filtre)

    @property
    def unique(self) -> bool:
        return len(self.valeurs_filtre) == 1

    @property
    def marge(self) -> float:
        return round(self.score - self.score_concurrent, 4)

    @property
    def libelle(self) -> str:
        """Libellé lisible du périmètre, pour les journaux et les messages."""
        if not self.libelles:
            return "—"
        if self.unique:
            return self.libelles[0]
        return " + ".join(self.libelles)

    def contient(self, *identifiants: Any) -> bool:
        """Vrai si l'un des identifiants fournis appartient au périmètre.

        La génération et la validation s'appuient sur cette méthode pour
        vérifier qu'un passage cité provient bien d'un document autorisé,
        sans jamais manipuler de nom de corpus en dur.
        """
        if not self.contraignant:
            return True
        autorises = set(self.valeurs_filtre) | set(self.libelles)
        return any(
            str(valeur).strip() in autorises for valeur in identifiants if valeur
        )


@dataclass
class RapportRecherche:
    """Résultat complet d'une recherche, passages et diagnostics compris."""

    requete: str
    profil: str
    filtres: dict[str, Any]
    passages: list[Passage]
    candidats_recuperes: int
    reranking_utilise: bool
    seuil_applique: float | None
    duree_secondes: float
    # Champs ajoutés pour le cloisonnement documentaire. Ils portent une
    # valeur par défaut afin de ne casser aucun appel existant.
    perimetre: PerimetreDocumentaire | None = None
    diversification_active: bool = True
    motif_absence: str | None = None

    @property
    def est_vide(self) -> bool:
        return not self.passages

    @property
    def documents_demandes(self) -> tuple[str, ...]:
        """Identifiants des documents imposés à la recherche, s'il y en a."""
        if self.perimetre is not None and self.perimetre.contraignant:
            return self.perimetre.valeurs_filtre
        return ()

    @property
    def contexte_insuffisant(self) -> bool:
        """Vrai lorsque la génération ne doit pas être appelée.

        La couche de génération s'appuie sur cette propriété pour refuser de
        répondre au lieu de laisser le LLM improviser à partir d'un contexte
        vide ou hors périmètre.
        """
        return self.est_vide or self.motif_absence is not None

    def vers_dict(self) -> dict[str, Any]:
        return asdict(self)


# ===========================================================================
# Validation et normalisation des filtres
# ===========================================================================


def _champ_par_nom(profil: Profil) -> dict[str, Champ]:
    return {champ.nom: champ for champ in profil.champs_metadonnees}


def champs_filtrables(profil: Profil | None = None) -> dict[str, str]:
    """
    Renvoie les champs autorisés pour une requête filtrée.

    Les champs métier proviennent exclusivement du profil actif. Les champs
    techniques sont ajoutés explicitement car ils existent dans tous les
    payloads Qdrant.
    """
    profil = profil or get_profil()
    resultat = {
        "doc_id": "texte",
        "source": "texte",
        "nom_fichier": "texte",
        "categorie": "texte",
        "ocr": "booleen",
        "page": "entier",
        "chunk_index": "entier",
        "hash_contenu": "texte",
    }
    resultat.update(
        {
            champ.nom: champ.type
            for champ in profil.champs_metadonnees
            if champ.filtrable
        }
    )
    return resultat


def _normaliser_technique(cle: str, valeur: Any, profil: Profil) -> Any:
    """Normalise les filtres techniques sans leur appliquer un profil métier."""
    if cle == "categorie":
        def categorie_unique(v: Any) -> str:
            texte = str(v).strip()
            if texte not in profil.classification.noms():
                autorisees = ", ".join(profil.classification.noms())
                raise FiltreInvalide(
                    f"Catégorie inconnue : {texte!r}. Valeurs autorisées : {autorisees}."
                )
            return texte

        if isinstance(valeur, (list, tuple, set)):
            return [categorie_unique(v) for v in valeur]
        return categorie_unique(valeur)

    if cle == "ocr":
        if isinstance(valeur, (list, tuple, set)):
            normalisees = [normaliser_booleen(v) for v in valeur]
            if any(v is None for v in normalisees):
                raise FiltreInvalide(f"Valeur booléenne invalide pour {cle!r}: {valeur!r}.")
            return normalisees
        normalisee = normaliser_booleen(valeur)
        if normalisee is None:
            raise FiltreInvalide(f"Valeur booléenne invalide pour {cle!r}: {valeur!r}.")
        return normalisee

    if cle in {"page", "chunk_index"}:
        if isinstance(valeur, dict):
            sortie: dict[str, int] = {}
            for borne, contenu in valeur.items():
                if borne not in {"gte", "lte", "gt", "lt"}:
                    raise FiltreInvalide(
                        f"Borne inconnue {borne!r} pour le filtre {cle!r}."
                    )
                entier = normaliser_entier(contenu)
                if entier is None:
                    raise FiltreInvalide(
                        f"Valeur entière invalide pour {cle!r}: {contenu!r}."
                    )
                sortie[borne] = entier
            return sortie

        if isinstance(valeur, (list, tuple, set)):
            sortie_liste = [normaliser_entier(v) for v in valeur]
            if any(v is None for v in sortie_liste):
                raise FiltreInvalide(f"Valeur entière invalide pour {cle!r}: {valeur!r}.")
            return sortie_liste

        entier = normaliser_entier(valeur)
        if entier is None:
            raise FiltreInvalide(f"Valeur entière invalide pour {cle!r}: {valeur!r}.")
        return entier

    # doc_id, source, nom_fichier et hash_contenu sont des identifiants exacts.
    if isinstance(valeur, (list, tuple, set)):
        return [str(v).strip() for v in valeur if str(v).strip()]
    return str(valeur).strip()


def _normaliser_element_champ(valeur: Any, champ: Champ) -> Any:
    """Normalise une valeur unique, y compris pour un champ déclaré comme liste."""
    cfg = get_config_technique().normalisation

    if champ.est_liste():
        # normaliser_valeur transforme un scalaire en liste. Pour un critère
        # individuel, on récupère ensuite son unique valeur normalisée.
        resultat = normaliser_valeur(valeur, champ, cfg)
        if isinstance(resultat, list):
            return resultat[0] if resultat else None
        return resultat

    return normaliser_valeur(valeur, champ, cfg)


def _normaliser_metier(cle: str, valeur: Any, champ: Champ) -> Any:
    """Normalise un filtre métier suivant le type déclaré dans le YAML."""
    if isinstance(valeur, dict):
        if not ({"gte", "lte", "gt", "lt"} & valeur.keys()):
            raise FiltreInvalide(
                f"Le filtre {cle!r} est un objet, mais aucune borne gte/lte/gt/lt n'est fournie."
            )
        if champ.type not in {"date", "nombre", "entier"}:
            raise FiltreInvalide(
                f"Les plages ne sont pas autorisées sur le champ {cle!r} de type {champ.type}."
            )

        sortie: dict[str, Any] = {}
        for borne, contenu in valeur.items():
            if borne not in {"gte", "lte", "gt", "lt"}:
                raise FiltreInvalide(
                    f"Borne inconnue {borne!r} pour le filtre {cle!r}."
                )
            normalisee = _normaliser_element_champ(contenu, champ)
            if normalisee is None:
                raise FiltreInvalide(
                    f"Valeur invalide pour {cle!r} ({champ.type}) : {contenu!r}."
                )
            sortie[borne] = normalisee
        return sortie

    if isinstance(valeur, (list, tuple, set)):
        normalisees: list[Any] = []
        for element in valeur:
            normalise = _normaliser_element_champ(element, champ)
            if normalise is not None and normalise not in normalisees:
                normalisees.append(normalise)
        if not normalisees:
            raise FiltreInvalide(f"Aucune valeur valide pour le filtre {cle!r}.")
        return normalisees

    normalisee = _normaliser_element_champ(valeur, champ)
    if normalisee is None:
        raise FiltreInvalide(
            f"Valeur invalide pour {cle!r} ({champ.type}) : {valeur!r}."
        )
    return normalisee


def preparer_filtres(
    criteres: dict[str, Any] | None,
    profil: Profil | None = None,
) -> dict[str, Any]:
    """
    Valide et normalise les critères avant leur traduction en filtre Qdrant.

    Refuser les champs inconnus est volontaire : une faute dans un nom de
    filtre ne doit pas être silencieusement ignorée, car elle élargirait la
    recherche et pourrait produire une réponse trompeuse.
    """
    if not criteres:
        return {}
    if not isinstance(criteres, dict):
        raise FiltreInvalide("Les critères doivent être fournis sous forme de dictionnaire.")

    profil = profil or get_profil()
    champs_metier = _champ_par_nom(profil)
    autorises = set(champs_filtrables(profil))
    sortie: dict[str, Any] = {}

    for cle, valeur in criteres.items():
        if cle not in autorises:
            liste = ", ".join(sorted(autorises))
            raise FiltreInvalide(
                f"Champ non filtrable ou inconnu : {cle!r}. Champs autorisés : {liste}."
            )
        if valeur is None:
            continue

        if cle in _CHAMPS_TECHNIQUES:
            sortie[cle] = _normaliser_technique(cle, valeur, profil)
        else:
            sortie[cle] = _normaliser_metier(cle, valeur, champs_metier[cle])

    return sortie


# ===========================================================================
# Catalogue documentaire générique
# ===========================================================================

# Jetons structurels, sans pouvoir discriminant dans une identité de
# document. La liste ne contient aucun terme métier ni aucun nom propre :
# elle reste donc valable pour n'importe quel corpus.
_JETONS_GENERIQUES: frozenset[str] = frozenset(
    {
        # anglais
        "annual", "integrated", "sustainability", "report", "reports",
        "document", "documents", "doc", "file", "files", "statement",
        "statements", "financial", "results", "review", "group", "limited",
        "ltd", "plc", "inc", "corp", "corporation", "company", "holdings",
        "final", "draft", "version", "edition", "the", "of", "and", "for",
        "in", "on", "to", "at", "by",
        # français
        "rapport", "rapports", "annuel", "annuelle", "integre", "integree",
        "durabilite", "financier", "financiere", "resultats", "societe",
        "groupe", "entreprise", "fichier", "dossier", "note", "compte",
        "rendu", "finale", "brouillon", "edition",
        "de", "du", "des", "la", "le", "les", "un", "une", "et", "en",
        "pour", "dans", "sur", "au", "aux",
        # extensions
        "pdf", "docx", "txt", "md", "html", "csv", "xlsx", "pptx",
    }
)

# Formulations qui signalent que l'utilisateur restreint sa question à une
# source. Leur présence renforce la confiance ; leur absence n'interdit rien.
_MARQUEURS_PORTEE: tuple[str, ...] = (
    "document", "report", "file", "filing", "statement", "edition",
    "according to", "based on", "from the", "in the",
    "rapport", "fichier", "dossier", "selon", "d apres", "dans le",
)

# Mots qui, accolés à une année, désignent l'exercice de publication du
# document et non l'année de la valeur recherchée.
_MOTS_PUBLICATION: frozenset[str] = frozenset(
    {
        "report", "reports", "statement", "statements", "filing", "filings",
        "edition", "document", "documents", "review",
        "rapport", "rapports", "exercice", "millesime",
    }
)

# Seuls ces qualificatifs peuvent s'intercaler entre une année et le nom de
# publication qu'elle date (« 2021 annual financial statements »). Les
# prépositions et articles en sont volontairement exclus : ils introduisent
# un nouveau groupe nominal, et les franchir ferait lire « la valeur de 2020
# dans le rapport 2022 » comme une demande sur la publication 2020.
_MODIFIEURS_PUBLICATION: frozenset[str] = frozenset(
    {
        "annual", "integrated", "sustainability", "financial", "interim",
        "consolidated", "final", "draft", "full", "half", "year",
        "annuel", "annuelle", "integre", "integree", "financier",
        "financiere", "consolide", "consolidee", "intermediaire", "finale",
    }
)

# Noms de clés de payload possibles pour chaque facette d'identité. Il s'agit
# de conventions de schéma, jamais de valeurs métier : aucun nom de société,
# de corpus ni de domaine n'apparaît ici. Une clé absente est ignorée.
_CLES_FACETTE: dict[str, tuple[str, ...]] = {
    "document_id": ("doc_id", "document_id", "id_document", "identifiant"),
    "nom_fichier": ("nom_fichier", "file_name", "filename", "fichier"),
    "source": ("source", "chemin", "path", "uri"),
    "titre": ("titre", "title", "document_title", "titre_document", "intitule"),
    "organisation": (
        "organisation", "organization", "org", "entreprise", "societe",
        "company", "auteur", "author", "editeur", "publisher", "emetteur",
    ),
    "type_document": (
        "type_document", "document_type", "type", "categorie", "nature",
    ),
    "annee": (
        "annee", "year", "document_year", "annee_document", "annee_publication",
        "publication_year", "exercice", "date_document", "date_publication",
    ),
    "alias": ("alias", "aliases", "synonymes", "synonyms", "acronymes"),
}

_MOTIF_MOTS = re.compile(r"[a-z0-9]+")
_MOTIF_ANNEE = re.compile(r"^(19|20)\d{2}$")

# Un score de couverture inférieur au seuil signifie que la question ne
# désigne pas assez clairement un document. Une marge trop faible entre les
# deux meilleurs candidats signifie que la question est ambiguë : dans les
# deux cas aucun filtre n'est posé, car un filtre erroné masque la bonne
# réponse au lieu de la protéger.
_SEUIL_RESOLUTION = 0.55
_MARGE_RESOLUTION = 0.15

# Distance maximale, en jetons, entre une année et le mot de publication qui
# la qualifie (« 2021 annual financial statements »).
_PORTEE_ANNEE = 3

_catalogue: CatalogueDocuments | None = None


def _normaliser_texte(texte: Any) -> str:
    """Minuscule, sans accents, sans ponctuation, espaces normalisés."""
    brut = unicodedata.normalize("NFKD", str(texte))
    brut = "".join(car for car in brut if not unicodedata.combining(car))
    brut = brut.lower().replace("_", " ").replace("-", " ")
    return " ".join(_MOTIF_MOTS.findall(brut))


def _ressemble_a_un_hash(jeton: str) -> bool:
    """Détecte un identifiant technique (uuid, sha, horodatage concaténé).

    Un identifiant calculé n'apparaîtra jamais dans une question ; l'inclure
    dans la signature ferait chuter mécaniquement le taux de couverture et
    empêcherait toute résolution.
    """
    if len(jeton) < 8:
        return False
    if re.fullmatch(r"[0-9a-f]+", jeton):
        return True
    return any(car.isdigit() for car in jeton) and any(car.isalpha() for car in jeton)


def _jeton_non_discriminant(jeton: str) -> bool:
    """Un jeton qui ne peut jamais identifier un document à lui seul."""
    return (
        len(jeton) < 3
        or jeton in _JETONS_GENERIQUES
        or bool(_MOTIF_ANNEE.match(jeton))
        or jeton.isdigit()
        or _ressemble_a_un_hash(jeton)
    )


def _racine_identifiant(identifiant: str) -> str:
    """Retire le chemin et l'extension d'un identifiant de document."""
    base = str(identifiant).replace("\\", "/").rsplit("/", 1)[-1]
    return re.sub(r"\.[a-z0-9]{1,5}$", "", base, flags=re.IGNORECASE)


def _jetons_utiles(texte: str) -> set[str]:
    """Jetons discriminants d'un libellé, hors mots structurels et années."""
    return {
        jeton
        for jeton in _normaliser_texte(_racine_identifiant(texte)).split()
        if not _jeton_non_discriminant(jeton)
    }


def _annees_dans(texte: Any) -> tuple[int, ...]:
    """Années plausibles contenues dans un texte, dédoublonnées et triées."""
    trouvees = {
        int(annee)
        for annee in re.findall(r"(?:19|20)\d{2}", str(texte))
    }
    return tuple(sorted(trouvees))


def _acronyme(jetons: Iterable[str]) -> str:
    """Sigle formé des initiales des jetons discriminants d'un libellé.

    Utile quand un utilisateur écrit le sigle plutôt que le nom complet. La
    construction est purement mécanique et ne suppose aucun corpus.
    """
    initiales = [jeton[0] for jeton in jetons if jeton]
    return "".join(initiales) if len(initiales) >= 2 else ""


def _valeur_facette(payload: dict[str, Any], facette: str) -> Any:
    """Première valeur non vide parmi les clés connues d'une facette."""
    for cle in _CLES_FACETTE.get(facette, ()):
        if cle in payload:
            valeur = payload[cle]
            if valeur is None:
                continue
            if isinstance(valeur, (list, tuple, set)):
                valeurs = [str(v).strip() for v in valeur if str(v).strip()]
                if valeurs:
                    return valeurs
                continue
            texte = str(valeur).strip()
            if texte:
                return texte
    return ""


def _texte_facette(payload: dict[str, Any], facette: str) -> str:
    valeur = _valeur_facette(payload, facette)
    if isinstance(valeur, list):
        return " ".join(valeur)
    return str(valeur)


def cles_catalogue(profil: Profil | None = None) -> list[str]:
    """Clés de payload à lire pour construire le catalogue.

    Les conventions de nommage connues sont complétées par les champs de
    métadonnées déclarés dans le profil actif : un corpus qui expose ses
    propres noms de champs reste donc exploitable sans modifier ce module.
    """
    profil = profil or get_profil()
    cles: list[str] = []
    for candidats in _CLES_FACETTE.values():
        cles.extend(candidats)
    cles.extend(champ.nom for champ in profil.champs_metadonnees)
    return sorted(dict.fromkeys(cles))


def _fiche_depuis_payload(payload: dict[str, Any]) -> FicheDocument | None:
    """Construit une entrée de catalogue à partir d'un payload de chunk."""
    champ_id = ""
    document_id = ""
    for cle in _CLES_FACETTE["document_id"]:
        valeur = str(payload.get(cle, "") or "").strip()
        if valeur:
            champ_id, document_id = cle, valeur
            break

    nom_fichier = _texte_facette(payload, "nom_fichier")
    source = _texte_facette(payload, "source")

    if not document_id:
        # Corpus sans identifiant stable : on retombe sur le libellé le plus
        # spécifique disponible, sans jamais échouer.
        document_id = nom_fichier or source
        champ_id = "nom_fichier" if nom_fichier else "source"
    if not document_id:
        return None

    titre = _texte_facette(payload, "titre")
    organisation = _texte_facette(payload, "organisation")
    type_document = _texte_facette(payload, "type_document")

    alias_brut = _valeur_facette(payload, "alias")
    alias = tuple(alias_brut) if isinstance(alias_brut, list) else (
        (alias_brut,) if alias_brut else ()
    )

    # Année de publication : la métadonnée fait foi ; à défaut on la déduit
    # du titre puis du nom de fichier, où elle figure presque toujours.
    annees = _annees_dans(_texte_facette(payload, "annee"))
    if not annees:
        annees = _annees_dans(titre) or _annees_dans(nom_fichier)

    # Signature lexicale : une par facette d'identité. Comparer facette par
    # facette est indispensable — un document richement décrit (organisation,
    # titre, sigle, nom de fichier abrégé) verrait sinon son taux de
    # couverture s'effondrer alors même que l'utilisateur l'a nommé
    # exactement. Le type de document est volontairement exclu : partagé par
    # tout le corpus, il ne discrimine rien.
    signatures: list[frozenset[str]] = []
    for libelle in (organisation, titre, nom_fichier, *alias):
        jetons_facette = _jetons_utiles(libelle)
        if jetons_facette:
            signatures.append(frozenset(jetons_facette))

    # Le sigle constitue une facette à part entière : un utilisateur écrit
    # soit le nom complet, soit son acronyme, jamais un mélange des deux.
    for libelle in (organisation, titre):
        sigle = _acronyme(sorted(_jetons_utiles(libelle)))
        if sigle and not _jeton_non_discriminant(sigle):
            signatures.append(frozenset({sigle}))

    if not signatures:
        # Dernier recours : le chemin, puis l'identifiant lui-même.
        secours = _jetons_utiles(source) or _jetons_utiles(document_id)
        if secours:
            signatures.append(frozenset(secours))

    jetons: set[str] = set()
    for signature in signatures:
        jetons |= signature

    return FicheDocument(
        document_id=document_id,
        champ_id=champ_id,
        nom_fichier=nom_fichier,
        source=source,
        titre=titre,
        organisation=organisation,
        type_document=type_document,
        annees=annees,
        alias=alias,
        jetons=frozenset(jetons),
        signatures=tuple(dict.fromkeys(signatures)),
    )


def _annees_de_publication(jetons: Sequence[str]) -> tuple[set[int], set[int]]:
    """Sépare les années de publication des années de valeur d'une question.

    Une année qualifie la publication lorsqu'elle précède un mot de
    publication, éventuellement séparée par des mots structurels
    (« 2021 annual financial statements »), ou lorsqu'elle suit
    immédiatement un tel mot (« rapport 2022 »).

    Toute autre année désigne la valeur recherchée et ne doit jamais servir
    à cloisonner la recherche : « la valeur de 2020 dans le rapport 2022 »
    porte sur une métrique 2020 dans un document publié en 2022.
    """
    publication: set[int] = set()
    valeur: set[int] = set()

    for indice, jeton in enumerate(jetons):
        if not _MOTIF_ANNEE.match(jeton):
            continue
        annee = int(jeton)

        # « rapport 2022 » : le mot de publication précède immédiatement.
        precedent = jetons[indice - 1] if indice > 0 else ""
        if precedent in _MOTS_PUBLICATION:
            publication.add(annee)
            continue

        # « 2021 annual financial statements » : le mot de publication suit,
        # après d'éventuels qualificatifs. Toute préposition, article ou mot
        # porteur de sens interrompt la lecture : dans « 2020 in the
        # document » comme dans « 2020 dans le rapport », l'année qualifie la
        # valeur demandée, pas le document.
        qualifiee = False
        for suivant in jetons[indice + 1 : indice + 1 + _PORTEE_ANNEE]:
            if suivant in _MOTS_PUBLICATION:
                qualifiee = True
                break
            if suivant not in _MODIFIEURS_PUBLICATION:
                break

        (publication if qualifiee else valeur).add(annee)

    return publication, valeur


@dataclass
class CatalogueDocuments:
    """Catalogue des documents indexés et pondération IDF de leurs jetons."""

    fiches: list[FicheDocument]
    idf: dict[str, float]

    @classmethod
    def construire(cls, fiches: list[FicheDocument]) -> CatalogueDocuments:
        """Pondère chaque jeton par sa rareté dans l'ensemble du catalogue.

        C'est ce calcul qui rend la résolution générique : un jeton présent
        dans beaucoup d'identités ne pèse presque rien, un jeton propre à un
        seul document pèse au maximum. Aucun nom n'a besoin d'être connu à
        l'avance et aucune règle métier n'est nécessaire.
        """
        frequences: dict[str, int] = {}
        for fiche in fiches:
            for jeton in fiche.jetons:
                frequences[jeton] = frequences.get(jeton, 0) + 1

        total = max(len(fiches), 1)
        idf = {
            jeton: math.log(1.0 + total / (1.0 + compte))
            for jeton, compte in frequences.items()
        }
        return cls(fiches=fiches, idf=idf)

    @property
    def est_vide(self) -> bool:
        return not self.fiches

    def par_identifiant(self, identifiant: str) -> FicheDocument | None:
        """Retrouve une fiche par identifiant, nom de fichier ou titre exact."""
        cible = str(identifiant).strip()
        if not cible:
            return None

        for fiche in self.fiches:
            if cible in {
                fiche.document_id,
                fiche.nom_fichier,
                fiche.source,
                fiche.titre,
            }:
                return fiche

        # Tolérance : comparaison normalisée, utile en ligne de commande.
        cible_normalisee = _normaliser_texte(_racine_identifiant(cible))
        for fiche in self.fiches:
            connus = {
                _normaliser_texte(_racine_identifiant(valeur))
                for valeur in (
                    fiche.document_id,
                    fiche.nom_fichier,
                    fiche.source,
                    fiche.titre,
                )
                if valeur
            }
            if cible_normalisee in connus:
                return fiche
        return None

    # ------------------------------------------------------------ résolution

    def resoudre(self, requete: str) -> PerimetreDocumentaire:
        """Détermine le périmètre documentaire visé par une question."""
        if self.est_vide:
            return PerimetreDocumentaire(statut="aucun", raison="catalogue_vide")

        requete_normalisee = _normaliser_texte(requete)
        jetons_requete = requete_normalisee.split()
        if not jetons_requete:
            return PerimetreDocumentaire(statut="aucun", raison="requete_vide")

        annees_pub, annees_val = _annees_de_publication(jetons_requete)
        ensemble_requete = set(jetons_requete)
        marqueur = any(marque in requete_normalisee for marque in _MARQUEURS_PORTEE)

        classement: list[tuple[float, FicheDocument, tuple[str, ...]]] = []
        for fiche in self.fiches:
            resultat = self._meilleure_facette(
                fiche, requete_normalisee, ensemble_requete
            )
            if resultat is None:
                continue
            couverture, reconnus = resultat
            if marqueur:
                couverture = min(1.0, couverture * 1.15)
            classement.append((couverture, fiche, reconnus))

        if not classement:
            return PerimetreDocumentaire(
                statut="aucun",
                raison="aucune_correspondance",
                annees_publication=tuple(sorted(annees_pub)),
                annees_valeur=tuple(sorted(annees_val)),
            )

        classement.sort(key=lambda element: (-element[0], element[1].libelle))
        meilleur_score, _, reconnus = classement[0]

        if meilleur_score < _SEUIL_RESOLUTION:
            return PerimetreDocumentaire(
                statut="aucun",
                score=round(meilleur_score, 4),
                raison="score_insuffisant",
                annees_publication=tuple(sorted(annees_pub)),
                annees_valeur=tuple(sorted(annees_val)),
            )

        # Les documents à égalité sur les mêmes jetons forment un périmètre
        # unique : deux exercices d'une même organisation ne sont pas des
        # candidats concurrents, ils répondent à la même intention.
        groupe = [
            fiche
            for score, fiche, jetons in classement
            if abs(score - meilleur_score) < 1e-9 and jetons == reconnus
        ]
        restants = [
            (score, fiche)
            for score, fiche, jetons in classement
            if not (abs(score - meilleur_score) < 1e-9 and jetons == reconnus)
        ]
        score_second, second = (
            (restants[0][0], restants[0][1].libelle) if restants else (0.0, None)
        )

        if meilleur_score - score_second < _MARGE_RESOLUTION:
            logger.info(
                "Résolution ambiguë entre %r et %r (%.3f contre %.3f) : "
                "aucun filtre documentaire appliqué.",
                groupe[0].libelle,
                second,
                meilleur_score,
                score_second,
            )
            return PerimetreDocumentaire(
                statut="ambigu",
                libelles=tuple(fiche.libelle for fiche in groupe),
                score=round(meilleur_score, 4),
                jetons_reconnus=reconnus,
                concurrent=second,
                score_concurrent=round(score_second, 4),
                raison="marge_insuffisante",
                annees_publication=tuple(sorted(annees_pub)),
                annees_valeur=tuple(sorted(annees_val)),
            )

        groupe = self._affiner_par_annee(groupe, annees_pub)

        return self._perimetre(
            groupe,
            statut="exact" if len(groupe) == 1 else "compatible",
            score=round(meilleur_score, 4),
            jetons_reconnus=reconnus,
            origine="detection",
            concurrent=second,
            score_concurrent=round(score_second, 4),
            annees_publication=tuple(sorted(annees_pub)),
            annees_valeur=tuple(sorted(annees_val)),
        )

    def perimetre_explicite(self, identifiants: Sequence[str]) -> PerimetreDocumentaire:
        """Construit un périmètre imposé par l'appelant, sans résolution."""
        fiches: list[FicheDocument] = []
        for identifiant in identifiants:
            fiche = self.par_identifiant(identifiant)
            if fiche is None:
                connus = ", ".join(sorted(f.libelle for f in self.fiches)[:20])
                raise DocumentInconnu(
                    f"Document inconnu dans la collection : {identifiant!r}. "
                    f"Documents indexés : {connus or 'aucun'}."
                )
            fiches.append(fiche)

        jetons: set[str] = set()
        for fiche in fiches:
            jetons |= fiche.jetons

        return self._perimetre(
            fiches,
            statut="exact" if len(fiches) == 1 else "compatible",
            score=1.0,
            jetons_reconnus=tuple(sorted(jetons)),
            origine="explicite",
        )

    # --------------------------------------------------------------- helpers

    def _affiner_par_annee(
        self,
        groupe: list[FicheDocument],
        annees_publication: set[int],
    ) -> list[FicheDocument]:
        """Restreint un périmètre à l'exercice de publication demandé.

        L'affinage ne s'applique qu'à des documents déjà retenus sur leur
        identité, et jamais à partir de l'année d'une valeur recherchée. Si
        aucun document ne porte l'année demandée — métadonnée absente ou
        millésime inconnu — le périmètre initial est conservé : mieux vaut un
        périmètre trop large qu'un périmètre vide qui ferait échouer la
        réponse à tort.
        """
        if not annees_publication or len(groupe) < 2:
            return groupe

        retenus = [
            fiche
            for fiche in groupe
            if set(fiche.annees) & annees_publication
        ]
        if not retenus:
            logger.info(
                "Aucun document du périmètre ne porte l'année de publication "
                "%s : périmètre initial conservé.",
                sorted(annees_publication),
            )
            return groupe

        logger.debug(
            "Périmètre affiné par année de publication %s : %d document(s).",
            sorted(annees_publication),
            len(retenus),
        )
        return retenus

    def _perimetre(
        self,
        fiches: list[FicheDocument],
        *,
        statut: str,
        score: float,
        jetons_reconnus: tuple[str, ...],
        origine: str,
        concurrent: str | None = None,
        score_concurrent: float = 0.0,
        annees_publication: tuple[int, ...] = (),
        annees_valeur: tuple[int, ...] = (),
    ) -> PerimetreDocumentaire:
        champ = self._champ_filtre(fiches)
        valeurs = tuple(fiche.identifiant_pour(champ) for fiche in fiches)
        if not all(valeurs):
            # Aucun champ exploitable pour tout le périmètre : on préfère ne
            # pas filtrer plutôt que d'exclure silencieusement un document.
            return PerimetreDocumentaire(
                statut="ambigu",
                libelles=tuple(fiche.libelle for fiche in fiches),
                score=score,
                jetons_reconnus=jetons_reconnus,
                origine=origine,
                raison="identifiant_indisponible",
                annees_publication=annees_publication,
                annees_valeur=annees_valeur,
            )

        return PerimetreDocumentaire(
            statut=statut,
            libelles=tuple(fiche.libelle for fiche in fiches),
            champ_filtre=champ,
            valeurs_filtre=valeurs,
            score=score,
            jetons_reconnus=jetons_reconnus,
            origine=origine,
            concurrent=concurrent,
            score_concurrent=score_concurrent,
            annees_publication=annees_publication,
            annees_valeur=annees_valeur,
        )

    @staticmethod
    def _champ_filtre(fiches: Sequence[FicheDocument]) -> str:
        """Choisit la clé de payload la plus fiable pour cloisonner.

        L'identifiant stable prime ; le nom de fichier puis le chemin servent
        de repli quand l'ingestion ne l'a pas renseigné. Le champ retenu doit
        être exploitable pour tous les documents du périmètre, sans quoi le
        filtre Qdrant en exclurait un silencieusement.
        """
        filtrables = set(champs_filtrables())
        candidats = [fiches[0].champ_id, *_CHAMPS_PORTEE_DOCUMENT] if fiches else list(
            _CHAMPS_PORTEE_DOCUMENT
        )
        for champ in dict.fromkeys(candidats):
            if champ not in filtrables:
                continue
            if all(fiche.identifiant_pour(champ) for fiche in fiches):
                return champ
        return "doc_id"

    def _meilleure_facette(
        self,
        fiche: FicheDocument,
        requete: str,
        jetons_requete: set[str],
    ) -> tuple[float, tuple[str, ...]] | None:
        """Meilleur taux de couverture parmi les facettes d'identité.

        Un utilisateur nomme un document par une seule de ses facettes :
        l'organisation, le titre, le nom de fichier ou un sigle. Retenir le
        maximum, et non l'union, évite de pénaliser les corpus dont les
        métadonnées sont les plus complètes.
        """
        meilleur: tuple[float, tuple[str, ...]] | None = None

        for signature in fiche.signatures:
            poids_total = sum(self.idf.get(jeton, 0.0) for jeton in signature)
            if poids_total <= 0.0:
                continue

            reconnus = tuple(
                sorted(
                    jeton
                    for jeton in signature
                    if self._jeton_present(jeton, requete, jetons_requete)
                )
            )
            if not reconnus:
                continue

            couverture = sum(self.idf.get(j, 0.0) for j in reconnus) / poids_total
            if meilleur is None or couverture > meilleur[0]:
                meilleur = (couverture, reconnus)

        return meilleur

    @staticmethod
    def _jeton_present(jeton: str, requete: str, jetons_requete: set[str]) -> bool:
        """Correspondance exacte, ou sous-chaîne pour un jeton long et rare."""
        if jeton in jetons_requete:
            return True
        return len(jeton) >= 6 and jeton in requete


def _documents_depuis_vectorstore(profil: Profil | None = None) -> list[FicheDocument]:
    """Lit l'identité des documents distincts depuis la collection."""
    if _lister_documents is None:
        logger.warning(
            "vectorstore.lister_documents() est absent : la résolution "
            "documentaire est désactivée et la recherche reste globale."
        )
        return []

    try:
        entrees = _lister_documents(champs=cles_catalogue(profil))
    except TypeError:
        # Signature plus ancienne, sans sélection de champs.
        entrees = _lister_documents()

    fiches: list[FicheDocument] = []
    vus: set[str] = set()

    for entree in entrees:
        payload = entree if isinstance(entree, dict) else getattr(entree, "payload", None)
        if not isinstance(payload, dict):
            continue

        fiche = _fiche_depuis_payload(payload)
        if fiche is None or fiche.document_id in vus:
            continue
        vus.add(fiche.document_id)
        fiches.append(fiche)

    return fiches


def catalogue(*, forcer: bool = False, profil: Profil | None = None) -> CatalogueDocuments:
    """Renvoie le catalogue documentaire, construit une seule fois."""
    global _catalogue
    if _catalogue is None or forcer:
        _catalogue = CatalogueDocuments.construire(_documents_depuis_vectorstore(profil))
        logger.debug(
            "Catalogue documentaire construit : %d document(s).",
            len(_catalogue.fiches),
        )
    return _catalogue


def reinitialiser_catalogue() -> None:
    """Invalide le cache du catalogue, à appeler après une ingestion."""
    global _catalogue
    _catalogue = None


def resoudre_document(requete: str) -> PerimetreDocumentaire:
    """Point d'entrée public de la résolution, réutilisable par l'agent."""
    return catalogue().resoudre(requete)


def identite_document(payload: dict[str, Any]) -> dict[str, str]:
    """Facettes d'identité lisibles d'un payload de chunk.

    Exposée publiquement pour que la génération et la validation étiquettent
    les extraits sans réimplémenter la lecture des métadonnées : les mêmes
    conventions de nommage servent partout, et un corpus qui n'expose que
    certaines facettes reste correctement décrit.
    """
    identite: dict[str, str] = {}
    for facette in ("titre", "nom_fichier", "organisation", "type_document", "annee"):
        valeur = _texte_facette(payload, facette).strip()
        if valeur:
            identite[facette] = valeur

    # Le libellé de document privilégie ce qu'un lecteur reconnaîtra.
    identite["document"] = (
        identite.get("titre")
        or identite.get("nom_fichier")
        or _texte_facette(payload, "document_id")
        or _texte_facette(payload, "source")
        or "document inconnu"
    )
    return identite


def _identifiants_dans_filtres(filtres: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    """Extrait un éventuel cloisonnement documentaire posé par l'appelant."""
    for champ in _CHAMPS_PORTEE_DOCUMENT:
        valeur = filtres.get(champ)
        if isinstance(valeur, str) and valeur.strip():
            return champ, (valeur.strip(),)
        if isinstance(valeur, (list, tuple)) and valeur:
            valeurs = tuple(str(v).strip() for v in valeur if str(v).strip())
            if valeurs:
                return champ, valeurs
    return "", ()


def _perimetre_depuis_filtres(
    champ: str,
    valeurs: tuple[str, ...],
) -> PerimetreDocumentaire:
    """Reconstitue le périmètre lorsqu'il vient des critères de l'appelant."""
    fiches = [catalogue().par_identifiant(valeur) for valeur in valeurs]
    jetons: set[str] = set()
    for fiche in fiches:
        if fiche is not None:
            jetons |= fiche.jetons

    return PerimetreDocumentaire(
        statut="exact" if len(valeurs) == 1 else "compatible",
        libelles=tuple(
            fiche.libelle if fiche is not None else valeur
            for fiche, valeur in zip(fiches, valeurs)
        ),
        champ_filtre=champ,
        valeurs_filtre=valeurs,
        score=1.0,
        jetons_reconnus=tuple(sorted(jetons)),
        origine="explicite",
    )


# ===========================================================================
# Transformation des résultats Qdrant
# ===========================================================================


def _empreinte_texte(texte: str) -> str:
    """Empreinte légère pour supprimer les doublons textuels exacts."""
    compact = " ".join(texte.casefold().split())
    return hashlib.sha1(compact.encode("utf-8"), usedforsecurity=False).hexdigest()


def _passage_depuis_resultat(
    resultat: Resultat,
    rang: int,
    score_reranking: float | None,
) -> Passage:
    payload = dict(resultat.payload)
    page = payload.get("page")
    chunk_index = payload.get("chunk_index")

    return Passage(
        citation=f"S{rang}",
        rang=rang,
        point_id=resultat.point_id,
        doc_id=str(payload.get("doc_id", "")),
        chunk_index=int(chunk_index) if isinstance(chunk_index, int) else None,
        texte=resultat.texte.strip(),
        source=str(payload.get("source", "")),
        nom_fichier=str(payload.get("nom_fichier", "")),
        page=int(page) if isinstance(page, int) else None,
        categorie=str(payload.get("categorie", "")),
        score_recherche=float(resultat.score),
        score_reranking=score_reranking,
        payload=payload,
    )


def _selectionner_diversifie(
    candidats: list[tuple[Resultat, float | None]],
    limite: int,
    max_par_document: int,
) -> list[tuple[Resultat, float | None]]:
    """
    Supprime les doublons exacts et limite la domination d'un document.

    Les chunks se chevauchent volontairement à l'ingestion. Sans cette étape,
    plusieurs résultats presque identiques d'un même document peuvent occuper
    tout le contexte et masquer d'autres sources pertinentes.

    Lorsque la question vise un document unique, l'appelant neutralise le
    plafond par document : la diversification n'a alors plus de sens, elle
    priverait la réponse de passages pertinents de la seule source autorisée.
    La déduplication, elle, reste toujours utile.
    """
    retenus: list[tuple[Resultat, float | None]] = []
    empreintes_vues: set[str] = set()
    par_document: dict[str, int] = {}

    for resultat, score_reranking in candidats:
        texte = resultat.texte.strip()
        if not texte:
            continue

        empreinte = _empreinte_texte(texte)
        if empreinte in empreintes_vues:
            continue

        doc_id = resultat.doc_id or resultat.source or resultat.point_id
        if par_document.get(doc_id, 0) >= max_par_document:
            continue

        retenus.append((resultat, score_reranking))
        empreintes_vues.add(empreinte)
        par_document[doc_id] = par_document.get(doc_id, 0) + 1

        if len(retenus) >= limite:
            break

    return retenus


def _cle_chunk(resultat: Resultat) -> tuple[str, int]:
    """Position d'un chunk dans son document, pour l'ordre et la déduplication."""
    index = (resultat.payload or {}).get("chunk_index")
    return (resultat.doc_id or resultat.source, int(index) if isinstance(index, int) else -1)


def etendre_contexte(
    retenus: list[tuple[Resultat, float | None]],
    *,
    rayon: int,
    max_chunks_ajoutes: int,
    taille_max_contexte: int,
    recuperer: Callable[..., list[Resultat]] = recuperer_contexte,
) -> list[tuple[Resultat, float | None]]:
    """
    Complète les passages retenus par leur contexte immédiat.

    Pour chaque chunk retenu :

    - s'il porte un ``parent_id``, tous les chunks du même parent sont
      récupérés — c'est ce qui reconstitue un tableau entier à partir d'une
      seule ligne trouvée ;
    - sinon, les chunks voisins du même document sont récupérés dans un
      rayon configurable.

    Les ajouts portent un score de reranking à ``None`` : ils n'ont pas été
    jugés pertinents par le reranker, ils ne servent qu'à compléter le
    contexte. Ils sont insérés à leur place naturelle, juste après le chunk
    qui les a appelés, afin que l'ordre de lecture reste cohérent.

    Le nombre et la taille des ajouts sont plafonnés : l'objectif est de
    donner au LLM un contexte complet, pas de lui envoyer le corpus.
    """
    if not retenus or (rayon <= 0 and max_chunks_ajoutes <= 0):
        return retenus

    deja_presents = {resultat.point_id for resultat, _ in retenus}
    budget_caracteres = taille_max_contexte - sum(
        len(resultat.texte) for resultat, _ in retenus
    )
    ajoutes = 0
    resultat_final: list[tuple[Resultat, float | None]] = []
    groupes_traites: set[str] = set()

    for resultat, score in retenus:
        resultat_final.append((resultat, score))

        if ajoutes >= max_chunks_ajoutes or budget_caracteres <= 0:
            continue

        payload = resultat.payload or {}
        parent_id = payload.get("parent_id")
        doc_id = resultat.doc_id
        if not doc_id:
            continue

        cle_groupe = f"{doc_id}|{parent_id}" if parent_id else ""
        if cle_groupe and cle_groupe in groupes_traites:
            continue

        try:
            if parent_id:
                groupes_traites.add(cle_groupe)
                voisins = recuperer(doc_id, parent_id=str(parent_id))
            else:
                _, index = _cle_chunk(resultat)
                if index < 0:
                    continue
                indices = [
                    i for i in range(index - rayon, index + rayon + 1) if i >= 0 and i != index
                ]
                voisins = recuperer(doc_id, indices=indices)
        except Exception as exc:  # noqa: BLE001 — le contexte est un bonus
            logger.warning("Expansion du contexte impossible (%s) : %s", doc_id, exc)
            continue

        for voisin in sorted(voisins, key=_cle_chunk):
            if voisin.point_id in deja_presents:
                continue
            if ajoutes >= max_chunks_ajoutes or len(voisin.texte) > budget_caracteres:
                break
            resultat_final.append((voisin, None))
            deja_presents.add(voisin.point_id)
            budget_caracteres -= len(voisin.texte)
            ajoutes += 1

    if ajoutes:
        logger.debug("Expansion du contexte : %d chunks ajoutés.", ajoutes)
    return resultat_final


def _hors_perimetre(resultat: Resultat, perimetre: PerimetreDocumentaire) -> bool:
    """Dernier garde-fou : un chunk hors périmètre ne doit pas passer.

    Le filtre Qdrant suffit normalement, mais un payload incomplet ou un
    index manquant ne doit jamais aboutir à une valeur attribuée à la
    mauvaise organisation. Ce contrôle est volontairement redondant.
    """
    payload = resultat.payload or {}
    valeurs = {
        str(payload.get(champ, "")).strip()
        for champ in (*_CHAMPS_PORTEE_DOCUMENT, perimetre.champ_filtre)
        if champ
    }
    valeurs.discard("")
    if not valeurs:
        return False  # payload trop pauvre pour trancher : on ne rejette pas
    return not perimetre.contient(*valeurs)


# ===========================================================================
# Recherche principale
# ===========================================================================


def rechercher_passages(
    requete: str,
    criteres: dict[str, Any] | None = None,
    *,
    profil: Profil | None = None,
    top_k: int | None = None,
    limite_candidats: int | None = None,
    utiliser_reranker: bool = True,
    appliquer_seuil: bool = False,
    seuil_pertinence: float | None = None,
    max_par_document: int = 3,
    documents: str | Sequence[str] | None = None,
    resolution_document: bool = True,
) -> RapportRecherche:
    """
    Exécute la récupération complète d'un RAG.

    Le seuil de pertinence est appliqué uniquement au score du reranker,
    normalisé entre 0 et 1. Il n'est pas appliqué au score RRF de Qdrant,
    car un score RRF n'a pas la même échelle qu'une similarité cosinus.

    Cloisonnement documentaire :
        - ``documents`` impose un périmètre et court-circuite la résolution ;
        - sinon, si la question désigne de façon fiable un ou plusieurs
          documents du catalogue, la recherche Qdrant y est restreinte et la
          diversification est désactivée pour un document unique ;
        - une résolution ambiguë ne pose aucun filtre : la recherche reste
          globale, car un filtre erroné masquerait la bonne réponse ;
        - si la recherche cloisonnée ne renvoie rien, le rapport est vide et
          porte un ``motif_absence``. Aucune reprise sans filtre n'est
          tentée : répondre depuis un autre document serait factuellement
          faux même si la citation est syntaxiquement valide.
    """
    debut = time.perf_counter()
    requete = " ".join(str(requete).split())
    if not requete:
        raise ErreurRecherche("La requête de recherche est vide.")
    if max_par_document < 1:
        raise ValueError("max_par_document doit être supérieur ou égal à 1.")

    profil = profil or get_profil()
    cfg = get_config_technique()
    settings = get_settings()

    infos = info_collection()
    if not infos.get("existe"):
        raise CollectionIndisponible(
            "La collection Qdrant n'existe pas. Lance d'abord l'ingestion."
        )
    if not infos.get("points", 0):
        raise CollectionIndisponible(
            "La collection Qdrant est vide. Indexe au moins un document avant la recherche."
        )

    filtres_prepares = preparer_filtres(criteres, profil)

    top_k_final = top_k or cfg.recherche.top_k_final
    if top_k_final < 1:
        raise ValueError("top_k doit être supérieur ou égal à 1.")

    # --- Détermination du périmètre documentaire ---------------------------
    champ_impose, valeurs_imposees = _identifiants_dans_filtres(filtres_prepares)

    if documents:
        demandes = [documents] if isinstance(documents, str) else list(documents)
        perimetre = catalogue(profil=profil).perimetre_explicite(demandes)
    elif valeurs_imposees:
        # L'appelant a déjà cloisonné : on respecte son choix sans le rejouer.
        perimetre = _perimetre_depuis_filtres(champ_impose, valeurs_imposees)
    elif resolution_document:
        perimetre = catalogue(profil=profil).resoudre(requete)
    else:
        perimetre = PerimetreDocumentaire(statut="aucun", raison="resolution_desactivee")

    if perimetre.contraignant:
        filtres_prepares[perimetre.champ_filtre] = (
            perimetre.valeurs_filtre[0]
            if perimetre.unique
            else list(perimetre.valeurs_filtre)
        )
        logger.info(
            "Recherche cloisonnée sur %s (statut=%s, origine=%s, score=%.3f).",
            perimetre.libelle,
            perimetre.statut,
            perimetre.origine,
            perimetre.score,
        )
    elif perimetre.statut == "ambigu":
        logger.info(
            "Périmètre ambigu (%s) : recherche globale conservée.",
            perimetre.raison or "sans précision",
        )

    filtre_qdrant = construire_filtre(filtres_prepares)

    # La diversification est neutralisée dès lors qu'une source unique est
    # imposée : plafonner les passages d'un document n'a plus de sens quand
    # c'est le seul document autorisé. Sur un périmètre de plusieurs
    # documents, le plafond garde son utilité.
    diversification_active = not (perimetre.contraignant and perimetre.unique)
    max_par_document_effectif = max_par_document if diversification_active else top_k_final

    candidats_voulus = limite_candidats or max(
        top_k_final * 4,
        cfg.recherche.top_k_dense,
        cfg.recherche.top_k_sparse,
    )
    candidats_voulus = max(top_k_final, min(candidats_voulus, 100))

    def _rapport(
        passages: list[Passage],
        candidats: int,
        reranking: bool,
        seuil: float | None,
        motif: str | None,
    ) -> RapportRecherche:
        return RapportRecherche(
            requete=requete,
            profil=profil.profile_name,
            filtres=filtres_prepares,
            passages=passages,
            candidats_recuperes=candidats,
            reranking_utilise=reranking,
            seuil_applique=seuil,
            duree_secondes=round(time.perf_counter() - debut, 4),
            perimetre=perimetre,
            diversification_active=diversification_active,
            motif_absence=motif,
        )

    dense, sparse = encoder_requete(
        requete,
        avec_sparse=cfg.qdrant.sparse_active,
    )
    resultats = rechercher(
        dense=dense,
        sparse=sparse if cfg.qdrant.sparse_active else None,
        filtre=filtre_qdrant,
        limite=candidats_voulus,
    )

    # --- Repli sûr ---------------------------------------------------------
    if perimetre.contraignant:
        resultats = [r for r in resultats if not _hors_perimetre(r, perimetre)]
        if not resultats:
            logger.warning(
                "Aucun passage trouvé dans le périmètre %s : la recherche "
                "n'est pas relancée sans filtre.",
                perimetre.libelle,
            )
            return _rapport([], 0, False, None, "perimetre_sans_passage")

    if not resultats:
        return _rapport([], 0, False, None, "aucun_candidat")

    reranking_actif = bool(
        utiliser_reranker
        and settings.reranker_enabled
        and resultats
    )

    candidats_tries: list[tuple[Resultat, float | None]]
    seuil_effectif: float | None = None

    if reranking_actif:
        classement = reranker(
            requete,
            [resultat.texte for resultat in resultats],
        )
        candidats_tries = [
            (resultats[indice], float(score))
            for indice, score in classement
        ]

        if appliquer_seuil:
            seuil_effectif = (
                cfg.recherche.score_min
                if seuil_pertinence is None
                else seuil_pertinence
            )
            candidats_tries = [
                (resultat, score)
                for resultat, score in candidats_tries
                if score is not None and score >= seuil_effectif
            ]
    else:
        # Qdrant renvoie déjà les résultats par pertinence décroissante.
        # Aucun score_min n'est appliqué : en hybride RRF, le score brut
        # n'est pas une probabilité et n'est pas comparable à 0,30.
        candidats_tries = [(resultat, None) for resultat in resultats]

    retenus = _selectionner_diversifie(
        candidats_tries,
        limite=top_k_final,
        max_par_document=max_par_document_effectif,
    )

    cfg_voisins = cfg.decoupage.voisins
    if cfg_voisins.actif and retenus:
        retenus = etendre_contexte(
            retenus,
            rayon=cfg_voisins.rayon,
            max_chunks_ajoutes=cfg_voisins.max_chunks_ajoutes,
            taille_max_contexte=cfg_voisins.taille_max_contexte,
        )
        if perimetre.contraignant:
            # Un voisin reste soumis au cloisonnement documentaire.
            retenus = [
                (resultat, score)
                for resultat, score in retenus
                if not _hors_perimetre(resultat, perimetre)
            ]

    passages = [
        _passage_depuis_resultat(resultat, rang, score_reranking)
        for rang, (resultat, score_reranking) in enumerate(retenus, start=1)
    ]

    motif: str | None = None
    if not passages:
        motif = (
            "perimetre_sans_passage"
            if perimetre.contraignant
            else "aucun_passage_retenu"
        )

    return _rapport(
        passages,
        len(resultats),
        reranking_actif,
        seuil_effectif,
        motif,
    )


# ===========================================================================
# Affichage et CLI
# ===========================================================================


def afficher_rapport(rapport: RapportRecherche) -> None:
    print("\n" + "=" * 72)
    print(f"RECHERCHE — profil « {rapport.profil} »")
    print("=" * 72)
    print(f"Requête             : {rapport.requete}")
    print(f"Filtres             : {rapport.filtres or '{}'}")

    perimetre = rapport.perimetre
    if perimetre is not None and perimetre.statut != "aucun":
        print(
            f"Périmètre           : {perimetre.libelle} "
            f"[{perimetre.statut}, origine={perimetre.origine}, "
            f"score={perimetre.score:.3f}, marge={perimetre.marge:.3f}]"
        )
        if perimetre.jetons_reconnus:
            print(f"Jetons reconnus     : {', '.join(perimetre.jetons_reconnus)}")
        if perimetre.annees_publication:
            print(
                "Année(s) publication: "
                + ", ".join(str(a) for a in perimetre.annees_publication)
            )
        if perimetre.annees_valeur:
            print(
                "Année(s) de valeur  : "
                + ", ".join(str(a) for a in perimetre.annees_valeur)
                + "  (non utilisée(s) pour filtrer)"
            )
        if perimetre.raison:
            print(f"Raison              : {perimetre.raison}")
    else:
        print("Périmètre           : — (recherche globale)")

    print(f"Diversification     : {'oui' if rapport.diversification_active else 'non'}")
    print(f"Candidats Qdrant    : {rapport.candidats_recuperes}")
    print(f"Reranking           : {'oui' if rapport.reranking_utilise else 'non'}")
    print(f"Seuil appliqué      : {rapport.seuil_applique}")
    print(f"Durée               : {rapport.duree_secondes:.3f}s")
    print(f"Passages retenus    : {len(rapport.passages)}")

    if rapport.motif_absence:
        print(f"Motif d'absence     : {rapport.motif_absence}")

    for passage in rapport.passages:
        extrait = " ".join(passage.texte.split())
        if len(extrait) > 260:
            extrait = extrait[:257] + "..."
        score = passage.score_final
        print("\n" + "-" * 72)
        print(
            f"[{passage.citation}] score={score:.4f} | "
            f"{passage.localisation} | catégorie={passage.categorie or '-'}"
        )
        print(extrait)

    print("=" * 72 + "\n")


def _charger_json_objet(texte: str | None) -> dict[str, Any] | None:
    if not texte:
        return None
    try:
        valeur = json.loads(texte)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"JSON de filtres invalide : {exc}") from exc
    if not isinstance(valeur, dict):
        raise argparse.ArgumentTypeError("--filtres doit contenir un objet JSON.")
    return valeur


def main() -> None:
    parseur = argparse.ArgumentParser(description="Recherche hybride dans le corpus")
    parseur.add_argument("requete", help="question ou formulation de recherche")
    parseur.add_argument("--filtres", default=None, help="objet JSON de filtres Qdrant")
    parseur.add_argument("--profil", default=None, help="profil YAML à utiliser")
    parseur.add_argument("--top-k", type=int, default=None)
    parseur.add_argument("--candidats", type=int, default=None)
    parseur.add_argument("--sans-reranker", action="store_true")
    parseur.add_argument(
        "--avec-seuil",
        action="store_true",
        help="Applique le seuil numérique du reranker.",
    )
    parseur.add_argument(
        "--document",
        action="append",
        default=None,
        dest="documents",
        help="Restreint la recherche à un document (répétable).",
    )
    parseur.add_argument(
        "--sans-resolution",
        action="store_true",
        help="Désactive la résolution automatique du périmètre documentaire.",
    )
    parseur.add_argument("--verbose", action="store_true")
    args = parseur.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    try:
        rapport = rechercher_passages(
            requete=args.requete,
            criteres=_charger_json_objet(args.filtres),
            profil=get_profil(args.profil),
            top_k=args.top_k,
            limite_candidats=args.candidats,
            utiliser_reranker=not args.sans_reranker,
            appliquer_seuil=args.avec_seuil,
            documents=args.documents,
            resolution_document=not args.sans_resolution,
        )
        afficher_rapport(rapport)
    except ErreurRecherche as exc:
        logger.error("Recherche impossible : %s", exc)
        raise SystemExit(1) from exc
    finally:
        fermer_client()


if __name__ == "__main__":
    main()