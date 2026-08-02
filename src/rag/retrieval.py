"""
Couche de récupération du RAG : question -> passages pertinents.

Ce module reste volontairement déterministe et sans LLM. Il orchestre :
    1. validation et normalisation des filtres ;
    2. encodage dense + sparse de la requête avec BGE-M3 ;
    3. recherche hybride dans Qdrant ;
    4. reranking des candidats ;
    5. déduplication et diversification des passages ;
    6. attribution d'identifiants de citation stables pour la génération.

Il constitue la frontière entre la base documentaire et la génération.
Plus tard, l'agent appellera cette couche comme un outil, sans dupliquer
la logique de recherche.

Utilisation manuelle :
    python -m src.rag.retrieval "Comment installer le produit ?"
    python -m src.rag.retrieval "Quels rapports datent de 2026 ?" \
        --filtres '{"categorie": "rapport", "date_document": {"gte": "2026-01-01"}}'
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from src.config import Champ, Profil, get_config_technique, get_profil, get_settings
from src.rag.embeddings import encoder_requete, reranker
from src.rag.normalization import normaliser_booleen, normaliser_entier, normaliser_valeur
from src.rag.vectorstore import (
    Resultat,
    construire_filtre,
    fermer_client,
    info_collection,
    rechercher,
)

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


# ===========================================================================
# Exceptions
# ===========================================================================


class ErreurRecherche(RuntimeError):
    """Erreur de haut niveau de la couche de récupération."""


class FiltreInvalide(ErreurRecherche):
    """Un filtre ne correspond pas au profil actif ou à un type attendu."""


class CollectionIndisponible(ErreurRecherche):
    """La collection Qdrant n'existe pas encore ou ne contient aucun point."""


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

    @property
    def est_vide(self) -> bool:
        return not self.passages

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
    appliquer_seuil: bool = True,
    seuil_pertinence: float | None = None,
    max_par_document: int = 3,
) -> RapportRecherche:
    """
    Exécute la récupération complète d'un RAG.

    Le seuil de pertinence est appliqué uniquement au score du reranker,
    normalisé entre 0 et 1. Il n'est pas appliqué au score RRF de Qdrant,
    car un score RRF n'a pas la même échelle qu'une similarité cosinus.
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
    filtre_qdrant = construire_filtre(filtres_prepares)

    top_k_final = top_k or cfg.recherche.top_k_final
    if top_k_final < 1:
        raise ValueError("top_k doit être supérieur ou égal à 1.")

    # On récupère plus de passages que le nombre final afin de laisser au
    # reranker et à la diversification une vraie marge de sélection.
    candidats_voulus = limite_candidats or max(
        top_k_final * 4,
        cfg.recherche.top_k_dense,
        cfg.recherche.top_k_sparse,
    )
    candidats_voulus = max(top_k_final, min(candidats_voulus, 100))

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
        max_par_document=max_par_document,
    )

    passages = [
        _passage_depuis_resultat(resultat, rang, score_reranking)
        for rang, (resultat, score_reranking) in enumerate(retenus, start=1)
    ]

    return RapportRecherche(
        requete=requete,
        profil=profil.profile_name,
        filtres=filtres_prepares,
        passages=passages,
        candidats_recuperes=len(resultats),
        reranking_utilise=reranking_actif,
        seuil_applique=seuil_effectif,
        duree_secondes=round(time.perf_counter() - debut, 4),
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
    print(f"Candidats Qdrant    : {rapport.candidats_recuperes}")
    print(f"Reranking           : {'oui' if rapport.reranking_utilise else 'non'}")
    print(f"Seuil appliqué      : {rapport.seuil_applique}")
    print(f"Durée               : {rapport.duree_secondes:.3f}s")
    print(f"Passages retenus    : {len(rapport.passages)}")

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
    parseur.add_argument("--sans-seuil", action="store_true")
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
            appliquer_seuil=not args.sans_seuil,
        )
        afficher_rapport(rapport)
    except ErreurRecherche as exc:
        logger.error("Recherche impossible : %s", exc)
        raise SystemExit(1) from exc
    finally:
        fermer_client()


if __name__ == "__main__":
    main()