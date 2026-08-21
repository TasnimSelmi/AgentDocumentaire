"""
Outil de recherche documentaire de l'agent.

Cet outil est une façade légère autour de ``src.rag.retrieval`` :
il ne réimplémente ni les embeddings, ni Qdrant, ni le reranking,
ni la résolution documentaire.

Le schéma exposé au LLM est construit dynamiquement depuis les champs
filtrables du profil technique actif. Un changement de profil reconstruit
donc automatiquement un outil ``search`` adapté au corpus courant.

Chaîne d'exécution :

    appel LLM
        ↓
    search(...)
        ↓
    rechercher_passages(...)
        ↓
    BGE-M3 / Qdrant / reranking
        ↓
    RapportRecherche
        ↓
    ResultatOutil
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, create_model

from src.config import get_profil
from src.rag.retrieval import (
    CollectionIndisponible,
    DocumentInconnu,
    ErreurRecherche,
    FiltreInvalide,
    champs_filtrables,
    rechercher_passages,
)
from src.tools.base import (
    ContexteOutil,
    DefinitionOutil,
    ResultatOutil,
    SourceOutil,
    outil,
)


# ===========================================================================
# 1. Types utilisables dans le schéma dynamique
# ===========================================================================


class PlageEntier(BaseModel):
    """Intervalle optionnel pour un champ entier."""

    gte: int | None = Field(default=None, description="Borne minimale incluse.")
    lte: int | None = Field(default=None, description="Borne maximale incluse.")
    gt: int | None = Field(default=None, description="Borne minimale exclue.")
    lt: int | None = Field(default=None, description="Borne maximale exclue.")


class PlageNombre(BaseModel):
    """Intervalle optionnel pour un champ numérique."""

    gte: float | None = Field(default=None, description="Borne minimale incluse.")
    lte: float | None = Field(default=None, description="Borne maximale incluse.")
    gt: float | None = Field(default=None, description="Borne minimale exclue.")
    lt: float | None = Field(default=None, description="Borne maximale exclue.")


class PlageDate(BaseModel):
    """
    Intervalle de dates.

    Les chaînes sont volontairement conservées telles quelles : la
    normalisation définitive appartient à retrieval.preparer_filtres().
    """

    gte: str | None = Field(
        default=None,
        description="Date minimale incluse.",
    )
    lte: str | None = Field(
        default=None,
        description="Date maximale incluse.",
    )
    gt: str | None = Field(
        default=None,
        description="Date minimale exclue.",
    )
    lt: str | None = Field(
        default=None,
        description="Date maximale exclue.",
    )


# ===========================================================================
# 2. Construction dynamique du schéma
# ===========================================================================


def _type_argument(type_champ: str) -> Any:
    """
    Traduit un type déclaré par le profil en type Pydantic pour le LLM.

    La validation métier définitive reste dans retrieval.py. Le schéma de
    l'outil sert surtout à guider correctement le tool calling.
    """

    type_champ = str(type_champ).strip().lower()

    if type_champ == "booleen":
        return bool | list[bool] | None

    if type_champ == "entier":
        return int | list[int] | PlageEntier | None

    if type_champ == "nombre":
        return float | list[float] | PlageNombre | None

    if type_champ == "date":
        return str | list[str] | PlageDate | None

    # texte, categorie et autres types scalaires.
    return str | list[str] | None


def _description_filtre(nom: str, type_champ: str) -> str:
    """Description courte destinée au schéma visible par le LLM."""

    descriptions = {
        "texte": (
            "Valeur exacte à filtrer. "
            "Une liste signifie que plusieurs valeurs sont acceptées."
        ),
        "booleen": (
            "Filtre booléen. Une liste peut être utilisée pour accepter "
            "plusieurs valeurs."
        ),
        "entier": (
            "Valeur entière, liste d'entiers ou intervalle "
            "avec gte/lte/gt/lt."
        ),
        "nombre": (
            "Valeur numérique, liste de nombres ou intervalle "
            "avec gte/lte/gt/lt."
        ),
        "date": (
            "Date, liste de dates ou intervalle avec gte/lte/gt/lt."
        ),
    }

    detail = descriptions.get(
        str(type_champ).lower(),
        "Valeur utilisée pour restreindre la recherche documentaire.",
    )

    return f"Filtre facultatif « {nom} ». {detail}"


def construire_schema_search() -> type[BaseModel]:
    """
    Construit le schéma de ``search`` depuis le profil technique actif.

    Exemple :

        profil A :
            entreprise, date_document

        profil B :
            client, type_contrat

    La même fabrique produit automatiquement deux schémas différents.
    """

    profil = get_profil()
    filtrables = champs_filtrables(profil)

    champs: dict[str, Any] = {
        "requete": (
            str,
            Field(
                ...,
                min_length=1,
                description=(
                    "Question ou formulation de recherche à retrouver dans "
                    "le corpus documentaire."
                ),
            ),
        ),
    }

    for nom, type_champ in filtrables.items():
        champs[nom] = (
            _type_argument(type_champ),
            Field(
                default=None,
                description=_description_filtre(nom, type_champ),
            ),
        )

    return create_model(
        "ArgumentsSearch",
        **champs,
    )


# ===========================================================================
# 3. Conversion du retrieval vers le contrat des outils
# ===========================================================================


def _source_depuis_passage(passage: Any) -> SourceOutil:
    """Convertit un Passage du RAG en SourceOutil."""

    return SourceOutil(
        doc_id=passage.doc_id,
        source=passage.source,
        nom_fichier=passage.nom_fichier,
        page=passage.page,
        categorie=passage.categorie,
        score=round(float(passage.score_final), 6),
        extrait=passage.texte,
    )


def _passage_pour_agent(passage: Any) -> dict[str, Any]:
    """
    Version compacte d'un passage destinée au contexte de l'agent.

    Le payload Qdrant complet n'est volontairement jamais transmis au LLM.
    """

    return {
        "rang": passage.rang,
        "document": passage.nom_fichier or passage.source or passage.doc_id,
        "page": passage.page,
        "score": round(float(passage.score_final), 4),
        "texte": passage.texte,
    }


# ===========================================================================
# 4. Implémentation
# ===========================================================================


def _executer_search(
    *,
    contexte: ContexteOutil | None = None,
    **kwargs: Any,
) -> ResultatOutil:
    """
    Exécute la recherche documentaire.

    Les filtres sont récupérés directement depuis les arguments dynamiques
    du tool calling puis transmis tels quels à retrieval.py, qui reste le
    responsable unique de leur validation et de leur normalisation.
    """

    requete = str(kwargs.pop("requete", "") or "").strip()

    if not requete:
        return ResultatOutil.echec(
            "search",
            "La requête de recherche est vide.",
        )

    # Tous les autres arguments du schéma sont des filtres facultatifs.
    criteres = {
        cle: valeur
        for cle, valeur in kwargs.items()
        if valeur is not None
    }

    # Les sous-modèles Pydantic des plages doivent être transformés en dict
    # avant d'être envoyés à retrieval.preparer_filtres().
    for cle, valeur in list(criteres.items()):
        if isinstance(valeur, BaseModel):
            criteres[cle] = valeur.model_dump(exclude_none=True)

    profil = get_profil()

    try:
        rapport = rechercher_passages(
            requete=requete,
            criteres=criteres or None,
            profil=profil,
        )

    except FiltreInvalide as exc:
        return ResultatOutil.echec(
            "search",
            f"Filtres invalides : {exc}",
            filtres=criteres,
        )

    except DocumentInconnu as exc:
        return ResultatOutil.echec(
            "search",
            str(exc),
        )

    except CollectionIndisponible as exc:
        return ResultatOutil.echec(
            "search",
            f"Corpus indisponible : {exc}",
        )

    except ErreurRecherche as exc:
        return ResultatOutil.echec(
            "search",
            f"Recherche impossible : {exc}",
        )

    # Aucun résultat n'est pas une erreur technique.
    # L'agent pourra reformuler la requête ou changer ses filtres.
    if rapport.est_vide:
        avertissements: list[str] = []

        if rapport.motif_absence:
            avertissements.append(
                f"Motif d'absence : {rapport.motif_absence}."
            )

        if (
            rapport.perimetre is not None
            and rapport.perimetre.statut == "ambigu"
        ):
            avertissements.append(
                "Le document visé par la requête est ambigu."
            )

        return ResultatOutil(
            outil="search",
            succes=True,
            message=(
                "Aucun passage pertinent n'a été trouvé. "
                "Reformule la recherche ou ajuste les filtres si nécessaire."
            ),
            donnees={
                "requete": rapport.requete,
                "filtres": rapport.filtres,
                "nombre_passages": 0,
            },
            avertissements=avertissements,
        )

    sources = [
        _source_depuis_passage(passage)
        for passage in rapport.passages
    ]

    passages = [
        _passage_pour_agent(passage)
        for passage in rapport.passages
    ]

    donnees: dict[str, Any] = {
        "requete": rapport.requete,
        "nombre_passages": len(passages),
        "passages": passages,
    }

    if rapport.filtres:
        donnees["filtres_appliques"] = rapport.filtres

    if rapport.perimetre is not None and rapport.perimetre.statut != "aucun":
        donnees["perimetre_documentaire"] = {
            "statut": rapport.perimetre.statut,
            "document": rapport.perimetre.libelle,
            "origine": rapport.perimetre.origine,
        }

    avertissements: list[str] = []

    if rapport.perimetre is not None and rapport.perimetre.statut == "ambigu":
        avertissements.append(
            "La requête semble désigner plusieurs documents sans permettre "
            "de choisir un périmètre fiable."
        )

    return ResultatOutil(
        outil="search",
        succes=True,
        message=(
            f"{len(passages)} passage(s) documentaire(s) pertinent(s) trouvé(s)."
        ),
        donnees=donnees,
        sources=sources,
        avertissements=avertissements,
    )


# ===========================================================================
# 5. Définition enregistrée
# ===========================================================================


@outil
def definir_search() -> DefinitionOutil:
    """
    Fabrique l'outil search pour le profil actif.

    Elle est rappelée lors de la reconstruction du registre afin que le
    schéma reflète toujours les champs filtrables du profil courant.
    """

    profil = get_profil()
    schema = construire_schema_search()

    filtrables = champs_filtrables(profil)
    noms_filtres = ", ".join(sorted(filtrables)) or "aucun"

    description = (
        "Recherche des passages pertinents dans le corpus documentaire. "
        "Utilise cet outil lorsque la réponse nécessite de retrouver des "
        "informations présentes dans les documents. "
        "Tu peux préciser des filtres lorsqu'ils sont explicitement connus "
        "dans la demande de l'utilisateur. "
        "N'invente jamais une valeur de filtre. "
        f"Filtres disponibles pour le profil actuel : {noms_filtres}."
    )

    return DefinitionOutil(
        nom="search",
        description=description,
        schema_arguments=schema,
        fonction=_executer_search,
        lecture_seule=True,
        actif=True,
    )