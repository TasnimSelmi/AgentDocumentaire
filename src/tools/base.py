"""
Contrat commun à tous les outils de l'agent.

Un outil se distingue d'une simple fonction par trois choses que le LLM
lit pour décider : son nom, le schéma typé de ses arguments, et sa
description. Ce module fixe ces trois éléments et normalise ce que les
outils renvoient.

Principe directeur : les outils enveloppent les modules rag/, ils ne
réimplémentent rien. retrieval.py reste le module de référence pour la
recherche, testable indépendamment de tout agent.

Deux formes de retour, volontairement distinctes :
  - ResultatOutil : objet structuré, consommé par l'agent
  - rendu_agent() : texte compact, injecté dans le contexte du LLM

Le LLM ne doit jamais recevoir le JSON brut d'un résultat : trop verbeux,
il sature la fenêtre de contexte et noie le signal.

Les outils peuvent partager un ContexteOutil pendant une requête agentique.
Les passages retrouvés par search y sont conservés afin que les outils
suivants puissent les réutiliser sans relancer inutilement le retrieval.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Qwen3 émet un bloc de raisonnement avant sa réponse. Il doit être retiré
# de tout texte destiné à l'utilisateur ou réinjecté dans un prompt.
_RE_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Au-delà, le retour d'un outil occupe une part déraisonnable du contexte.
LIMITE_RENDU_CARACTERES = 6_000


def nettoyer_reflexion(texte: str) -> str:
    """
    Retire le bloc <think> des modèles à raisonnement explicite.

    Un bloc non fermé (génération interrompue par num_predict) laisserait
    une balise ouvrante orpheline : on tronque alors à partir d'elle.
    """
    propre = _RE_THINK.sub("", texte)
    ouvrante = propre.lower().find("<think>")
    if ouvrante != -1:
        propre = propre[:ouvrante]
    return propre.strip()


# ===========================================================================
# 1. Résultat normalisé
# ===========================================================================


@dataclass
class SourceOutil:
    """
    Référence vers un passage réel du corpus.

    Les outils ne formatent pas les citations : ils remontent les sources,
    et la couche de génération décide de leur présentation. C'est ce qui
    permet à l'agent d'agréger les sources de plusieurs appels sans
    collision d'identifiants.
    """

    doc_id: str
    source: str
    nom_fichier: str
    page: int | None = None
    categorie: str = ""
    score: float = 0.0
    extrait: str = ""

    @property
    def localisation(self) -> str:
        nom = self.nom_fichier or self.source or "source inconnue"
        return f"{nom}, page {self.page}" if self.page is not None else nom


@dataclass
class ResultatOutil:
    """
    Retour uniforme de tout outil.

    Un échec n'est pas une exception : l'agent doit pouvoir lire le message
    d'erreur et corriger son appel suivant. Lever une exception ferait
    tomber la boucle entière pour un argument mal formé.
    """

    outil: str
    succes: bool
    message: str = ""
    donnees: dict[str, Any] = field(default_factory=dict)
    sources: list[SourceOutil] = field(default_factory=list)
    avertissements: list[str] = field(default_factory=list)
    duree_secondes: float = 0.0

    @classmethod
    def echec(
        cls,
        outil: str,
        message: str,
        **donnees: Any,
    ) -> ResultatOutil:
        return cls(
            outil=outil,
            succes=False,
            message=message,
            donnees=donnees,
        )

    def vers_dict(self) -> dict[str, Any]:
        return asdict(self)

    def rendu_agent(self, limite: int = LIMITE_RENDU_CARACTERES) -> str:
        """
        Rendu texte destiné au contexte du LLM.

        Le socle transforme les données structurées en texte compact,
        ajoute les sources et avertissements, puis applique une limite
        de taille afin de préserver la fenêtre de contexte du LLM.
        """
        if not self.succes:
            details = _rendu_donnees(self.donnees) if self.donnees else ""
            texte = f"[{self.outil}] ÉCHEC — {self.message}"
            if details:
                texte += f"\n{details}"
            return texte

        morceaux: list[str] = []

        if self.message:
            morceaux.append(self.message)

        if self.donnees:
            morceaux.append(_rendu_donnees(self.donnees))

        if self.sources:
            morceaux.append(
                "Sources : "
                + " | ".join(
                    source.localisation
                    for source in self.sources[:8]
                )
            )

        for avertissement in self.avertissements:
            morceaux.append(f"⚠ {avertissement}")

        texte = "\n".join(
            morceau
            for morceau in morceaux
            if morceau
        )

        if len(texte) > limite:
            texte = texte[: limite - 20].rstrip() + "\n[RENDU TRONQUÉ]"

        return texte


def _rendu_donnees(
    donnees: dict[str, Any],
    indentation: int = 0,
) -> str:
    """Rendu lisible et compact d'un dictionnaire, sans accolades JSON."""
    lignes: list[str] = []
    marge = "  " * indentation

    for cle, valeur in donnees.items():
        if valeur in (None, "", [], {}):
            continue

        if isinstance(valeur, dict):
            lignes.append(f"{marge}{cle} :")
            sous_bloc = _rendu_donnees(
                valeur,
                indentation + 1,
            )
            if sous_bloc:
                lignes.append(sous_bloc)

        elif isinstance(valeur, (list, tuple)):
            if all(
                isinstance(element, (str, int, float, bool))
                for element in valeur
            ):
                lignes.append(
                    f"{marge}{cle} : "
                    + ", ".join(str(element) for element in valeur)
                )
            else:
                lignes.append(f"{marge}{cle} :")
                for element in valeur:
                    if isinstance(element, dict):
                        sous_bloc = _rendu_donnees(
                            element,
                            indentation + 1,
                        )
                        if sous_bloc:
                            lignes.append(sous_bloc)
                    else:
                        lignes.append(f"{marge}  - {element}")

        else:
            lignes.append(f"{marge}{cle} : {valeur}")

    return "\n".join(lignes)


# ===========================================================================
# 2. Contexte partagé entre les outils
# ===========================================================================


@dataclass
class ContexteOutil:
    """
    Contexte partagé entre les outils pendant une requête agentique.

    Les passages retrouvés par ``search`` sont conservés ici afin que
    ``extract``, ``summarize`` et ``classify`` puissent les réutiliser
    sans relancer le retrieval.

    Une instance doit être créée pour chaque requête utilisateur.
    """

    question: str = ""
    llm: Any | None = None
    profil_domaine: Any | None = None

    sources: list[SourceOutil] = field(default_factory=list)
    resultats: list[ResultatOutil] = field(default_factory=list)

    def ajouter_resultat(self, resultat: ResultatOutil) -> None:
        """Conserve un résultat et agrège ses sources sans doublons."""
        self.resultats.append(resultat)

        existantes = {
            (
                source.doc_id,
                source.source,
                source.nom_fichier,
                source.page,
                source.extrait,
            )
            for source in self.sources
        }

        for source in resultat.sources:
            cle = (
                source.doc_id,
                source.source,
                source.nom_fichier,
                source.page,
                source.extrait,
            )

            if cle in existantes:
                continue

            self.sources.append(source)
            existantes.add(cle)

    @property
    def a_des_sources(self) -> bool:
        """Vrai si au moins un passage documentaire est disponible."""
        return bool(self.sources)

    def vider_sources(self) -> None:
        """Vide les passages documentaires accumulés."""
        self.sources.clear()

    def vider(self) -> None:
        """Réinitialise les données transitoires de la requête."""
        self.sources.clear()
        self.resultats.clear()

    def bloc_domaine(self) -> str:
        """Contexte métier optionnel destiné aux outils LLM."""
        if self.profil_domaine is None:
            return ""

        methode = getattr(
            self.profil_domaine,
            "bloc_contexte_domaine",
            None,
        )

        if not callable(methode):
            return ""

        try:
            return str(methode()).strip()
        except Exception:  # noqa: BLE001
            logger.exception(
                "Impossible de construire le contexte du profil de domaine."
            )
            return ""


# ===========================================================================
# 3. Contrat d'un outil
# ===========================================================================


class FonctionOutil(Protocol):
    """Signature attendue de l'implémentation d'un outil."""

    def __call__(
        self,
        *,
        contexte: ContexteOutil | None = None,
        **kwargs: Any,
    ) -> ResultatOutil:
        ...


@dataclass
class DefinitionOutil:
    """
    Description complète d'un outil, indépendante de LangChain.

    Cette indirection évite de coupler le registre à une version précise
    de LangChain : changer de framework agentique ne demanderait que de
    réécrire ``vers_langchain``.
    """

    nom: str
    description: str
    schema_arguments: type[BaseModel]
    fonction: FonctionOutil
    lecture_seule: bool = True
    actif: bool = True

    def executer(
        self,
        *,
        contexte: ContexteOutil | None = None,
        **kwargs: Any,
    ) -> ResultatOutil:
        """
        Exécute l'outil en convertissant toute exception en échec lisible.

        Si un contexte d'exécution est fourni, le résultat et ses sources
        y sont automatiquement conservés pour les appels suivants.
        """
        debut = time.perf_counter()

        try:
            resultat = self.fonction(
                contexte=contexte,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 — un outil ne casse jamais la boucle
            logger.exception(
                "Outil %s en échec",
                self.nom,
            )
            resultat = ResultatOutil.echec(
                self.nom,
                f"{type(exc).__name__} : {exc}",
            )

        if not isinstance(resultat, ResultatOutil):
            resultat = ResultatOutil.echec(
                self.nom,
                (
                    f"Retour invalide : {type(resultat).__name__} "
                    "au lieu de ResultatOutil."
                ),
            )

        resultat.outil = self.nom
        resultat.duree_secondes = round(
            time.perf_counter() - debut,
            4,
        )

        if contexte is not None:
            contexte.ajouter_resultat(resultat)

        return resultat

    def vers_langchain(
        self,
        contexte: ContexteOutil | None = None,
    ) -> BaseTool:
        """
        Convertit en outil LangChain, exploitable par bind_tools.

        Le LLM reçoit seulement le rendu compact de l'outil.
        Le ResultatOutil structuré reste conservé dans ContexteOutil.
        """

        def _adaptateur(**kwargs: Any) -> str:
            resultat = self.executer(
                contexte=contexte,
                **kwargs,
            )
            return resultat.rendu_agent()

        return StructuredTool.from_function(
            func=_adaptateur,
            name=self.nom,
            description=self.description,
            args_schema=self.schema_arguments,
            return_direct=False,
        )


# ===========================================================================
# 4. Registre
# ===========================================================================


class RegistreOutils:
    """
    Collection d'outils disponibles pour l'agent.

    Le registre est reconstruit à chaque changement de profil : les
    schémas d'arguments dépendent des champs filtrables déclarés dans le
    YAML, donc un profil différent produit des outils différents.

    Un contexte partagé peut être associé au registre. Tous les outils
    exécutés via ce registre utilisent alors automatiquement ce contexte.
    """

    def __init__(
        self,
        contexte: ContexteOutil | None = None,
    ) -> None:
        self._outils: dict[str, DefinitionOutil] = {}
        self.contexte = contexte

    def enregistrer(self, definition: DefinitionOutil) -> None:
        if definition.nom in self._outils:
            raise ValueError(
                f"Outil déjà enregistré : {definition.nom}"
            )
        self._outils[definition.nom] = definition

    def obtenir(self, nom: str) -> DefinitionOutil | None:
        return self._outils.get(nom)

    def noms(
        self,
        actifs_seulement: bool = True,
    ) -> list[str]:
        return sorted(
            nom
            for nom, definition in self._outils.items()
            if definition.actif or not actifs_seulement
        )

    def definitions(
        self,
        actifs_seulement: bool = True,
    ) -> list[DefinitionOutil]:
        return [
            definition
            for _, definition in sorted(self._outils.items())
            if definition.actif or not actifs_seulement
        ]

    def vers_langchain(
        self,
        actifs_seulement: bool = True,
    ) -> list[BaseTool]:
        return [
            definition.vers_langchain(
                contexte=self.contexte,
            )
            for definition in self.definitions(actifs_seulement)
        ]

    def executer(
        self,
        nom: str,
        **kwargs: Any,
    ) -> ResultatOutil:
        definition = self.obtenir(nom)

        if definition is None:
            disponibles = ", ".join(self.noms()) or "aucun"
            resultat = ResultatOutil.echec(
                nom,
                f"Outil inconnu. Outils disponibles : {disponibles}.",
            )

            if self.contexte is not None:
                self.contexte.ajouter_resultat(resultat)

            return resultat

        if not definition.actif:
            resultat = ResultatOutil.echec(
                nom,
                "Outil désactivé dans la configuration.",
            )

            if self.contexte is not None:
                self.contexte.ajouter_resultat(resultat)

            return resultat

        return definition.executer(
            contexte=self.contexte,
            **kwargs,
        )

    def bloc_prompt(self) -> str:
        """Liste des outils, pour insertion dans le prompt système de l'agent."""
        return "\n".join(
            (
                f"- {definition.nom} : "
                f"{definition.description.strip().splitlines()[0]}"
            )
            for definition in self.definitions()
        )

    def __len__(self) -> int:
        return len(self._outils)

    def __contains__(self, nom: object) -> bool:
        return nom in self._outils


# ===========================================================================
# 5. Décorateur d'enregistrement
# ===========================================================================


_FABRIQUES: list[Callable[[], DefinitionOutil]] = []


def outil(
    fabrique: Callable[[], DefinitionOutil],
) -> Callable[[], DefinitionOutil]:
    """
    Déclare une fabrique d'outil, appelée à la construction du registre.

    On enregistre une fabrique et non une définition : les schémas
    d'arguments dépendent du profil actif, qui peut changer entre deux
    constructions. Une définition figée à l'import serait périmée.

    Usage dans un module d'outil :

        @outil
        def definir_search() -> DefinitionOutil:
            return DefinitionOutil(...)
    """
    _FABRIQUES.append(fabrique)
    return fabrique


def fabriques_enregistrees() -> list[Callable[[], DefinitionOutil]]:
    """Retourne une copie des fabriques enregistrées."""
    return list(_FABRIQUES)


def reinitialiser_fabriques() -> None:
    """Vide la liste des fabriques. Utile en test."""
    _FABRIQUES.clear()