"""
État de workflow de l'agent documentaire.

Ce module ne fait aucune recherche, n'appelle aucun LLM et ne contient
aucune preuve documentaire. Il décrit uniquement la progression d'une
requête utilisateur à travers le futur graphe agentique.

Séparation volontaire, à ne jamais confondre :

    ContexteOutil  (src/tools/base.py)   ->  PREUVES
        passages retrouvés, résultats d'outils, LLM partagé.
        Composant existant, déjà testé, non modifié.

    EtatAgent      (ce module)           ->  WORKFLOW
        requête courante, budget de tentatives, trace d'exécution.

Recopier les preuves ici reviendrait à réimplémenter ce que
ContexteOutil fait déjà. L'état ne duplique donc rien : la session
(src/agent/session.py) tient les deux objets côte à côte.

Champs volontairement absents à ce stade
----------------------------------------
Aucun champ n'est ajouté « au cas où ». `intention`, `route`, `reponse`
et `demande_action` appartiennent au workflow cible mais rien ne les
alimenterait aujourd'hui : ils arriveront avec le nœud qui les produit.

Le profil de domaine est présent, mais uniquement comme contexte métier.
Il n'est jamais une preuve et ne peut produire aucune citation : aucune
méthode de ce module ne le rapproche des sources documentaires.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class ErreurEtatAgent(RuntimeError):
    """Erreur de manipulation de l'état du workflow."""


class BudgetTentativesEpuise(ErreurEtatAgent):
    """
    Le nombre maximal de tentatives est atteint.

    Levée plutôt que silencieusement ignorée : une boucle qui dépasse son
    budget est un défaut de conception du graphe, pas un cas nominal.
    L'appelant doit interroger `peut_reessayer()` avant de reformuler.
    """


def _maintenant() -> str:
    """Horodatage ISO 8601 en UTC, stable pour les traces et les tests."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class EtapeTrace:
    """
    Une étape franchie par l'agent, conservée pour l'observabilité.

    La trace est le seul moyen de comprendre après coup pourquoi l'agent a
    répondu ce qu'il a répondu : quel outil a été choisi, combien de
    passages ont été trouvés, si une reformulation a eu lieu.
    """

    nom: str
    message: str = ""
    donnees: dict[str, Any] = field(default_factory=dict)
    horodatage: str = field(default_factory=_maintenant)

    def vers_dict(self) -> dict[str, Any]:
        return {
            "nom": self.nom,
            "message": self.message,
            "donnees": dict(self.donnees),
            "horodatage": self.horodatage,
        }

    def __str__(self) -> str:
        return f"{self.nom} — {self.message}" if self.message else self.nom


@dataclass
class EtatAgent:
    """
    État de progression d'une requête utilisateur.

    Attributes:
        requete_initiale: formulation d'origine, jamais modifiée. Elle sert
            de référence pour la réponse finale et permet de mesurer l'écart
            introduit par une reformulation.
        requete_courante: formulation réellement envoyée au retrieval. Elle
            diffère de l'initiale dès la première reformulation.
        profil_technique: nom du profil `config/schemas/<nom>.yaml` actif.
            Conservé parce que les schémas d'arguments des outils en
            dépendent : une trace sans lui serait ininterprétable.
        profil_domaine: `DomainProfile` actif, ou None. Contexte métier et
            lexical uniquement, jamais une source de faits.
        tentatives: nombre de passages de recherche déjà consommés.
        max_tentatives: borne dure de toute boucle. Vient de
            `config/default.yaml` -> `agent.max_iterations`.
        trace: étapes franchies, dans l'ordre.
    """

    requete_initiale: str
    requete_courante: str
    profil_technique: str = ""
    profil_domaine: Any | None = None
    tentatives: int = 0
    max_tentatives: int = 6
    trace: list[EtapeTrace] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.requete_initiale = " ".join(str(self.requete_initiale).split())
        self.requete_courante = " ".join(str(self.requete_courante).split())

        if not self.requete_initiale:
            raise ErreurEtatAgent("La requête initiale est vide.")

        if not self.requete_courante:
            self.requete_courante = self.requete_initiale

        if self.max_tentatives < 1:
            raise ErreurEtatAgent(
                "max_tentatives doit être supérieur ou égal à 1 "
                f"(reçu : {self.max_tentatives})."
            )

        if self.tentatives < 0:
            raise ErreurEtatAgent("tentatives ne peut pas être négatif.")

    # -- Contexte métier ----------------------------------------------------

    @property
    def a_un_profil_domaine(self) -> bool:
        """
        Vrai lorsqu'un contexte métier est disponible.

        L'absence de profil est un cas nominal : le système doit continuer
        de fonctionner exactement comme avant.
        """
        return self.profil_domaine is not None

    @property
    def nom_profil_domaine(self) -> str | None:
        """Nom du profil de domaine actif, pour la journalisation."""
        if self.profil_domaine is None:
            return None
        return getattr(self.profil_domaine, "profile_name", None)

    # -- Budget de boucle ---------------------------------------------------

    @property
    def peut_reessayer(self) -> bool:
        """Vrai tant que le budget de tentatives n'est pas épuisé."""
        return self.tentatives < self.max_tentatives

    @property
    def tentatives_restantes(self) -> int:
        return max(0, self.max_tentatives - self.tentatives)

    @property
    def a_ete_reformulee(self) -> bool:
        """Vrai si la requête courante s'écarte de la requête initiale."""
        return self.requete_courante != self.requete_initiale

    def incrementer_tentative(self) -> int:
        """
        Consomme une tentative de recherche.

        Returns:
            Le nouveau nombre de tentatives consommées.

        Raises:
            BudgetTentativesEpuise: si le budget est déjà atteint. Aucune
                boucle ne peut donc devenir infinie, même en cas d'erreur
                dans une arête conditionnelle.
        """
        if not self.peut_reessayer:
            raise BudgetTentativesEpuise(
                f"Budget de tentatives épuisé ({self.tentatives}/"
                f"{self.max_tentatives}) pour : {self.requete_initiale!r}."
            )

        self.tentatives += 1
        return self.tentatives

    def reformuler(self, nouvelle_requete: str, *, motif: str = "") -> str:
        """
        Remplace la requête courante et journalise le changement.

        La requête initiale reste intacte : c'est elle qui sera présentée à
        l'utilisateur, pas la formulation intermédiaire de l'agent.

        Raises:
            ErreurEtatAgent: si la nouvelle formulation est vide.
        """
        propre = " ".join(str(nouvelle_requete).split())

        if not propre:
            raise ErreurEtatAgent("La requête reformulée est vide.")

        ancienne = self.requete_courante
        self.requete_courante = propre

        self.ajouter_trace(
            "reformulation",
            motif or "Requête reformulée.",
            requete_precedente=ancienne,
            requete_courante=propre,
        )
        return propre

    # -- Observabilité ------------------------------------------------------

    def ajouter_trace(
        self,
        nom: str,
        message: str = "",
        **donnees: Any,
    ) -> EtapeTrace:
        """Enregistre une étape franchie et la retourne."""
        etape = EtapeTrace(nom=nom, message=message, donnees=donnees)
        self.trace.append(etape)
        logger.debug("Étape agent : %s", etape)
        return etape

    def noms_etapes(self) -> list[str]:
        """Séquence des étapes franchies, dans l'ordre."""
        return [etape.nom for etape in self.trace]

    def resume_trace(self) -> str:
        """Rendu lisible de la trace, destiné aux journaux et au débogage."""
        if not self.trace:
            return "Aucune étape enregistrée."

        lignes = [
            f"Requête      : {self.requete_initiale}",
        ]

        if self.a_ete_reformulee:
            lignes.append(f"Reformulée   : {self.requete_courante}")

        lignes.append(f"Profil tech. : {self.profil_technique or 'inconnu'}")
        lignes.append(f"Profil métier: {self.nom_profil_domaine or 'aucun'}")
        lignes.append(
            f"Tentatives   : {self.tentatives}/{self.max_tentatives}"
        )
        lignes.append("Étapes :")

        for index, etape in enumerate(self.trace, start=1):
            lignes.append(f"  {index}. {etape}")

        return "\n".join(lignes)

    def vers_dict(self) -> dict[str, Any]:
        """
        Représentation sérialisable de l'état.

        Le profil de domaine est réduit à son nom : le sérialiser entièrement
        laisserait croire qu'il fait partie des données de la réponse.
        """
        return {
            "requete_initiale": self.requete_initiale,
            "requete_courante": self.requete_courante,
            "profil_technique": self.profil_technique,
            "profil_domaine": self.nom_profil_domaine,
            "tentatives": self.tentatives,
            "max_tentatives": self.max_tentatives,
            "trace": [etape.vers_dict() for etape in self.trace],
        }


__all__ = [
    "EtatAgent",
    "EtapeTrace",
    "ErreurEtatAgent",
    "BudgetTentativesEpuise",
]