"""
Fabrique de session agentique.

Ce module est le chaînon manquant entre les outils existants et le futur
graphe : il assemble en un seul appel tout ce qu'un agent doit avoir sous
la main pour traiter une requête.

    construire_session("Quel est le niveau B-BBEE d'Absa en 2022 ?")
        │
        ├── construire_llm()                  -> ChatOllama (src/llm/factory.py)
        ├── load_active_domain_profile()      -> DomainProfile | None
        ├── ContexteOutil(...)                -> preuves partagées
        ├── construire_registre(...)          -> search/extract/summarize/classify
        └── EtatAgent(...)                    -> workflow + budget + trace

Le module ne réimplémente rien. En particulier :

    - il n'appelle ni Qdrant, ni les embeddings, ni rechercher_passages() :
      ce sont les outils qui encapsulent déjà le RAG ;
    - il ne construit aucun prompt de génération ;
    - il ne prend aucune décision. Le choix d'un outil appartiendra au
      graphe, pas à la session.

Conséquence pratique : une session est instanciable sans serveur Ollama et
sans collection Qdrant. Seule l'exécution effective d'un outil en a besoin.

Outils de lecture et outils d'action
------------------------------------
`DefinitionOutil.lecture_seule` existe déjà et vaut True pour les quatre
outils actuels. La session s'en sert dès maintenant comme point de contrôle :
par défaut, seuls les outils de lecture sont exposés. Le jour où des outils
d'écriture branchés sur les APIs de l'entreprise existeront, il faudra une
autorisation explicite pour qu'un agent puisse seulement les voir.

Aucune API d'entreprise n'est supposée, inventée ou appelée ici.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from src.config import get_config_technique, get_profil
from src.profiling import DomainProfile, load_active_domain_profile
from src.tools.base import (
    ContexteOutil,
    DefinitionOutil,
    RegistreOutils,
    ResultatOutil,
    fabriques_enregistrees,
)

logger = logging.getLogger(__name__)

# Importer le paquet enregistre les fabriques des quatre outils via le
# décorateur @outil. Sans cet import, fabriques_enregistrees() renvoie une
# liste vide : l'import est donc fonctionnel, pas décoratif.
import src.tools as _enregistrement_des_outils  # noqa: E402,F401  (effet de bord voulu)


class ErreurSession(RuntimeError):
    """La session agentique n'a pas pu être construite."""


# ===========================================================================
# 1. Construction du registre
# ===========================================================================


def construire_registre(
    contexte: ContexteOutil | None = None,
    *,
    fabriques: Sequence[Callable[[], DefinitionOutil]] | None = None,
    inclure_outils_ecriture: bool = False,
) -> RegistreOutils:
    """
    Construit un registre à partir des fabriques déclarées par @outil.

    Les fabriques sont appelées maintenant, et non figées à l'import : les
    schémas d'arguments dépendent du profil technique actif, qui peut avoir
    changé entre-temps. C'est exactement le contrat prévu par
    `src/tools/base.py`.

    Args:
        contexte: contexte partagé associé au registre. Tous les outils
            exécutés via ce registre l'utiliseront automatiquement.
        fabriques: liste de fabriques à utiliser. Par défaut, celles
            enregistrées globalement. Le paramètre existe pour les tests :
            il évite d'avoir à manipuler l'état global.
        inclure_outils_ecriture: expose aussi les outils qui modifient un
            système externe (`lecture_seule=False`). Faux par défaut.

    Returns:
        Un `RegistreOutils` prêt à l'emploi.

    Raises:
        ErreurSession: si une fabrique échoue, ou si le registre est vide.
            Un profil YAML invalide doit se voir immédiatement, pas
            produire un agent silencieusement dépourvu d'outils.
    """
    registre = RegistreOutils(contexte=contexte)

    a_utiliser = (
        list(fabriques)
        if fabriques is not None
        else fabriques_enregistrees()
    )

    if not a_utiliser:
        raise ErreurSession(
            "Aucune fabrique d'outil n'est enregistrée. Vérifie que "
            "src.tools est bien importé."
        )

    ignores: list[str] = []

    for fabrique in a_utiliser:
        try:
            definition = fabrique()
        except Exception as exc:  # noqa: BLE001 — remontée explicite
            raise ErreurSession(
                f"La fabrique {getattr(fabrique, '__name__', fabrique)!r} "
                f"a échoué : {type(exc).__name__} — {exc}"
            ) from exc

        if not definition.lecture_seule and not inclure_outils_ecriture:
            ignores.append(definition.nom)
            continue

        registre.enregistrer(definition)

    if ignores:
        logger.info(
            "Outils d'action non exposés (autorisation requise) : %s.",
            ", ".join(sorted(ignores)),
        )

    if len(registre) == 0:
        raise ErreurSession(
            "Le registre ne contient aucun outil exploitable."
        )

    return registre


# ===========================================================================
# 2. Session
# ===========================================================================


@dataclass
class SessionAgent:
    """
    Tout ce dont l'agent dispose pour traiter une requête utilisateur.

    Les deux objets centraux restent distincts et complémentaires :

        contexte -> les PREUVES  (ContexteOutil, existant)
        etat     -> le WORKFLOW  (EtatAgent, nouveau)

    Une instance correspond à une requête. Deux requêtes successives ne
    partagent ni sources ni trace.
    """

    etat: Any  # EtatAgent — annoté en Any pour éviter un import circulaire
    contexte: ContexteOutil
    registre: RegistreOutils
    llm: Any | None = None

    # -- Accès aux outils ---------------------------------------------------

    def noms_outils(self) -> list[str]:
        """Noms des outils actuellement disponibles."""
        return self.registre.noms()

    def outils_langchain(self) -> list[Any]:
        """
        Outils au format LangChain, prêts pour `bind_tools`.

        Le graphe de l'étape suivante consommera cette méthode telle quelle.
        """
        return self.registre.vers_langchain()

    def bloc_outils(self) -> str:
        """Description des outils, pour le prompt système de l'agent."""
        return self.registre.bloc_prompt()

    def bloc_domaine(self) -> str:
        """
        Contexte métier issu du profil de domaine, ou chaîne vide.

        Réutilise `ContexteOutil.bloc_domaine()`, qui tolère déjà l'absence
        de profil et les objets ne sachant pas le rendre.
        """
        return self.contexte.bloc_domaine()

    # -- Exécution ----------------------------------------------------------

    def executer_outil(self, nom: str, **arguments: Any) -> ResultatOutil:
        """
        Exécute un outil et consigne le passage dans la trace.

        Un échec d'outil n'est jamais une exception : `DefinitionOutil`
        renvoie un `ResultatOutil` en échec, que l'agent pourra lire pour
        corriger son appel suivant. La session préserve ce contrat.
        """
        resultat = self.registre.executer(nom, **arguments)

        self.etat.ajouter_trace(
            "outil",
            f"{nom} : {'succès' if resultat.succes else 'échec'}",
            outil=nom,
            succes=resultat.succes,
            duree_secondes=resultat.duree_secondes,
            sources_retournees=len(resultat.sources),
            sources_cumulees=len(self.contexte.sources),
        )
        return resultat

    # -- État des preuves ---------------------------------------------------

    @property
    def a_des_preuves(self) -> bool:
        """Vrai si au moins un passage documentaire a été récupéré."""
        return self.contexte.a_des_sources

    @property
    def nombre_preuves(self) -> int:
        return len(self.contexte.sources)

    def outils_utilises(self) -> list[str]:
        """Outils réellement exécutés pendant cette session, dans l'ordre."""
        return [resultat.outil for resultat in self.contexte.resultats]

    def resume(self) -> str:
        """Vue lisible de la session, trace comprise."""
        return (
            f"{self.etat.resume_trace()}\n"
            f"Outils dispo.: {', '.join(self.noms_outils()) or 'aucun'}\n"
            f"Outils exéc. : {', '.join(self.outils_utilises()) or 'aucun'}\n"
            f"Preuves      : {self.nombre_preuves}"
        )


# ===========================================================================
# 3. Fabrique publique
# ===========================================================================


def construire_session(
    requete: str,
    *,
    llm: Any | None = None,
    profil_domaine: DomainProfile | None = None,
    charger_profil_domaine: bool = True,
    max_tentatives: int | None = None,
    fabriques: Sequence[Callable[[], DefinitionOutil]] | None = None,
    inclure_outils_ecriture: bool = False,
) -> SessionAgent:
    """
    Assemble une session agentique complète pour une requête.

    Args:
        requete: demande de l'utilisateur.
        llm: client LLM à partager entre les outils. Par défaut, celui du
            projet (`src.llm.factory.construire_llm`). Injectable pour les
            tests, comme le fait déjà `suggest_domain_profile`.
        profil_domaine: profil imposé. S'il est fourni, aucun chargement
            n'a lieu.
        charger_profil_domaine: charge le profil actif défini par
            `ACTIVE_DOMAIN_PROFILE`. Mettre à False permet de travailler
            explicitement sans contexte métier.
        max_tentatives: borne des boucles. Par défaut,
            `config/default.yaml` -> `agent.max_iterations`.
        fabriques: fabriques d'outils à utiliser (tests).
        inclure_outils_ecriture: expose les outils d'action.

    Returns:
        Une `SessionAgent` prête à l'emploi.

    Raises:
        ErreurSession: requête vide, ou registre inconstructible.
        DomainProfileNotFoundError: si `ACTIVE_DOMAIN_PROFILE` désigne un
            profil absent. Comportement existant, volontairement conservé :
            une configuration erronée doit être visible.
    """
    from src.agent.state import EtatAgent  # import local : évite un cycle

    question = " ".join(str(requete).split())

    if not question:
        raise ErreurSession("La requête de la session est vide.")

    # --- Contexte métier ---------------------------------------------------
    # Le profil de domaine apporte du vocabulaire, jamais des faits.
    if profil_domaine is None and charger_profil_domaine:
        profil_domaine = load_active_domain_profile()

    if profil_domaine is not None:
        logger.info(
            "Session : profil de domaine « %s » (%s).",
            profil_domaine.profile_name,
            profil_domaine.domain,
        )
    else:
        logger.info("Session : aucun profil de domaine, contexte métier absent.")

    # --- LLM partagé -------------------------------------------------------
    # Les outils ne construisent jamais leur propre client : le même LLM
    # circule par ContexteOutil.
    if llm is None:
        from src.llm.factory import construire_llm

        llm = construire_llm()

    # --- Preuves -----------------------------------------------------------
    contexte = ContexteOutil(
        question=question,
        llm=llm,
        profil_domaine=profil_domaine,
    )

    # --- Outils ------------------------------------------------------------
    registre = construire_registre(
        contexte=contexte,
        fabriques=fabriques,
        inclure_outils_ecriture=inclure_outils_ecriture,
    )

    # --- Workflow ----------------------------------------------------------
    if max_tentatives is None:
        max_tentatives = get_config_technique().agent.max_iterations

    etat = EtatAgent(
        requete_initiale=question,
        requete_courante=question,
        profil_technique=get_profil().profile_name,
        profil_domaine=profil_domaine,
        max_tentatives=max_tentatives,
    )

    etat.ajouter_trace(
        "session",
        "Session agentique construite.",
        outils=registre.noms(),
        profil_technique=etat.profil_technique,
        profil_domaine=etat.nom_profil_domaine,
        max_tentatives=max_tentatives,
    )

    logger.info(
        "Session prête : %d outil(s) [%s], budget %d tentative(s).",
        len(registre),
        ", ".join(registre.noms()),
        max_tentatives,
    )

    return SessionAgent(
        etat=etat,
        contexte=contexte,
        registre=registre,
        llm=llm,
    )


__all__ = [
    "SessionAgent",
    "construire_session",
    "construire_registre",
    "ErreurSession",
]