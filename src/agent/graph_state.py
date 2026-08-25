"""
État porté par le graphe agentique.

Séparé de `graph.py` pour que `nodes.py` puisse l'importer normalement (pas
seulement sous `TYPE_CHECKING`) sans créer de cycle d'import : LangGraph
résout les annotations de type des nœuds via `typing.get_type_hints`, qui
échoue si `EtatGraphe` n'est visible qu'en import différé.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agent.session import SessionAgent


@dataclass
class EtatGraphe:
    """
    État porté par le graphe.

    `session` n'est jamais réassigné par un nœud : seuls ses attributs
    internes (`etat`, `contexte`) sont mutés via leurs méthodes existantes.
    C'est ce qui permet à LangGraph de conserver ces mutations d'un nœud à
    l'autre malgré la reconstruction de `EtatGraphe` à chaque étape.

    `reponse` reste `None` tant que `generer_reponse` n'a pas été atteint —
    c'est-à-dire tant que la boucle rechercher/reformuler continue.

    Deux jugements distincts, calculés par `noeud_evaluer_preuves` et lus
    tels quels par `router_apres_evaluation` (le routage ne doit jamais
    diverger du jugement déjà journalisé dans la trace) :

        preuves_pertinentes  Niveau 1 — le meilleur passage récupéré
            dépasse-t-il `nodes.SEUIL_PERTINENCE_MINIMALE` (score de
            reranking) ? Déterministe, sans appel LLM.

        preuves_suffisantes  Niveau 2 — ces passages, déjà jugés
            pertinents, contiennent-ils réellement l'information demandée ?
            Jugement LLM borné (voir `nodes._juger_suffisance`). Reste à
            `None` tant que le niveau 1 n'est pas franchi : un passage non
            pertinent n'est jamais soumis à ce jugement.

    Une réponse n'est générée par le LLM que si les deux valent `True`.
    `raison_insuffisance` porte le motif du niveau 2 lorsqu'il échoue, pour
    la reformulation et pour le message de refus. `stagnation` signale que
    la reformulation n'a pas fait bouger le score de pertinence d'une
    tentative à l'autre : l'agent peut alors s'arrêter avant
    `max_tentatives` plutôt que de consommer tout le budget pour rien.

    `intention` (Action 03B) — calculée par `noeud_detecter_intention`, lue
    telle quelle par `router_intention`, jamais recalculée dans le routeur
    (même discipline que `preuves_pertinentes`/`preuves_suffisantes`
    ci-dessus). Vaut ``"search"`` ou ``"summarize"``. `reponse` porte alors
    soit un `ReponseRAG` (chemin SEARCH, inchangé), soit un `ResultatOutil`
    (chemin SUMMARIZE) — voir `src.agent.graph.invoquer_agent`.
    """

    session: SessionAgent
    reponse: Any | None = None
    preuves_pertinentes: bool | None = None
    preuves_suffisantes: bool | None = None
    raison_insuffisance: str | None = None
    stagnation: bool = False
    intention: str | None = None


__all__ = ["EtatGraphe"]
