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

    `preuves_suffisantes` porte la décision calculée par `noeud_evaluer_preuves`
    (fondée sur les scores de pertinence, voir `nodes.SEUIL_PERTINENCE_MINIMALE`)
    et lue telle quelle par `router_apres_evaluation`, pour que le routage ne
    puisse jamais diverger du jugement déjà journalisé dans la trace.
    """

    session: SessionAgent
    reponse: Any | None = None
    preuves_suffisantes: bool | None = None


__all__ = ["EtatGraphe"]
