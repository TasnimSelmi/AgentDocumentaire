"""
Graphe agentique minimal : rechercher -> évaluer -> (répondre | reformuler).

Premier graphe volontairement réduit à 3 décisions, à valider manuellement
avant d'y ajouter `extract`/`summarize`/`classify` ou un routage plus fin :

    rechercher -> évaluer_preuves -> répondre
                        |
                        +-> reformuler -> rechercher (boucle)

Le routage est 100% déterministe (`router_apres_evaluation`), jamais confié
au tool-calling du LLM : la fiabilité du function-calling de Qwen3 8B via
Ollama n'est pas établie dans ce dépôt, et cette boucle doit rester
prévisible. `SessionAgent.outils_langchain()` reste disponible sans rien
casser si un routage plus autonome est voulu plus tard.

Ce module ne réimplémente ni le RAG, ni les outils, ni `EtatAgent` /
`SessionAgent` : il les consomme. `src/agent/state.py` et
`src/agent/session.py` restent intacts.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agent.graph_state import EtatGraphe
from src.agent.nodes import (
    noeud_evaluer_preuves,
    noeud_generer_reponse,
    noeud_reformuler,
    noeud_rechercher,
    router_apres_evaluation,
)
from src.agent.session import construire_session


def construire_graphe() -> Any:
    """
    Assemble et compile le graphe. Aucun état partagé entre appels : chaque
    compilation produit un graphe indépendant, sans checkpointer (une
    requête ne survit pas au-delà de son propre `invoke`).
    """
    graphe = StateGraph(EtatGraphe)

    graphe.add_node("rechercher", noeud_rechercher)
    graphe.add_node("evaluer_preuves", noeud_evaluer_preuves)
    graphe.add_node("reformuler", noeud_reformuler)
    graphe.add_node("generer_reponse", noeud_generer_reponse)

    graphe.add_edge(START, "rechercher")
    graphe.add_edge("rechercher", "evaluer_preuves")
    graphe.add_conditional_edges(
        "evaluer_preuves",
        router_apres_evaluation,
        {
            "generer_reponse": "generer_reponse",
            "reformuler": "reformuler",
        },
    )
    graphe.add_edge("reformuler", "rechercher")
    graphe.add_edge("generer_reponse", END)

    return graphe.compile()


# Le graphe est sans état propre (aucun checkpointer) : une seule instance
# compilée peut être réutilisée par toutes les requêtes.
_GRAPHE = construire_graphe()


def invoquer_agent(
    requete: str,
    **kwargs_session: Any,
) -> Any:
    """
    Point d'entrée public : construit une session puis exécute le graphe.

    Args:
        requete: demande de l'utilisateur.
        **kwargs_session: transmis tels quels à `construire_session`
            (llm, profil_domaine, max_tentatives, ...).

    Returns:
        Le `ReponseRAG` produit par `generer_reponse`. Le budget de
        tentatives (`EtatAgent.max_tentatives`) garantit que ce nœud est
        toujours atteint : la boucle ne peut pas tourner indéfiniment.
    """
    session = construire_session(requete, **kwargs_session)

    # Le budget de récursion de LangGraph reste un filet de sécurité
    # générique, distinct du budget métier (`EtatAgent.max_tentatives`) :
    # chaque tentative traverse 2 nœuds (rechercher, evaluer_preuves) plus,
    # en cas de reformulation, un 3e (reformuler) — d'où la marge.
    recursion_limit = max(25, session.etat.max_tentatives * 3 + 5)

    resultat = _GRAPHE.invoke(
        EtatGraphe(session=session),
        config={"recursion_limit": recursion_limit},
    )

    return resultat["reponse"]


__all__ = [
    "EtatGraphe",
    "construire_graphe",
    "invoquer_agent",
]
