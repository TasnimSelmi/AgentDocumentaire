"""
Graphe agentique : détecter_intention ->
    (SEARCH | SUMMARIZE | CLASSIFY | EXTRACT | COMPARE | SYNTHESIZE).

Branche SEARCH — inchangée depuis le premier graphe (Action « graphe
minimal ») :

    rechercher -> évaluer_preuves -> répondre
                        |
                        +-> reformuler -> rechercher (boucle)

Branche SUMMARIZE (Action 03B) :

    summarize -> répondre

Branche CLASSIFY (Action « classify Option E ») :

    classify -> répondre

Branche EXTRACT (Action 04) :

    extract -> répondre

Branches COMPARE / SYNTHESIZE (P1.5) — activées uniquement quand le signal
multi-document déterministe (`src.agent.multidoc`, P1.4) est explicite
(`is_multidoc` + `operation_hint` ∈ {compare, synthesize}) :

    compare   -> répondre     (MAP borné par document -> REDUCE inter-document)
    synthesize -> répondre

Aucune boucle, aucun planner, aucun ReAct : le routage reste 100 %
déterministe et borné.

Le routage est 100% déterministe, jamais confié au tool-calling du LLM : la
fiabilité du function-calling de Qwen3 8B via Ollama n'est pas établie dans
ce dépôt, et le graphe doit rester prévisible — pour `router_apres_evaluation`
comme pour `router_intention` (vocabulaire fermé + classifieurs LLM bornés
pour les deux seules zones grises CLASSIFY/SEARCH et SEARCH/EXTRACT, voir
`src.agent.nodes`, jamais un choix de tool par le LLM). Le parsing des
champs demandés par EXTRACT (`_parser_champs_extraction`, dans
`src.agent.nodes`) est lui aussi un appel LLM borné, distinct du routage :
il ne choisit ni tool, ni document, ni catégorie.
`SessionAgent.outils_langchain()` reste disponible sans rien casser si un
routage plus autonome est voulu plus tard.

Ce module ne réimplémente ni le RAG, ni les outils, ni `EtatAgent` /
`SessionAgent` : il les consomme. `src/agent/state.py` et
`src/agent/session.py` restent intacts. Les branches SEARCH, SUMMARIZE et
CLASSIFY (nœuds et routeurs) n'ont subi aucune modification de comportement
par l'ajout de EXTRACT.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agent.graph_state import EtatGraphe
from src.agent.nodes import (
    noeud_classify,
    noeud_compare,
    noeud_detecter_intention,
    noeud_evaluer_preuves,
    noeud_extract,
    noeud_generer_reponse,
    noeud_reformuler,
    noeud_rechercher,
    noeud_summarize,
    noeud_synthesize,
    router_apres_evaluation,
    router_intention,
)
from src.agent.session import construire_session


def construire_graphe() -> Any:
    """
    Assemble et compile le graphe. Aucun état partagé entre appels : chaque
    compilation produit un graphe indépendant, sans checkpointer (une
    requête ne survit pas au-delà de son propre `invoke`).
    """
    graphe = StateGraph(EtatGraphe)

    graphe.add_node("detecter_intention", noeud_detecter_intention)
    graphe.add_node("rechercher", noeud_rechercher)
    graphe.add_node("evaluer_preuves", noeud_evaluer_preuves)
    graphe.add_node("reformuler", noeud_reformuler)
    graphe.add_node("generer_reponse", noeud_generer_reponse)
    graphe.add_node("summarize", noeud_summarize)
    graphe.add_node("classify", noeud_classify)
    graphe.add_node("extract", noeud_extract)
    graphe.add_node("compare", noeud_compare)
    graphe.add_node("synthesize", noeud_synthesize)

    graphe.add_edge(START, "detecter_intention")
    graphe.add_conditional_edges(
        "detecter_intention",
        router_intention,
        {
            "rechercher": "rechercher",
            "summarize": "summarize",
            "classify": "classify",
            "extract": "extract",
            "compare": "compare",
            "synthesize": "synthesize",
        },
    )
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
    graphe.add_edge("summarize", END)
    graphe.add_edge("classify", END)
    graphe.add_edge("extract", END)
    graphe.add_edge("compare", END)
    graphe.add_edge("synthesize", END)

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
        Selon l'intention détectée (`EtatGraphe.intention`) :
            - SEARCH : un `ReponseRAG` (`src.rag.generation`), produit par
              `generer_reponse` — comportement inchangé. Le budget de
              tentatives (`EtatAgent.max_tentatives`) garantit que ce nœud
              est toujours atteint : la boucle ne peut pas tourner
              indéfiniment.
            - SUMMARIZE, CLASSIFY ou EXTRACT : un `ResultatOutil`
              (`src.tools.base`), le retour tel quel de l'outil
              correspondant (succès ou échec).
            - COMPARE ou SYNTHESIZE (P1.5) : un `ResultatOutil` également —
              `donnees["comparaison"]` / `donnees["synthese"]` porte la
              structure typée, `sources` la provenance par document. En cas de
              résolution documentaire non fiable, un `ResultatOutil` en échec
              (abstention déterministe), jamais un repli vers SEARCH.
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
