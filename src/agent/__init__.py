"""
Couche agentique de l'agent documentaire.

Ce paquet n'ajoute aucune capacité documentaire : il réutilise celles qui
existent déjà. Le RAG (`src/rag/`), les outils (`src/tools/`) et le profil
de domaine (`src/profiling/`) restent les composants de référence et ne
sont jamais réimplémentés ici.

Deux flux distincts traversent la couche agentique et ne doivent jamais
être confondus :

    DomainProfile      ->  contexte métier et lexical
    Corpus documentaire ->  preuves factuelles et citations

Usage :

    from src.agent import construire_session

    session = construire_session("Quelles sont les conditions de résiliation ?")
    session.noms_outils()
    resultat = session.executer_outil("search", requete=session.etat.requete_courante)

Ou, pour la boucle complète recherche/reformulation/réponse déjà orchestrée :

    from src.agent import invoquer_agent

    reponse = invoquer_agent("Quelles sont les conditions de résiliation ?")
"""

from src.agent.graph import (
    AgentResponse,
    EtatGraphe,
    construire_graphe,
    executer_agent,
    invoquer_agent,
)
from src.agent.session import (
    ErreurSession,
    SessionAgent,
    construire_registre,
    construire_session,
)
from src.agent.state import (
    BudgetTentativesEpuise,
    ErreurEtatAgent,
    EtapeTrace,
    EtatAgent,
)

__all__ = [
    "construire_session",
    "construire_registre",
    "SessionAgent",
    "ErreurSession",
    "EtatAgent",
    "EtapeTrace",
    "ErreurEtatAgent",
    "BudgetTentativesEpuise",
    "invoquer_agent",
    "executer_agent",
    "AgentResponse",
    "construire_graphe",
    "EtatGraphe",
]