"""
Démo en ligne de commande de l'agent documentaire complet (LangGraph).

Contrairement à `python -m src.rag.generation`, qui n'exécute que le RAG
« brut » (une seule recherche, une seule génération), ce script passe par le
graphe agentique réel : boucle rechercher -> évaluer (pertinence puis
suffisance) -> répondre | reformuler, avec refus déterministe si le budget
s'épuise sans preuves validées.

Usage :
    python scripts/demo_agent.py "Who is Sasol's Chief Executive Officer?"
    python scripts/demo_agent.py "What is the boiling point of water?" --verbose
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.graph import construire_graphe
from src.agent.graph_state import EtatGraphe
from src.agent.session import construire_session
from src.rag.vectorstore import fermer_client


def demo(question: str, *, verbose: bool = False) -> None:
    graphe = construire_graphe()
    debut = time.perf_counter()

    session = construire_session(question)
    limite = max(25, session.etat.max_tentatives * 3 + 5)
    resultat = graphe.invoke(EtatGraphe(session=session), config={"recursion_limit": limite})

    duree = time.perf_counter() - debut
    session = resultat["session"]
    reponse = resultat["reponse"]

    print("\n" + "=" * 78)
    print(f"QUESTION : {question}")
    print("=" * 78)

    if verbose:
        print("\nDéroulé de l'agent :")
        for etape in session.etat.trace:
            if etape.nom == "evaluation_pertinence":
                d = etape.donnees
                print(
                    f"  [{etape.nom}] score={d['score_pertinence_maximal']} "
                    f"(seuil {d['seuil_pertinence']}) -> pertinent={d['pertinent']}"
                )
            elif etape.nom == "evaluation_suffisance":
                d = etape.donnees
                print(f"  [{etape.nom}] suffisant={d['suffisant']} — {d['raison']}")
            elif etape.nom in ("reformulation", "reformulation_echec"):
                print(f"  [{etape.nom}] {etape.donnees.get('requete_courante', etape.message)}")
        print()

    print(f"Recherches effectuées : {session.etat.tentatives}")
    print(f"Reformulations        : {sum(1 for e in session.etat.trace if e.nom == 'reformulation')}")
    print()
    print(reponse.reponse)
    print()

    if reponse.sources:
        print("Sources :")
        for source in reponse.sources:
            page = f", p.{source.page}" if source.page is not None else ""
            print(f"  [{source.citation}] {source.document or source.nom_fichier}{page}")
    else:
        print("Sources : aucune (refus — pas de réponse sans preuve documentaire).")

    print(f"\nTemps de réponse : {duree:.1f}s")
    print("=" * 78 + "\n")


def main() -> None:
    parseur = argparse.ArgumentParser(description="Démo de l'agent documentaire.")
    parseur.add_argument("question")
    parseur.add_argument(
        "--verbose", action="store_true", help="affiche le raisonnement pas à pas de l'agent"
    )
    args = parseur.parse_args()

    try:
        demo(args.question, verbose=args.verbose)
    finally:
        fermer_client()


if __name__ == "__main__":
    main()
