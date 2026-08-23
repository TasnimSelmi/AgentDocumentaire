"""
Nœuds du graphe agentique.

Chaque nœud est une fonction pure du point de vue du graphe : elle reçoit
l'état courant (`EtatGraphe`), agit sur les objets existants qu'il porte
(`SessionAgent`, `EtatAgent`, `ContexteOutil`) via leurs propres méthodes,
puis renvoie explicitement les champs modifiés. Aucun nœud ne réimplémente
le RAG : `rechercher` passe par l'outil `search` (donc par
`src.rag.retrieval`), `generer_reponse` délègue entièrement à
`src.rag.generation.generer_reponse`.

Aucun nœud ne route lui-même vers le nœud suivant : le routage
conditionnel (`router_apres_evaluation`) est séparé des nœuds pour rester
lisible et testable indépendamment de l'exécution du graphe.
"""

from __future__ import annotations

import logging

from src.agent.graph_state import EtatGraphe
from src.config import get_profil
from src.llm.common import extraire_json_objet, invoquer_llm
from src.rag.generation import generer_reponse

logger = logging.getLogger(__name__)

# Score utilisé : `SourceOutil.score` (src/tools/base.py), copié tel quel
# depuis `Passage.score_final` (src/rag/retrieval.py). Avec le reranker actif
# (comportement par défaut de l'outil `search` — `_executer_search` n'y touche
# pas), ce score est la sortie sigmoïde de BAAI/bge-reranker-v2-m3
# (`compute_score(..., normalize=True)`, voir src/rag/embeddings.py), donc
# dans [0, 1]. Sans reranker (`RERANKER_ENABLED=false`), `score_final` retombe
# sur le score de fusion RRF brut, qui n'est PAS une probabilité comparable à
# ce seuil (le même avertissement existe déjà dans retrieval.py pour
# `score_min`) : ce seuil suppose donc le reranker actif, la configuration
# par défaut du projet.
#
# Calibrage empirique (2026-08-22, corpus NLP papers actuellement indexé,
# 9 requêtes manuelles — voir tests/agent/test_nodes.py pour les cas
# couverts par des tests) :
#   - score maximal observé sur des requêtes hors-corpus / non pertinentes
#     (sujets ESG/finance absents du corpus, questions générales sans lien) :
#     0.0052 à 0.0559
#   - score maximal observé sur des requêtes dont l'évidence est réellement
#     présente dans le corpus : 0.1164 (cas faible mais correct) à 0.9058
#     (cas net)
# 0.08 se situe entre les deux, avec de la marge des deux côtés. Échantillon
# réduit : à revoir avec une mesure plus large avant un déploiement en
# production (voir BACKLOG), mais suffisant pour rendre ce nœud sensible à la
# pertinence sans jugement LLM.
SEUIL_PERTINENCE_MINIMALE = 0.08

_SYSTEME_REFORMULATION = """Tu reformules une requête de recherche documentaire.

La recherche précédente n'a retourné aucun passage suffisamment pertinent.
Ta tâche :
proposer une nouvelle formulation qui a de meilleures chances de retrouver
des passages dans le corpus, en gardant exactement la même intention.

Règles strictes :
- Ne change jamais le sujet de la question.
- N'invente aucun fait, aucune entité, aucune date qui ne soit pas déjà
  présente dans la requête initiale.
- Préfère des synonymes, une formulation plus littérale ou plus générale.
- Réponds uniquement avec un objet JSON de la forme :
  {"requete_reformulee": "..."}
"""


def noeud_rechercher(etat: EtatGraphe) -> dict:
    """Consomme une tentative et exécute l'outil `search`."""
    session = etat.session

    session.etat.incrementer_tentative()
    session.executer_outil("search", requete=session.etat.requete_courante)

    return {"session": session}


def noeud_evaluer_preuves(etat: EtatGraphe) -> dict:
    """
    Décide si les preuves accumulées justifient une réponse.

    Sensible à la pertinence, pas seulement à la présence de passages :
    `search` (src/tools/search.py) n'applique jamais de seuil de pertinence
    (`rechercher_passages(appliquer_seuil=False)` par défaut), donc une
    requête hors sujet renvoie quand même `top_k` passages. Compter les
    sources ne distinguerait jamais une évidence faible d'une évidence
    absente. Le jugement retenu est déterministe et sans appel LLM : le
    score de pertinence maximal parmi les sources déjà accumulées, comparé à
    `SEUIL_PERTINENCE_MINIMALE`.
    """
    session = etat.session

    score_maximal = max(
        (source.score for source in session.contexte.sources),
        default=0.0,
    )
    suffisant = score_maximal >= SEUIL_PERTINENCE_MINIMALE

    session.etat.ajouter_trace(
        "evaluation_preuves",
        "Preuves suffisantes." if suffisant else "Preuves insuffisantes.",
        nombre_preuves=session.nombre_preuves,
        score_pertinence_maximal=round(score_maximal, 4),
        seuil_pertinence=SEUIL_PERTINENCE_MINIMALE,
        tentatives=session.etat.tentatives,
        suffisant=suffisant,
    )

    return {"session": session, "preuves_suffisantes": suffisant}


def router_apres_evaluation(etat: EtatGraphe) -> str:
    """
    Arête conditionnelle après `evaluer_preuves`.

    Lit `etat.preuves_suffisantes`, déjà calculé et journalisé par
    `noeud_evaluer_preuves`, plutôt que de recalculer un jugement : le
    routage ne doit jamais pouvoir diverger de ce que dit la trace. Une
    réponse est tentée dès que les preuves sont jugées suffisantes, ou dès
    que le budget de tentatives est épuisé : `generer_reponse` (donc
    `src.rag.generation`) a déjà sa propre logique de refus sourcé quand le
    contexte est insuffisant, elle n'a pas besoin d'être dupliquée ici.
    """
    session = etat.session

    if etat.preuves_suffisantes or not session.etat.peut_reessayer:
        return "generer_reponse"

    return "reformuler"


def noeud_reformuler(etat: EtatGraphe) -> dict:
    """
    Reformule la requête courante via un jugement LLM borné.

    Un échec (LLM indisponible, JSON invalide) n'interrompt pas la boucle :
    la requête reste inchangée et la tentative suivante consommera quand
    même le budget, qui reste la garde-fou anti-boucle infinie.
    """
    session = etat.session

    utilisateur = (
        f"Requête initiale : {session.etat.requete_initiale}\n"
        f"Dernière formulation sans résultat : {session.etat.requete_courante}"
    )

    try:
        texte = invoquer_llm(
            session.llm,
            systeme=_SYSTEME_REFORMULATION,
            utilisateur=utilisateur,
        )
        objet = extraire_json_objet(texte)
        nouvelle_requete = str(objet.get("requete_reformulee", "")).strip()

        if not nouvelle_requete:
            raise ValueError("Reformulation vide retournée par le LLM.")

        session.etat.reformuler(
            nouvelle_requete,
            motif="Reformulation automatique (preuves insuffisantes).",
        )

    except Exception as exc:  # noqa: BLE001 — la boucle ne doit jamais casser
        logger.warning("Reformulation impossible : %s", exc)
        session.etat.ajouter_trace(
            "reformulation_echec",
            f"{type(exc).__name__} : {exc}",
        )

    return {"session": session}


def noeud_generer_reponse(etat: EtatGraphe) -> dict:
    """
    Génère la réponse finale en délégant entièrement à `src.rag.generation`.

    Ce nœud ne réutilise pas les sources déjà accumulées par l'outil
    `search` : `generer_reponse` relance son propre pipeline
    récupération + génération sur la requête courante, ce qui lui laisse la
    responsabilité entière des citations, du périmètre documentaire et du
    refus sourcé — exactement le contrat déjà testé de `src.rag.generation`.
    """
    session = etat.session

    resultat = generer_reponse(
        question=session.etat.requete_courante,
        profil=get_profil(),
        profil_domaine=session.contexte.profil_domaine,
        llm=session.llm,
    )

    session.etat.ajouter_trace(
        "reponse",
        "Réponse générée.",
        # `resultat.contexte_suffisant` (ReponseRAG, src.rag.generation) ne
        # vérifie que la validité structurelle du contexte (passages non
        # vides, périmètre respecté — voir src/rag/validation.py::valider_contexte,
        # qui n'applique explicitement aucun seuil de score). Ce n'est pas un
        # signal de pertinence ni de grounding : le renommer ici évite qu'il
        # soit lu dans la trace agent comme équivalent au jugement de
        # pertinence déjà posé par `evaluation_preuves` (fondé sur les scores
        # du reranking). On ne renomme pas le champ `ReponseRAG` lui-même :
        # il est utilisé tel quel par toute la chaîne d'évaluation existante.
        rag_contexte_structurellement_valide=resultat.contexte_suffisant,
        nombre_sources=len(resultat.sources),
    )

    return {"session": session, "reponse": resultat}


__all__ = [
    "noeud_rechercher",
    "noeud_evaluer_preuves",
    "router_apres_evaluation",
    "noeud_reformuler",
    "noeud_generer_reponse",
]
