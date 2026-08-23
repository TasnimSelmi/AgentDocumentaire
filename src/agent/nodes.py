"""
Nœuds du graphe agentique.

Chaque nœud est une fonction pure du point de vue du graphe : elle reçoit
l'état courant (`EtatGraphe`), agit sur les objets existants qu'il porte
(`SessionAgent`, `EtatAgent`, `ContexteOutil`) via leurs propres méthodes,
puis renvoie explicitement les champs modifiés. Aucun nœud ne réimplémente
le RAG : `rechercher` passe par l'outil `search` (donc par
`src.rag.retrieval`), `generer_reponse` délègue entièrement à
`src.rag.generation.generer_depuis_recherche`, à partir du même
`RapportRecherche` que celui déjà évalué — jamais d'une seconde recherche
indépendante (voir `noeud_generer_reponse`).

Aucun nœud ne route lui-même vers le nœud suivant : le routage
conditionnel (`router_apres_evaluation`) est séparé des nœuds pour rester
lisible et testable indépendamment de l'exécution du graphe.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from src.agent.graph_state import EtatGraphe
from src.agent.session import SessionAgent
from src.config import get_profil
from src.llm.common import extraire_json_objet, invoquer_llm
from src.rag.generation import ReponseRAG, generer_depuis_recherche, refuser_sans_generation

logger = logging.getLogger(__name__)

# Score utilisé : `Passage.score_final` (src/rag/retrieval.py), lu
# directement sur le dernier `RapportRecherche` (voir
# `ContexteOutil.dernier_rapport_recherche`). Avec le reranker actif
# (comportement par défaut de l'outil `search` — `_executer_search` n'y touche
# pas), ce score est la sortie sigmoïde de BAAI/bge-reranker-v2-m3
# (`compute_score(..., normalize=True)`, voir src/rag/embeddings.py), donc
# dans [0, 1]. Sans reranker (`RERANKER_ENABLED=false`), `score_final` retombe
# sur le score de fusion RRF brut, qui n'est PAS une probabilité comparable à
# ce seuil (le même avertissement existe déjà dans retrieval.py pour
# `score_min`) : ce seuil suppose donc le reranker actif, la configuration
# par défaut du projet.
#
# Calibrage empirique (2026-08-23, corpus ESG/finance actuellement indexé —
# 7 rapports de durabilité réels, data/documents/financial_reports/ — sur les
# 54 questions de evaluation/data/finance_esg.jsonl, script de calibration
# ad hoc appelant rechercher_passages() sur chaque question et relevant le
# score de reranking maximal) :
#   - score maximal observé sur les 4 requêtes hors-sujet (aucun rapport avec
#     le corpus, ex. « Who won the FIFA World Cup in 2018? ») : 0.0099 à 0.0876
#   - score maximal observé sur les 46 requêtes répondables (single-document
#     answerable/document-scoped/cross-document/ambiguous confondues) :
#     0.3054 à 0.9991
# L'ancien seuil (0.08, calibré sur le corpus NLP papers alors indexé) aurait
# laissé passer la requête hors-sujet la plus haute de ce nouveau corpus
# (0.0876 > 0.08) : il n'était pas transférable tel quel. 0.15 est retenu ici
# — marge large au-dessus du hors-sujet (0.0876) et bien en dessous de la
# requête répondable la plus faible (0.3054). Ce calibrage reste valable
# après l'introduction du niveau 2 (suffisance) ci-dessous : la façon dont ce
# score est calculé (score_final du meilleur passage d'un RapportRecherche
# frais) n'a pas changé, seule sa source dans le code a changé (dernier
# rapport plutôt que sources cumulées, voir noeud_evaluer_preuves).
#
# Limite mesurée, non résolue par ce seuil : les questions hors-corpus mais
# THÉMATIQUEMENT proches (ex. "According to Steinhoff's 2022 ESG report, what
# was its B-BBEE level ?" — entreprise réelle, mais aucun rapport Steinhoff
# n'est indexé) obtiennent des scores bien au-dessus de ce seuil (0.35 à 0.73
# sur les 4 cas mesurés) : un score de reranking élevé signale une similarité
# sémantique de passage, pas la présence réelle du fait demandé. C'est
# précisément la limite que le niveau 2 (`_juger_suffisance`) est chargé de
# rattraper : ce seuil reste responsable du niveau 1 (pertinence) seulement.
SEUIL_PERTINENCE_MINIMALE = 0.15

# En dessous de cet écart, deux scores de pertinence maximaux successifs sont
# considérés comme identiques (garde-fou anti-stagnation, section 10).
EPSILON_STAGNATION = 0.01

_SYSTEME_REFORMULATION = """Tu reformules une requête de recherche documentaire.

La recherche précédente n'a pas permis de retrouver l'information nécessaire
pour répondre à la question.

Ta tâche : proposer une nouvelle formulation qui a de meilleures chances de
retrouver cette information dans le corpus, en gardant exactement la même
intention.

Règles strictes :
- Ne change jamais le sujet de la question.
- N'invente aucun fait, aucune entité, aucune date qui ne soit pas déjà
  présente dans la requête initiale.
- Si le motif indique que des passages liés au sujet ont été trouvés mais
  sans la valeur exacte demandée, essaie une formulation plus précise ou
  plus littérale (vocabulaire probable du document, termes techniques),
  plutôt qu'un synonyme plus vague.
- Sinon (aucun passage pertinent trouvé), préfère des synonymes, une
  formulation plus littérale ou plus générale.
- Réponds uniquement avec un objet JSON de la forme :
  {"requete_reformulee": "..."}
"""

# ===========================================================================
# Niveau 2 : suffisance factuelle des preuves déjà jugées pertinentes
# ===========================================================================

_SYSTEME_SUFFISANCE = """Tu juges si des passages documentaires suffisent à répondre à une question.

Règles strictes :
- Utilise UNIQUEMENT les passages fournis ci-dessous. Aucune connaissance
  externe, même si elle te paraît certaine.
- "suffisant" vaut vrai si les passages, PRIS ENSEMBLE, contiennent
  l'information PRÉCISE demandée (la ou les valeurs exactes, le fait exact,
  le nom exact). Une question qui compare deux éléments (deux entreprises,
  deux années, deux documents) peut être suffisante même si aucun passage
  seul ne donne les deux valeurs : il suffit que la combinaison des passages
  fournis les donne toutes.
- "suffisant" vaut faux si un ou plusieurs éléments nécessaires à la réponse
  manquent complètement, ou si les passages parlent seulement du même sujet,
  de la même entreprise ou de la même thématique sans donner l'information
  précise demandée.
- "elements_support" liste les identifiants de citation (ex. "S1") des
  passages qui, ensemble, permettent réellement de répondre. Liste vide si
  "suffisant" vaut faux.
- "raison" est une phrase courte expliquant le verdict.
- Réponds uniquement avec un objet JSON strict de la forme :
  {"suffisant": true|false, "raison": "...", "elements_support": ["S1", ...]}
"""


@dataclass
class VerdictSuffisance:
    """Résultat du jugement de suffisance factuelle (niveau 2)."""

    suffisant: bool
    raison: str
    elements_support: list[str] = field(default_factory=list)
    origine: str = "llm"  # "llm" | "repli_technique"


def _prompt_suffisance(question: str, passages: list) -> str:
    blocs = [
        f"[{p.citation}] ({p.nom_fichier or p.source or 'source inconnue'}, "
        f"page {p.page if p.page is not None else 'inconnue'})\n{p.texte.strip()}"
        for p in passages
    ]
    return f"QUESTION\n{question}\n\nPASSAGES RÉCUPÉRÉS\n" + "\n\n---\n\n".join(blocs)


def _juger_suffisance(llm, question: str, passages: list) -> VerdictSuffisance:
    """
    Jugement LLM borné : une seule évaluation, jamais de boucle interne.

    Ne reçoit ni le profil de domaine (jamais une preuve) ni aucune
    connaissance externe : uniquement la question et les passages déjà
    récupérés. Tout échec (LLM indisponible, JSON invalide, structure
    inattendue) retombe sur un verdict `suffisant=False` : ce jugement ne
    doit jamais conclure « suffisant » par défaut.
    """
    try:
        texte = invoquer_llm(
            llm,
            systeme=_SYSTEME_SUFFISANCE,
            utilisateur=_prompt_suffisance(question, passages),
        )
        objet = extraire_json_objet(texte)

        suffisant = objet.get("suffisant")
        if not isinstance(suffisant, bool):
            raise ValueError(
                f"Champ 'suffisant' manquant ou non booléen : {objet!r}"
            )

        raison = str(objet.get("raison", "")).strip()

        elements = objet.get("elements_support", [])
        if not isinstance(elements, list):
            elements = []
        elements = [str(e).strip() for e in elements if str(e).strip()]

        return VerdictSuffisance(
            suffisant=suffisant,
            raison=raison,
            elements_support=elements,
            origine="llm",
        )

    except Exception as exc:  # noqa: BLE001 — jugement borné, repli conservateur
        logger.warning("Jugement de suffisance impossible : %s", exc)
        return VerdictSuffisance(
            suffisant=False,
            raison=(
                f"Jugement de suffisance indisponible ({type(exc).__name__} : "
                f"{exc}) — repli conservateur, preuves considérées comme non "
                "suffisantes."
            ),
            elements_support=[],
            origine="repli_technique",
        )


def _stagnation(session: SessionAgent, score_maximal: float) -> bool:
    """
    Vrai si le score de pertinence n'a pas bougé depuis la tentative
    précédente (garde-fou déterministe, section 10 — n'examine que le
    niveau 1 : le niveau 2 n'a pas de signal numérique simple à comparer).
    """
    scores_precedents = [
        etape.donnees.get("score_pertinence_maximal")
        for etape in session.etat.trace
        if etape.nom == "evaluation_pertinence"
    ]
    if not scores_precedents:
        return False
    dernier = scores_precedents[-1]
    return dernier is not None and abs(score_maximal - dernier) < EPSILON_STAGNATION


# ===========================================================================
# Nœuds
# ===========================================================================


def noeud_rechercher(etat: EtatGraphe) -> dict:
    """Consomme une tentative et exécute l'outil `search`."""
    session = etat.session

    session.etat.incrementer_tentative()
    session.executer_outil("search", requete=session.etat.requete_courante)

    return {"session": session}


def noeud_evaluer_preuves(etat: EtatGraphe) -> dict:
    """
    Décide si les preuves du dernier rapport de recherche justifient une
    réponse, en deux niveaux distincts.

    Niveau 1 — pertinence (déterministe, sans LLM) : le meilleur passage du
    dernier `RapportRecherche` (`ContexteOutil.dernier_rapport_recherche`,
    écrit par `src.tools.search`) dépasse-t-il `SEUIL_PERTINENCE_MINIMALE` ?
    Lu sur le dernier rapport plutôt que sur les sources cumulées de la
    session : c'est exactement l'ensemble de preuves qui sera transmis à la
    génération si le niveau 2 passe aussi, donc le jugement porte sur les
    mêmes preuves que celles réellement utilisées (voir `noeud_generer_reponse`).

    Niveau 2 — suffisance factuelle (jugement LLM borné, une seule
    évaluation) : n'est calculé QUE si le niveau 1 passe. Un passage jugé non
    pertinent n'est jamais soumis à ce jugement, pour ne pas payer un appel
    LLM sur des cas déjà tranchés.
    """
    session = etat.session
    rapport = session.contexte.dernier_rapport_recherche
    passages = rapport.passages if rapport is not None else []

    score_maximal = max((p.score_final for p in passages), default=0.0)
    pertinent = score_maximal >= SEUIL_PERTINENCE_MINIMALE
    stagnation = (not pertinent) and _stagnation(session, score_maximal)

    session.etat.ajouter_trace(
        "evaluation_pertinence",
        "Retrieval pertinent." if pertinent else "Retrieval non pertinent.",
        nombre_preuves=len(passages),
        score_pertinence_maximal=round(score_maximal, 4),
        seuil_pertinence=SEUIL_PERTINENCE_MINIMALE,
        tentatives=session.etat.tentatives,
        pertinent=pertinent,
        stagnation=stagnation,
    )

    if not pertinent:
        return {
            "session": session,
            "preuves_pertinentes": False,
            "preuves_suffisantes": None,
            "raison_insuffisance": None,
            "stagnation": stagnation,
        }

    verdict = _juger_suffisance(
        session.llm, session.etat.requete_courante, passages
    )

    session.etat.ajouter_trace(
        "evaluation_suffisance",
        (
            "Preuves suffisantes."
            if verdict.suffisant
            else "Preuves pertinentes mais insuffisantes."
        ),
        suffisant=verdict.suffisant,
        raison=verdict.raison,
        elements_support=verdict.elements_support,
        origine=verdict.origine,
    )

    return {
        "session": session,
        "preuves_pertinentes": True,
        "preuves_suffisantes": verdict.suffisant,
        "raison_insuffisance": None if verdict.suffisant else verdict.raison,
        "stagnation": False,
    }


def router_apres_evaluation(etat: EtatGraphe) -> str:
    """
    Arête conditionnelle après `evaluer_preuves`.

    Lit `etat.preuves_pertinentes` / `etat.preuves_suffisantes`, déjà
    calculés et journalisés par `noeud_evaluer_preuves`, plutôt que de
    recalculer un jugement : le routage ne doit jamais pouvoir diverger de ce
    que dit la trace.

    Une réponse n'est tentée que si les deux niveaux valident les preuves.
    Sinon, l'agent reformule tant qu'il reste du budget — sauf en cas de
    stagnation détectée (score de pertinence inchangé depuis la tentative
    précédente), qui interrompt la boucle avant `max_tentatives`. Dans tous
    les cas d'arrêt sans preuves validées, `generer_reponse` produit un refus
    déterministe : voir `noeud_generer_reponse`, qui n'appelle le LLM de
    génération QUE si les deux niveaux sont vrais.
    """
    session = etat.session
    preuves_validees = bool(etat.preuves_pertinentes) and bool(etat.preuves_suffisantes)

    if preuves_validees or not session.etat.peut_reessayer or etat.stagnation:
        return "generer_reponse"

    return "reformuler"


def noeud_reformuler(etat: EtatGraphe) -> dict:
    """
    Reformule la requête courante via un jugement LLM borné.

    Le motif transmis distingue les deux niveaux : « rien de pertinent » ou
    « pertinent mais valeur précise absente » (avec la raison donnée par le
    niveau 2), afin que la reformulation cherche une meilleure formulation
    et non une réponse déguisée.

    Un échec (LLM indisponible, JSON invalide) n'interrompt pas la boucle :
    la requête reste inchangée et la tentative suivante consommera quand
    même le budget, qui reste la garde-fou anti-boucle infinie.
    """
    session = etat.session

    if etat.preuves_pertinentes and etat.raison_insuffisance:
        motif = (
            "La recherche précédente a retrouvé des passages liés au sujet, "
            "mais ils ne contiennent pas l'information précise demandée.\n"
            f"Raison : {etat.raison_insuffisance}"
        )
    else:
        motif = "La recherche précédente n'a retourné aucun passage suffisamment pertinent."

    utilisateur = (
        f"Requête initiale : {session.etat.requete_initiale}\n"
        f"Dernière formulation sans résultat : {session.etat.requete_courante}\n"
        f"{motif}"
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


def _raison_refus(etat: EtatGraphe, session: SessionAgent, rapport) -> str:
    """Motif du refus déterministe, distinguant budget épuisé et stagnation."""
    if not session.etat.peut_reessayer:
        arret = f"budget de tentatives épuisé ({session.etat.tentatives}/{session.etat.max_tentatives})"
    else:
        arret = "arrêt anticipé : la reformulation n'améliore pas le score de pertinence"

    if not etat.preuves_pertinentes:
        score = round(
            max((p.score_final for p in rapport.passages), default=0.0), 4
        ) if rapport is not None else 0.0
        return (
            f"Recherche interrompue ({arret}) : aucun passage suffisamment "
            f"pertinent n'a été retrouvé (score maximal {score} < seuil "
            f"{SEUIL_PERTINENCE_MINIMALE})."
        )

    return (
        f"Recherche interrompue ({arret}) : les passages récupérés sont "
        "pertinents mais jugés insuffisants pour répondre avec fiabilité. "
        f"{etat.raison_insuffisance or ''}".strip()
    )


def noeud_generer_reponse(etat: EtatGraphe) -> dict:
    """
    Génère la réponse finale, ou refuse, à partir du MÊME rapport de
    recherche que celui déjà évalué par `noeud_evaluer_preuves`.

    Ne relance plus jamais de recherche indépendante : l'ancienne version de
    ce nœud appelait `src.rag.generation.generer_reponse`, qui reconstruit son
    propre `RapportRecherche` depuis zéro — un second retrieval capable de
    produire un contexte structurellement valide (`valider_contexte` ne
    connaît aucun seuil de score) même quand le verdict de l'agent avait
    jugé les preuves non pertinentes ou non suffisantes, contournant ainsi ce
    verdict. Le nœud délègue maintenant à `generer_depuis_recherche`, qui
    prend un `RapportRecherche` déjà construit et ne relance rien.

    Si les preuves n'ont pas été validées (niveau 1 ou niveau 2 à False), le
    LLM de génération n'est PAS appelé : `refuser_sans_generation` construit
    directement un refus déterministe, cohérent avec le contrat `ReponseRAG`
    et réutilisant le même message que le refus RAG non-agentique.
    """
    session = etat.session
    debut = time.perf_counter()

    rapport = session.contexte.dernier_rapport_recherche
    preuves_validees = bool(etat.preuves_pertinentes) and bool(etat.preuves_suffisantes)

    if not preuves_validees:
        resultat = _refuser(session, etat, rapport, debut)
    elif rapport is not None:
        resultat = generer_depuis_recherche(
            question=session.etat.requete_courante,
            recherche=rapport,
            profil=get_profil(),
            profil_domaine=session.contexte.profil_domaine,
            llm=session.llm,
        )
    else:
        # Invariant rompu en théorie impossible : preuves_validees=True exige
        # qu'un rapport ait été jugé pertinent ET suffisant par
        # noeud_evaluer_preuves, donc non None à ce stade. Filet de sécurité.
        resultat = _refuser(session, etat, None, debut)

    session.etat.ajouter_trace(
        "reponse",
        "Réponse générée." if preuves_validees else "Refus déterministe (preuves non validées).",
        # `resultat.contexte_suffisant` (ReponseRAG, src.rag.generation) ne
        # vérifie que la validité structurelle du contexte (passages non
        # vides, périmètre respecté — voir src/rag/validation.py::valider_contexte,
        # qui n'applique explicitement aucun seuil de score ni de suffisance
        # factuelle). Ce n'est pas le même signal que preuves_validees
        # ci-dessus (fondé sur le score de reranking ET le jugement de
        # suffisance) : le renommer ici évite qu'il soit lu dans la trace
        # agent comme équivalent. On ne renomme pas le champ `ReponseRAG`
        # lui-même : il est utilisé tel quel par toute la chaîne d'évaluation
        # existante.
        rag_contexte_structurellement_valide=resultat.contexte_suffisant,
        nombre_sources=len(resultat.sources),
        genere_par_llm=preuves_validees,
    )

    return {"session": session, "reponse": resultat}


def _refuser(
    session: SessionAgent,
    etat: EtatGraphe,
    rapport,
    debut: float,
) -> ReponseRAG:
    """
    Refus déterministe : preuves non pertinentes ou non suffisantes, budget
    épuisé (ou stagnation détectée). Aucun appel LLM de génération.
    """
    profil = get_profil()
    profil_domaine = session.contexte.profil_domaine

    if rapport is None:
        # Cas limite : aucune recherche n'a jamais abouti à un RapportRecherche
        # exploitable (ex. corpus indisponible à chaque tentative).
        # `refuser_sans_generation` exige un RapportRecherche pour construire
        # son message (périmètre, motif d'absence) ; il n'y en a aucun ici.
        return ReponseRAG(
            question=session.etat.requete_courante,
            reponse=(
                "Aucune recherche documentaire n'a pu aboutir : le corpus est "
                "resté indisponible pendant toutes les tentatives autorisées."
            ),
            profil=profil.profile_name,
            contexte_suffisant=False,
            citations_valides=True,
            citations_reparees=False,
            profil_domaine=(
                profil_domaine.profile_name if profil_domaine is not None else None
            ),
            recherche=None,
            avertissements=["Corpus indisponible pendant toute la session."],
            duree_secondes=round(time.perf_counter() - debut, 4),
        )

    return refuser_sans_generation(
        session.etat.requete_courante,
        rapport,
        profil,
        [_raison_refus(etat, session, rapport)],
        debut,
        profil_domaine=profil_domaine,
    )


__all__ = [
    "noeud_rechercher",
    "noeud_evaluer_preuves",
    "router_apres_evaluation",
    "noeud_reformuler",
    "noeud_generer_reponse",
    "VerdictSuffisance",
]
