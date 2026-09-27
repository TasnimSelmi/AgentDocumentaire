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
import re
import time
import unicodedata
from dataclasses import dataclass, field

from src.agent.graph_state import EtatGraphe
from src.agent.multidoc import detecter_multidoc
from src.agent.session import SessionAgent
from src.config import get_profil
from src.llm.common import extraire_json_objet, invoquer_llm
from src.rag.generation import ReponseRAG, generer_depuis_recherche, refuser_sans_generation
from src.rag.retrieval import resoudre_document
from src.tools.base import ResultatOutil
from src.tools.compare import comparer
from src.tools.synthesize import synthetiser_documents

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


# ===========================================================================
# Détection d'intention (Action 03B) : SEARCH vs SUMMARIZE
# ===========================================================================
#
# Vocabulaire fermé, générique et bilingue (FR/EN) : signale une INTENTION de
# résumé documentaire, jamais un domaine métier ni un corpus particulier.
# Choisi plutôt qu'un classifieur LLM parce que la distinction SEARCH vs
# SUMMARIZE ne demande aucune nuance sémantique (contrairement au jugement de
# suffisance, niveau 2 de la boucle QA, qui en a réellement besoin) : un
# répertoire de verbes suffit, et reste déterministe, sans latence LLM, sans
# mode d'échec réseau — cohérent avec le niveau 1 (pertinence) de la boucle
# QA existante, également déterministe. Par défaut (aucun déclencheur
# trouvé), SEARCH : repli sûr vers le chemin existant et déjà validé.
_MOTIF_MOTS_INTENTION = re.compile(r"[a-z0-9]+")

_JETONS_SUMMARIZE = {
    # français — accents déjà retirés par _normaliser_intention
    "resume", "resumer", "resumes", "resumee", "resumees",
    "synthese", "syntheses", "synthetise", "synthetiser",
    "sommaire",
    # anglais
    "summarize", "summarise", "summary", "summaries",
    "summarizing", "summarising", "tldr",
}
_BIGRAMMES_SUMMARIZE = ("points cles", "key points", "en bref", "in short")

# CLASSIFY (cette action) : uniquement des formes VERBALES d'action, jamais
# le nom "catégorie"/"classification" seul — voir _JETONS_CLASSIFY_AMBIGUS
# ci-dessous pour la raison. "classe" est ambigu (nom commun ou impératif de
# "classer") : n'est retenu que si c'est le premier mot de la requête,
# approximation simple d'un impératif ("Classe ce document." vs "Quelle
# classe de risque ?").
_JETONS_CLASSIFY_SURS = {
    "classifie", "classifier", "classifions", "classifiez", "classifient",
    "classify", "classifies", "classifying",
    "categorize", "categorizes", "categorizing",
    "categorise", "categorises", "categorising",
}
_MOT_CLASSIFY_IMPERATIF = "classe"

# "catégorie" et "classification" sont de purs noms : ils peuvent aussi bien
# désigner une DEMANDE de classement du document ("Quelle catégorie
# correspond à ce rapport ?") qu'un FAIT mentionné dans son contenu
# ("Quelle est la classification de risque mentionnée dans ce document ?").
# Cette distinction dépend de la structure de la phrase, pas du vocabulaire :
# aucune heuristique lexicale fermée ne peut la trancher de façon fiable.
# `_detecter_intention` renvoie donc un sentinel dédié pour ce cas précis, et
# `noeud_detecter_intention` le résout par un classifieur LLM borné (voir
# `_desambiguiser_intention_classify`) — jamais par un mot-clé supplémentaire
# qui recréerait le même risque de faux positif.
_JETONS_CLASSIFY_AMBIGUS = {"categorie", "classification", "categorisation"}

_AMBIGU_CLASSIFY = "ambigu_classify"

_SYSTEME_DESAMBIGUISATION_CLASSIFY = """Tu distingues deux intentions possibles pour une requête adressée à un agent documentaire.

SEARCH : la requête pose une question factuelle. Cela reste vrai même si la
question porte sur une catégorie, une classification ou un type mentionné
DANS le contenu d'un document (ex. « Quelle est la classification de risque
mentionnée dans ce document ? » — on demande un FAIT présent dans le texte).

CLASSIFY : la requête demande explicitement de classer, catégoriser, ou
déterminer la catégorie/le type DU DOCUMENT LUI-MÊME, parmi des catégories
prédéfinies (ex. « Quelle catégorie correspond à ce rapport ? »).

Réponds uniquement avec un objet JSON strict :
{"intention": "SEARCH"|"CLASSIFY"}
"""

# EXTRACT (Action 04) : uniquement des formes VERBALES d'action sûres,
# bilingues — jamais le nom "extraction" seul, pour la même raison que
# "catégorie"/"classification" sont exclus des jetons sûrs de CLASSIFY :
# "Quelle est la méthode d'extraction des données utilisée dans ce
# rapport ?" porte sur un FAIT du document, pas sur une demande
# d'extraction. "extrait"/"extraits" sont exclus pour la même raison que
# "classe" chez CLASSIFY (forme nominale ou verbale au présent, ambiguë) —
# seules les formes non ambiguës (impératif, infinitif) sont retenues.
_JETONS_EXTRACT_SURS = {
    "extrais", "extrayez", "extrayons",
    "extraire",
    "extract", "extracts", "extracting",
}

# Motif purement STRUCTUREL (aucun mot métier) : une formulation répétée
# "champ : ?" (ex. « Fournisseur : ? Date : ? Montant : ? ») signale de
# façon fiable une demande de plusieurs valeurs structurées, quel que soit
# le vocabulaire employé — jamais une liste de noms de champs codée en dur.
_MOTIF_CHAMP_VALEUR = re.compile(r"\S+\s*:\s*\?")

# Signal d'énumération — PAS une condition suffisante à elle seule pour
# router vers EXTRACT (voir _AMBIGU_SEARCH_EXTRACT juste en dessous) : une
# requête énumérant plusieurs éléments peut aussi bien annoncer une
# extraction implicite (« Donne-moi X, Y et Z ») qu'une question analytique
# (« Compare X et Y »). Cette ambiguïté structurelle est précisément ce que
# le classifieur LLM borné doit trancher — jamais une règle Python seule
# du type "si plusieurs éléments sont mentionnés, alors EXTRACT".
_MARQUEURS_ENUMERATION = (",", " et ", " and ", " ainsi que ")

_AMBIGU_SEARCH_EXTRACT = "ambigu_search_extract"

_SYSTEME_DESAMBIGUISATION_EXTRACT = """Tu distingues deux intentions possibles pour une requête adressée à un agent documentaire.

SEARCH : la requête pose une question factuelle ou analytique, y compris
lorsqu'elle porte sur plusieurs éléments à la fois (ex. « Compare le chiffre
d'affaires 2024 et 2025 », « Quelle est la différence entre X et Y ? »). Le
but attendu est une réponse rédigée, pas une liste de champs structurés.

EXTRACT : la requête demande explicitement de récupérer plusieurs
informations, valeurs ou attributs INDÉPENDANTS d'un document, sous forme
structurée (ex. « Donne-moi le fournisseur, la date et le montant total. »,
« Quels sont le numéro de facture et la date d'échéance ? »). Le but attendu
est une liste de champs et de valeurs, pas une réponse rédigée ni une
analyse comparative.

Réponds uniquement avec un objet JSON strict :
{"intention": "SEARCH"|"EXTRACT"}
"""


# ==========================================================================
# P1.2 — Extensions de vocabulaire de routage (étape "routing dédiée")
# ==========================================================================
#
# Prévu par docs/DO_NOT_TOUCH.md §3. Corrige des lacunes de SOUS-détection
# mesurées par evaluation/evaluate_routing.py (cas RT-028, RT-031..RT-034,
# RT-038, RT-039, RT-041, RT-048, RT-049). Règles suivies :
#
#   - expressions MULTI-MOTS uniquement, jamais un jeton isolé : « type » ou
#     « nature » seuls recréent exactement le faux positif que la zone grise
#     _JETONS_CLASSIFY_AMBIGUS existe pour éviter ;
#   - aucun vocabulaire métier, aucune liste de types de documents ;
#   - l'invariant « SEARCH ne perd jamais un cas » est verrouillé par le banc
#     (SEARCH 24/24) et par tests/agent/test_nodes.py.
#
# Comparées en sous-chaîne sur le texte normalisé, comme _BIGRAMMES_SUMMARIZE.

# SUMMARIZE : demandes indirectes de synthèse d'un document.
_EXPRESSIONS_SUMMARIZE = (
    "points essentiels",
    "idees principales",
    "idee principale",
    "grands axes",
    "main takeaways",
    "key takeaways",
    "tl;dr",
    "tl dr",
    "tl-dr",
)

# CLASSIFY (sûr) : la requête porte explicitement sur le TYPE / la NATURE du
# document LUI-MÊME (objet non ambigu : « ... de ce document »), ou demande
# impérativement de le déterminer. Distinct de _JETONS_CLASSIFY_AMBIGUS, où
# « catégorie »/« classification » nus peuvent désigner un fait du contenu.
_EXPRESSIONS_CLASSIFY_SURES = (
    "type de document",
    "type de ce document",
    "type de fichier",
    "type de ce fichier",
    "quel type de document",
    "quel genre de document",
    "nature de ce document",
    "nature du document",
    "nature de ce fichier",
    "determine la nature de",
    "determiner la nature de",
    "determine le type de",
    "determiner le type de",
    "is this document a",
    "type of document",
    "kind of document",
    "document type",
)

# CLASSIFY (ambigu) : « s'agit-il d'un X ou d'un Y ? » — souvent une
# identification de type de document, mais aussi bien une question factuelle
# (« s'agit-il d'un montant HT ou TTC ? »). Tranché par
# _desambiguiser_intention_classify, jamais ici.
_EXPRESSIONS_AMBIGU_CLASSIFY = (
    "s'agit-il d'un",
    "s'agit-il d'une",
)

# EXTRACT : verbe de récupération. Seul, « récupère » est ambigu (« récupère
# le fichier joint ») ; combiné à une énumération, il désigne sans ambiguïté
# une extraction structurée de plusieurs champs.
_JETONS_EXTRACT_RECUPERATION = {
    "recupere", "recuperer", "recuperez", "recuperons",
    "recueille", "recueillir", "recueillez",
}
_EXPRESSIONS_EXTRACT_SURES = (
    "recupere les champs",
    "recuperer les champs",
    "recupere les informations",
    "recupere les valeurs",
    "recupere les donnees",
)


def _normaliser_intention(texte: str) -> str:
    """Minuscule, sans accents — utilitaire local, l'agent ne dépend pas des
    fonctions privées du Core (`retrieval._normaliser_texte`)."""
    brut = unicodedata.normalize("NFKD", str(texte))
    brut = "".join(caractere for caractere in brut if not unicodedata.combining(caractere))
    return brut.lower()


def _detecter_intention(requete: str) -> str:
    """
    Classifie une requête en ``"search"``, ``"summarize"``, ``"classify"``,
    ``"extract"``, ``"ambigu_classify"`` ou ``"ambigu_search_extract"``
    (zones grises résolues ensuite par `_desambiguiser_intention_classify`/
    `_desambiguiser_intention_search_extract`, voir `noeud_detecter_intention`).

    Déterministe, bornée, testable indépendamment du graphe et du corpus.
    """
    normalisee = _normaliser_intention(requete)
    jetons = _MOTIF_MOTS_INTENTION.findall(normalisee)
    ensemble_jetons = set(jetons)

    if ensemble_jetons & _JETONS_SUMMARIZE:
        return "summarize"
    if any(bigramme in normalisee for bigramme in _BIGRAMMES_SUMMARIZE):
        return "summarize"
    if any(expression in normalisee for expression in _EXPRESSIONS_SUMMARIZE):
        return "summarize"

    if ensemble_jetons & _JETONS_CLASSIFY_SURS:
        return "classify"
    if jetons and jetons[0] == _MOT_CLASSIFY_IMPERATIF:
        return "classify"
    if any(expression in normalisee for expression in _EXPRESSIONS_CLASSIFY_SURES):
        return "classify"

    if ensemble_jetons & _JETONS_CLASSIFY_AMBIGUS:
        return _AMBIGU_CLASSIFY
    if any(expression in normalisee for expression in _EXPRESSIONS_AMBIGU_CLASSIFY):
        return _AMBIGU_CLASSIFY

    if ensemble_jetons & _JETONS_EXTRACT_SURS:
        return "extract"
    if any(expression in normalisee for expression in _EXPRESSIONS_EXTRACT_SURES):
        return "extract"
    if _MOTIF_CHAMP_VALEUR.search(requete):
        return "extract"

    marqueur_enumeration = any(
        marqueur in normalisee for marqueur in _MARQUEURS_ENUMERATION
    )
    if marqueur_enumeration and (ensemble_jetons & _JETONS_EXTRACT_RECUPERATION):
        return "extract"
    if marqueur_enumeration:
        return _AMBIGU_SEARCH_EXTRACT

    return "search"


def _desambiguiser_intention_classify(llm: Any, requete: str) -> str:
    """
    Classifieur LLM borné, appelé uniquement pour `_AMBIGU_CLASSIFY` : une
    seule évaluation, ensemble de sortie fermé ``{"search", "classify"}``,
    jamais de choix de tool par le LLM lui-même. Toute sortie invalide ou
    tout échec (LLM indisponible, JSON invalide) retombe sur ``"search"`` —
    repli sûr vers le chemin existant, jamais l'inverse.
    """
    try:
        texte = invoquer_llm(
            llm,
            systeme=_SYSTEME_DESAMBIGUISATION_CLASSIFY,
            utilisateur=requete,
        )
        objet = extraire_json_objet(texte)
        valeur = str(objet.get("intention", "")).strip().upper()
        if valeur == "CLASSIFY":
            return "classify"
    except Exception as exc:  # noqa: BLE001 — classifieur borné, repli conservateur
        logger.warning("Désambiguïsation d'intention CLASSIFY impossible : %s", exc)
    return "search"


def _desambiguiser_intention_search_extract(llm: Any, requete: str) -> str:
    """
    Classifieur LLM borné, appelé uniquement pour `_AMBIGU_SEARCH_EXTRACT` :
    une seule évaluation, ensemble de sortie fermé ``{"search", "extract"}``,
    jamais de choix de tool par le LLM lui-même, jamais de choix de
    document, jamais de champ à extraire (voir `_parser_champs_extraction`,
    un appel borné séparé, déclenché seulement une fois EXTRACT confirmé).
    Toute sortie invalide ou tout échec (LLM indisponible, JSON invalide)
    retombe sur ``"search"`` — repli sûr vers le chemin existant, jamais
    l'inverse.
    """
    try:
        texte = invoquer_llm(
            llm,
            systeme=_SYSTEME_DESAMBIGUISATION_EXTRACT,
            utilisateur=requete,
        )
        objet = extraire_json_objet(texte)
        valeur = str(objet.get("intention", "")).strip().upper()
        if valeur == "EXTRACT":
            return "extract"
    except Exception as exc:  # noqa: BLE001 — classifieur borné, repli conservateur
        logger.warning("Désambiguïsation d'intention EXTRACT impossible : %s", exc)
    return "search"


_SYSTEME_PARSING_CHAMPS_EXTRACTION = """Tu identifies la liste des informations demandées dans une requête adressée à un système d'extraction documentaire.

RÈGLES STRICTES
- Identifie uniquement les champs/informations EXPLICITEMENT demandés dans la requête.
- N'invente jamais un champ absent de la requête.
- Reformule chaque champ en une courte étiquette (2 à 6 mots), sans changer son sens.
- Conserve la langue de la requête.
- Si aucun champ n'est identifiable, renvoie une liste vide.

Réponds uniquement avec un objet JSON strict :
{"champs": ["...", "..."]}
"""


def _parser_champs_extraction(llm: Any, requete: str) -> list[str]:
    """
    Appel LLM borné (Action 04), déclenché uniquement une fois l'intention
    EXTRACT déjà confirmée : une seule évaluation, sortie structurée stricte
    ``{"champs": [...]}``, validée en Python (liste de chaînes non vides,
    dédupliquées, dans leur ordre d'apparition). Ce parseur ne choisit ni
    document, ni tool, ni catégorie — il segmente uniquement la requête en
    champs demandés, sans jamais en inventer un absent. Tout échec (LLM
    indisponible, JSON invalide, structure inattendue) retombe sur une liste
    vide : `extract` échoue alors proprement de lui-même ("Aucun champ
    d'extraction valide n'a été fourni."), jamais un champ inventé.
    """
    try:
        texte = invoquer_llm(
            llm,
            systeme=_SYSTEME_PARSING_CHAMPS_EXTRACTION,
            utilisateur=requete,
        )
        objet = extraire_json_objet(texte)
        champs_bruts = objet.get("champs", [])

        if not isinstance(champs_bruts, list):
            return []

        champs: list[str] = []
        for champ in champs_bruts:
            champ = " ".join(str(champ).split())
            if champ and champ not in champs:
                champs.append(champ)
        return champs

    except Exception as exc:  # noqa: BLE001 — parseur borné, repli conservateur
        logger.warning("Extraction des champs demandés impossible : %s", exc)
        return []


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


# Intentions qu'un signal multi-document (P1.4) peut supplanter. On ne
# détourne JAMAIS une demande explicite de classification ou d'extraction,
# ni une zone grise non résolue.
_INTENTIONS_SUPPLANTABLES_PAR_MULTIDOC = frozenset({"search", "summarize"})

# Nombre minimal de références de fichiers explicites pour qu'un verbe de
# comparaison/synthèse prenne la précédence sur une zone grise (P1.5, §2.7).
MINIMUM_REFERENCES_MULTIDOC = 2


def _appliquer_signal_multidoc(intention: str, signal: Any) -> str:
    """
    Routing minimal P1.5 : une intention SEARCH/SUMMARIZE bascule vers
    COMPARE/SYNTHESIZE **uniquement** si le signal multi-document est
    explicite (`is_multidoc` et `operation_hint` ∈ {compare, synthesize}).
    Sinon l'intention est renvoyée intacte.

    Fonction pure, partagée avec `evaluation.evaluate_routing` pour que le
    banc mesure exactement le même routage.
    """
    if (
        intention in _INTENTIONS_SUPPLANTABLES_PAR_MULTIDOC
        and getattr(signal, "is_multidoc", False)
        and getattr(signal, "operation_hint", "none") in {"compare", "synthesize"}
    ):
        return signal.operation_hint
    return intention


def noeud_detecter_intention(etat: EtatGraphe) -> dict:
    """
    Premier nœud du graphe : SEARCH, SUMMARIZE, CLASSIFY, EXTRACT, ou —
    depuis P1.5 — COMPARE / SYNTHESIZE lorsque le signal multi-document
    (`src.agent.multidoc`, déterministe, sans LLM) l'indique explicitement.

    La détection lexicale (`_detecter_intention`) tranche seule dans
    l'immense majorité des cas, sans LLM. Deux zones grises, chacune
    résolue par son propre classifieur LLM borné — jamais par un mot-clé
    supplémentaire, qui reproduirait la même ambiguïté :
        - « catégorie »/« classification » employés comme noms (voir
          `_JETONS_CLASSIFY_AMBIGUS`) -> `_desambiguiser_intention_classify` ;
        - énumération de plusieurs éléments sans verbe d'extraction ni motif
          structurel « champ : ? » (voir `_MARQUEURS_ENUMERATION`) ->
          `_desambiguiser_intention_search_extract`.

    Le signal multi-document est appliqué APRÈS résolution des zones grises
    (`_appliquer_signal_multidoc`) : il ne supplante que SEARCH/SUMMARIZE.
    """
    session = etat.session
    requete = session.etat.requete_courante

    intention = _detecter_intention(requete)
    signal = detecter_multidoc(requete)

    # P1.5 — PRÉCÉDENCE MULTI-DOCUMENT SUR LES ZONES GRISES.
    # Des références de fichiers explicites (>= 2) combinées à un verbe de
    # comparaison / synthèse explicite ne constituent PAS une zone grise
    # CLASSIFY/EXTRACT : on route directement vers compare/synthesize, SANS
    # appeler de désambiguïsateur LLM. Cela couvre le cas où un mot comme
    # « classification » / « catégorie » dans la requête ferait d'abord
    # tomber `_detecter_intention` dans `_AMBIGU_CLASSIFY` (ou une énumération
    # dans `_AMBIGU_SEARCH_EXTRACT`), que `_appliquer_signal_multidoc` ne
    # supplanterait pas ensuite (il ne supplante que search / summarize).
    multidoc_explicite = (
        getattr(signal, "is_multidoc", False)
        and len(getattr(signal, "references_detectees", ())) >= MINIMUM_REFERENCES_MULTIDOC
        and getattr(signal, "operation_hint", "none") in {"compare", "synthesize"}
    )

    desambiguisation_llm = (
        not multidoc_explicite
        and intention in {_AMBIGU_CLASSIFY, _AMBIGU_SEARCH_EXTRACT}
    )

    if multidoc_explicite:
        intention = signal.operation_hint
    elif intention == _AMBIGU_CLASSIFY:
        intention = _desambiguiser_intention_classify(session.llm, requete)
    elif intention == _AMBIGU_SEARCH_EXTRACT:
        intention = _desambiguiser_intention_search_extract(session.llm, requete)

    intention = _appliquer_signal_multidoc(intention, signal)

    session.etat.ajouter_trace(
        "intention",
        f"Intention détectée : {intention}.",
        intention=intention,
        desambiguisation_llm=desambiguisation_llm,
        multidoc=signal.is_multidoc,
        operation_hint=signal.operation_hint,
        multidoc_explicite=multidoc_explicite,
    )

    return {"session": session, "intention": intention, "multidoc_signal": signal}


def router_intention(etat: EtatGraphe) -> str:
    """
    Arête conditionnelle après `detecter_intention`.

    Lit `etat.intention`, déjà calculé et journalisé par
    `noeud_detecter_intention`, comme `router_apres_evaluation` le fait pour
    ses propres champs : le routage ne doit jamais diverger de ce que dit la
    trace. Toute valeur inattendue (y compris `None`) route vers l'existant
    (`rechercher`), le repli sûr.
    """
    if etat.intention == "summarize":
        return "summarize"
    if etat.intention == "classify":
        return "classify"
    if etat.intention == "extract":
        return "extract"
    if etat.intention == "compare":
        return "compare"
    if etat.intention == "synthesize":
        return "synthesize"
    return "rechercher"


def _resoudre_perimetre_document(requete: str, corpus_id: str = "default") -> tuple[Any, str]:
    """
    Résolution documentaire partagée par `noeud_summarize` et
    `noeud_classify` : texte libre -> `PerimetreDocumentaire`, via
    `resoudre_document` — résolution floue déjà publique et déjà utilisée en
    interne par `search`/`rechercher_passages`, explicitement documentée
    comme « réutilisable par l'agent » (`src/rag/retrieval.py`). Aucune
    logique de résolution n'est réimplémentée ici.

    Ne casse jamais le graphe : toute erreur retombe sur l'absence de
    périmètre (`None`), journalisée comme telle par l'appelant.

    Returns:
        ``(perimetre, statut)`` — ``perimetre`` est `None` si la résolution
        a échoué techniquement ; ``statut`` vaut alors ``"erreur"``, sinon
        `perimetre.statut` (``"exact"``, ``"compatible"``, ``"ambigu"`` ou
        ``"aucun"``).
    """
    try:
        perimetre = resoudre_document(requete, corpus_id=corpus_id)
        return perimetre, perimetre.statut
    except Exception as exc:  # noqa: BLE001 — la résolution ne doit jamais casser le graphe
        logger.warning("Résolution documentaire impossible : %s", exc)
        return None, "erreur"


# ---------------------------------------------------------------------------
# Ciblage documentaire partagé par SUMMARIZE / CLASSIFY / EXTRACT
# ---------------------------------------------------------------------------
#
# Le nom exact d'un fichier n'est JAMAIS une précondition. Un document désigné
# de façon fiable par la requête reste prioritaire (mode document complet) ;
# sinon, la recherche existante (outil `search`, donc `src.rag.retrieval`,
# non modifié) désigne les documents pertinents. Seuls les documents
# réellement retrouvés sont traités, et la réponse ne cite que leurs
# passages. Une précision n'est demandée que si la recherche ne retrouve
# aucun document pertinent.

# Nombre maximal de documents traités lorsque plusieurs correspondent :
# borne le coût (un appel d'outil par document pour CLASSIFY / EXTRACT) et
# la taille de la réponse.
MAX_DOCUMENTS_CIBLES = 3

# Un document retrouvé par la recherche n'est retenu que si son meilleur
# passage atteint cette fraction du meilleur score de la recherche (en plus
# de `SEUIL_PERTINENCE_MINIMALE`) : écarte les documents qui n'apparaissent
# qu'en queue de top-k. Choix de conception générique, non calibré sur un
# corpus particulier.
RATIO_PERTINENCE_DOCUMENT = 0.5

# "document_complet" : valeur historique de la trace, conservée pour ses
# consommateurs (harnais d'évaluation).
MODE_DOCUMENT_NOMME = "document_complet"
MODE_DOCUMENTS_NOMMES = "documents_nommes"
MODE_CIBLE_PAR_RECHERCHE = "cible_par_recherche"
MODE_AUCUN_DOCUMENT_PERTINENT = "aucun_document_pertinent"


@dataclass(frozen=True)
class DocumentCible:
    """Document retenu : `identifiant` est transmis aux outils, `libelle` affiché."""

    identifiant: str
    libelle: str


@dataclass(frozen=True)
class CiblageDocumentaire:
    """
    Documents sur lesquels exécuter SUMMARIZE / CLASSIFY / EXTRACT.

    ``mode`` vaut :
        - ``document_complet``         : un document désigné de façon fiable ;
        - ``documents_nommes``         : plusieurs documents désignés (statut
          ``compatible`` du résolveur), tous retenus ;
        - ``cible_par_recherche``      : documents désignés par la recherche ;
        - ``aucun_document_pertinent`` : la recherche n'a rien retrouvé.
    """

    mode: str
    documents: tuple[DocumentCible, ...] = ()
    statut_resolution: str = ""

    @property
    def libelles(self) -> tuple[str, ...]:
        return tuple(document.libelle for document in self.documents)

    @property
    def identifiants(self) -> tuple[str, ...]:
        return tuple(document.identifiant for document in self.documents)

    @property
    def document_demande(self) -> str | None:
        """Identifiant du document unique désigné par la requête, sinon None."""
        return self.documents[0].identifiant if self.mode == MODE_DOCUMENT_NOMME else None

    @property
    def document_complet(self) -> bool:
        """
        Vrai si les documents sont traités en entier : désignés par la
        requête, ou document unique retrouvé par la recherche. Plusieurs
        documents retrouvés par la recherche : seuls leurs passages
        retrouvés sont traités (réponse ciblée sur la demande, coût borné).
        """
        return self.mode != MODE_CIBLE_PAR_RECHERCHE or len(self.documents) == 1


def _documents_retrouves(session: SessionAgent, requete: str) -> tuple[DocumentCible, ...]:
    """
    Documents désignés par la recherche, du plus au moins pertinent.

    Réutilise les preuves déjà présentes si une recherche a déjà eu lieu dans
    cette session ; sinon exécute `search` via le registre (les passages
    rejoignent `ContexteOutil.sources`, donc la trace et les citations).
    Chaque document est noté par le score de son meilleur passage
    (`Passage.score_final`) ; ne sont retenus que ceux qui dépassent
    `SEUIL_PERTINENCE_MINIMALE` et `RATIO_PERTINENCE_DOCUMENT` × meilleur
    score, dans la limite de `MAX_DOCUMENTS_CIBLES`. Déterministe pour une
    recherche donnée.
    """
    if not session.a_des_preuves:
        session.executer_outil("search", requete=requete)

    # Passages du dernier `RapportRecherche` (scores de reranking) ; à défaut,
    # sources déjà présentes dans le contexte (`SourceOutil.score`).
    rapport = session.contexte.dernier_rapport_recherche
    preuves = list(getattr(rapport, "passages", None) or session.contexte.sources)

    meilleurs: dict[str, tuple[float, str]] = {}
    for preuve in preuves:
        identifiant = preuve.doc_id or preuve.nom_fichier or preuve.source
        if not identifiant:
            continue
        score = float(
            (preuve.score_final if hasattr(preuve, "score_final") else preuve.score) or 0.0
        )
        libelle = preuve.nom_fichier or preuve.source or preuve.doc_id
        if identifiant not in meilleurs or score > meilleurs[identifiant][0]:
            meilleurs[identifiant] = (score, libelle)

    if not meilleurs:
        return ()

    score_max = max(score for score, _ in meilleurs.values())
    seuil = max(SEUIL_PERTINENCE_MINIMALE, RATIO_PERTINENCE_DOCUMENT * score_max)
    classes = sorted(meilleurs.items(), key=lambda item: -item[1][0])
    return tuple(
        DocumentCible(identifiant=identifiant, libelle=libelle)
        for identifiant, (score, libelle) in classes
        if score >= seuil
    )[:MAX_DOCUMENTS_CIBLES]


def _cibler_documents(session: SessionAgent, requete: str) -> CiblageDocumentaire:
    """
    Détermine le ou les documents à traiter, sans jamais exiger de nom exact.

    1. Résolution documentaire fiable (`perimetre.contraignant`) : le ou les
       documents désignés, sans recherche (au plus `MAX_DOCUMENTS_CIBLES`).
    2. Sinon (aucune référence, référence ambiguë ou sous le seuil) : les
       documents retrouvés par la recherche (`_documents_retrouves`).
    3. Recherche sans résultat pertinent : `aucun_document_pertinent`.
    """
    perimetre, statut = _resoudre_perimetre_document(
        requete, corpus_id=session.contexte.corpus_id
    )

    if perimetre is not None and perimetre.contraignant:
        valeurs = perimetre.valeurs_filtre[:MAX_DOCUMENTS_CIBLES]
        libelles = perimetre.libelles if len(perimetre.libelles) == len(perimetre.valeurs_filtre) else ()
        documents = tuple(
            DocumentCible(identifiant=valeur, libelle=libelles[i] if libelles else valeur)
            for i, valeur in enumerate(valeurs)
        )
        mode = MODE_DOCUMENT_NOMME if len(documents) == 1 else MODE_DOCUMENTS_NOMMES
        return CiblageDocumentaire(mode=mode, documents=documents, statut_resolution=statut)

    documents = _documents_retrouves(session, requete)
    if not documents:
        return CiblageDocumentaire(mode=MODE_AUCUN_DOCUMENT_PERTINENT, statut_resolution=statut)
    return CiblageDocumentaire(
        mode=MODE_CIBLE_PAR_RECHERCHE, documents=documents, statut_resolution=statut
    )


def _refus_aucun_document(session: SessionAgent, outil: str) -> ResultatOutil:
    """Seul cas où une précision est demandée : rien de pertinent n'a été retrouvé."""
    resultat = ResultatOutil.echec(
        outil,
        "Aucun document du corpus ne correspond à cette demande. "
        "Reformule la demande ou précise le document visé.",
        motif=MODE_AUCUN_DOCUMENT_PERTINENT,
    )
    session.contexte.ajouter_resultat(resultat)
    return resultat


def _executer_par_document(
    session: SessionAgent,
    outil: str,
    ciblage: CiblageDocumentaire,
    *,
    document_complet: bool,
    **arguments: Any,
) -> ResultatOutil:
    """
    Exécute `outil` (classify ou extract, qui refusent de mélanger plusieurs
    documents) une fois par document ciblé, puis assemble les résultats.

    ``document_complet`` : document entier (`documents=[id]`) — voir
    `CiblageDocumentaire.document_complet` ; sinon passages déjà retrouvés par la
    recherche, cloisonnés à ce document (`document=id`).

    Un seul document : le résultat de l'outil est rendu tel quel. Plusieurs :
    un résultat unique, document par document ; les citations de chaque
    document sont préfixées `D<n>` (convention multi-document existante,
    ex. `D1S2`) pour rester sans collision.
    """
    resultats: list[tuple[DocumentCible, ResultatOutil]] = []
    for document in ciblage.documents:
        cible = (
            {"documents": [document.identifiant]}
            if document_complet
            else {"document": document.identifiant}
        )
        resultats.append((document, session.executer_outil(outil, **arguments, **cible)))

    if len(resultats) == 1:
        return resultats[0][1]

    lignes = [f"{len(resultats)} documents traités :"]
    par_document: list[dict[str, Any]] = []
    citations: list[str] = []
    sources: list[Any] = []
    avertissements: list[str] = []
    for rang, (document, resultat) in enumerate(resultats, start=1):
        lignes.append(f"- [D{rang}] {document.libelle} : {resultat.message}")
        par_document.append(
            {
                "reference": f"D{rang}",
                "document": document.libelle,
                "succes": resultat.succes,
                "message": resultat.message,
                "donnees": resultat.donnees,
            }
        )
        citations.extend(f"D{rang}{c}" for c in (resultat.donnees.get("citations") or []))
        if resultat.succes:
            sources.extend(resultat.sources)
        avertissements.extend(f"[D{rang}] {a}" for a in resultat.avertissements)

    succes = any(resultat.succes for _, resultat in resultats)
    donnees: dict[str, Any] = {
        "documents": list(ciblage.libelles),
        "resultats_par_document": par_document,
        "citations": citations,
    }
    if not succes:
        donnees["motif"] = "echec_tous_documents"
    return ResultatOutil(
        outil=outil,
        succes=succes,
        message="\n".join(lignes),
        donnees=donnees,
        sources=sources,
        avertissements=avertissements,
    )


def _mise_a_jour_noeud(
    session: SessionAgent, resultat: ResultatOutil, ciblage: CiblageDocumentaire
) -> dict:
    """
    Champs renvoyés au graphe. Quand les documents ont été désignés par la
    recherche, la réponse le dit explicitement : l'utilisateur voit sur quels
    documents porte le résultat, même s'il en visait un autre.
    """
    if ciblage.mode == MODE_CIBLE_PAR_RECHERCHE:
        resultat.avertissements.append(
            "Aucun document désigné par son nom : documents retenus par la "
            "recherche — " + ", ".join(ciblage.libelles) + "."
        )
    return {
        "session": session,
        "reponse": resultat,
        "documents_resolus": ciblage.libelles,
    }


def noeud_summarize(etat: EtatGraphe) -> dict:
    """
    Exécute l'outil `summarize` via le registre, sur les documents désignés
    par `_cibler_documents` (le nom exact n'est jamais requis) :

        - un seul document (désigné par la requête ou seul retrouvé par la
          recherche) : `summarize(documents=[id])`, document complet ;
        - plusieurs documents désignés par la requête :
          `summarize(documents=[ids])` — l'outil résume déjà plusieurs
          documents en conservant la provenance de chaque information ;
        - plusieurs documents retrouvés par la recherche :
          `summarize(documents=None)` sur les passages retrouvés (mode
          contextuel) — un résumé ciblé sur la demande, cité ;
        - rien de pertinent : refus demandant de préciser, sans appeler
          `summarize`.

    Succès ou échec, le `ResultatOutil` devient directement la réponse finale.
    """
    session = etat.session
    requete = session.etat.requete_courante
    ciblage = _cibler_documents(session, requete)

    documents_demandes: list[str] | None = None
    if ciblage.mode == MODE_AUCUN_DOCUMENT_PERTINENT:
        resultat = _refus_aucun_document(session, "summarize")
    elif ciblage.document_complet:
        documents_demandes = list(ciblage.identifiants)
        resultat = session.executer_outil(
            "summarize", objectif=requete, documents=documents_demandes
        )
    else:
        resultat = session.executer_outil("summarize", objectif=requete, documents=None)

    session.etat.ajouter_trace(
        "summarize",
        "Résumé produit." if resultat.succes else "Résumé impossible.",
        succes=resultat.succes,
        documents_demandes=documents_demandes,
        document_demande=ciblage.document_demande,
        documents_cibles=list(ciblage.libelles),
        resolution_documentaire=ciblage.statut_resolution,
        mode=ciblage.mode,
    )

    return _mise_a_jour_noeud(session, resultat, ciblage)


def noeud_classify(etat: EtatGraphe) -> dict:
    """
    Exécute l'outil `classify` via le registre, sur les documents désignés
    par `_cibler_documents` (le nom exact n'est jamais requis) :

        - documents désignés par la requête, ou document unique retrouvé
          par la recherche : classification du document complet (Option E
          de `src.tools.classify`), un appel par document ;
        - plusieurs documents retrouvés par la recherche : classification
          des passages retrouvés, cloisonnée document par document
          (`classify` refuse de mélanger plusieurs documents dans un même
          vote) ;
        - rien de pertinent : refus demandant de préciser.

    Catégories : toujours celles du profil technique actif
    (`profil.classification.noms()`), jamais inventées ni demandées à
    l'utilisateur. Succès ou échec, le `ResultatOutil` devient directement
    la réponse finale.
    """
    session = etat.session
    requete = session.etat.requete_courante
    categories = get_profil().classification.noms()
    ciblage = _cibler_documents(session, requete)

    if ciblage.mode == MODE_AUCUN_DOCUMENT_PERTINENT:
        resultat = _refus_aucun_document(session, "classify")
    else:
        resultat = _executer_par_document(
            session,
            "classify",
            ciblage,
            document_complet=ciblage.document_complet,
            categories=categories,
        )

    session.etat.ajouter_trace(
        "classify",
        "Classification produite." if resultat.succes else "Classification impossible.",
        succes=resultat.succes,
        document_demande=ciblage.document_demande,
        documents_cibles=list(ciblage.libelles),
        resolution_documentaire=ciblage.statut_resolution,
        mode=ciblage.mode,
    )

    return _mise_a_jour_noeud(session, resultat, ciblage)


def noeud_extract(etat: EtatGraphe) -> dict:
    """
    Exécute l'outil `extract` via le registre, sur les documents désignés
    par `_cibler_documents` (le nom exact n'est jamais requis) — même
    ciblage que `noeud_classify`, un appel par document.

    Remplace l'invariant P1.6 (EX-03), qui refusait toute extraction sans
    document nommé : le périmètre n'est plus choisi « par convenance » et
    en silence, il est désigné par la recherche selon une règle
    déterministe (`_documents_retrouves`) et chaque document traité est
    nommé dans la réponse, avec ses propres citations.

    L'AGRÉGATION reste spécifique à extract (déduplication de valeurs,
    jamais un vote). Les champs demandés sont obtenus par un appel LLM borné
    distinct (`_parser_champs_extraction`), déclenché une seule fois ici.
    """
    session = etat.session
    requete = session.etat.requete_courante

    champs = _parser_champs_extraction(session.llm, requete)
    ciblage = _cibler_documents(session, requete)

    if ciblage.mode == MODE_AUCUN_DOCUMENT_PERTINENT:
        resultat = _refus_aucun_document(session, "extract")
    else:
        resultat = _executer_par_document(
            session,
            "extract",
            ciblage,
            document_complet=ciblage.document_complet,
            champs=champs,
        )

    session.etat.ajouter_trace(
        "extract",
        "Extraction produite." if resultat.succes else "Extraction impossible.",
        succes=resultat.succes,
        champs_demandes=champs,
        document_demande=ciblage.document_demande,
        documents_cibles=list(ciblage.libelles),
        resolution_documentaire=ciblage.statut_resolution,
        mode=ciblage.mode,
    )

    return _mise_a_jour_noeud(session, resultat, ciblage)


def _signal_multidoc_courant(etat: EtatGraphe, requete: str) -> Any:
    """Réutilise le signal déjà calculé par `noeud_detecter_intention` ;
    le recalcule (déterministe) uniquement en repli défensif."""
    signal = getattr(etat, "multidoc_signal", None)
    if signal is not None:
        return signal
    return detecter_multidoc(requete)


def noeud_compare(etat: EtatGraphe) -> dict:
    """
    Branche COMPARE (P1.5). Compare 2 à 4 documents explicitement nommés par
    l'utilisateur : MAP borné par document (contenu de CE document
    uniquement) -> REDUCE inter-document -> `ResultatOutil` avec provenance
    par document. Aucun search global. Résolution non fiable -> abstention
    déterministe (jamais de repli vers `search`).
    """
    session = etat.session
    requete = session.etat.requete_courante
    signal = _signal_multidoc_courant(etat, requete)

    resultat = comparer(
        requete,
        getattr(signal, "references_detectees", ()),
        llm=session.llm,
        profil_domaine=session.contexte.profil_domaine,
        corpus_id=session.contexte.corpus_id,
    )
    session.contexte.ajouter_resultat(resultat)

    documents = tuple(resultat.donnees.get("comparaison", {}).get("documents", ())) \
        if resultat.succes else ()

    session.etat.ajouter_trace(
        "compare",
        "Comparaison produite." if resultat.succes else "Comparaison impossible.",
        succes=resultat.succes,
        references=list(getattr(signal, "references_detectees", ())),
        documents_resolus=list(documents),
        motif=resultat.donnees.get("motif"),
    )

    return {
        "session": session,
        "reponse": resultat,
        "documents_resolus": documents,
        "resultat_compare": resultat.donnees.get("comparaison") if resultat.succes else None,
    }


def noeud_synthesize(etat: EtatGraphe) -> dict:
    """
    Branche SYNTHESIZE (P1.5). Synthèse transversale de 2 à 4 documents
    explicitement nommés : même structure MAP -> REDUCE que `noeud_compare`.
    Les divergences entre documents sont conservées explicitement. Aucun
    search global ; résolution non fiable -> abstention déterministe.
    """
    session = etat.session
    requete = session.etat.requete_courante
    signal = _signal_multidoc_courant(etat, requete)

    resultat = synthetiser_documents(
        requete,
        getattr(signal, "references_detectees", ()),
        llm=session.llm,
        profil_domaine=session.contexte.profil_domaine,
        corpus_id=session.contexte.corpus_id,
    )
    session.contexte.ajouter_resultat(resultat)

    documents = tuple(resultat.donnees.get("synthese", {}).get("documents", ())) \
        if resultat.succes else ()

    session.etat.ajouter_trace(
        "synthesize",
        "Synthèse produite." if resultat.succes else "Synthèse impossible.",
        succes=resultat.succes,
        references=list(getattr(signal, "references_detectees", ())),
        documents_resolus=list(documents),
        motif=resultat.donnees.get("motif"),
    )

    return {
        "session": session,
        "reponse": resultat,
        "documents_resolus": documents,
        "resultat_synthesize": resultat.donnees.get("synthese") if resultat.succes else None,
    }


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
    "noeud_detecter_intention",
    "router_intention",
    "noeud_summarize",
    "noeud_classify",
    "noeud_extract",
    "noeud_compare",
    "noeud_synthesize",
    "noeud_rechercher",
    "noeud_evaluer_preuves",
    "router_apres_evaluation",
    "noeud_reformuler",
    "noeud_generer_reponse",
    "VerdictSuffisance",
]
