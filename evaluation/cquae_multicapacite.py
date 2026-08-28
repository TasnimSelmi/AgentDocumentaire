"""
Harnais d'évaluation end-to-end MULTI-CAPACITÉS sur CQuAE.

Contrairement à `evaluate_agent.py` (SEARCH uniquement, suppose que
`resultat["reponse"]` est toujours un `ReponseRAG`), ce module gère les
QUATRE branches du graphe agentique — SEARCH, SUMMARIZE, CLASSIFY, EXTRACT —
dont les deux dernières renvoient un `ResultatOutil` (`src.tools.base`), une
forme structurellement différente d'un `ReponseRAG`.

Séparation gold stricte, structurelle
--------------------------------------
`executer_cas()` est la SEULE fonction de ce module qui touche l'agent. Elle
ne reçoit et ne lit que `cas.query` (une chaîne) ; aucun champ gold
(`gold_qids`, `champs[].gold_qid`, `source_document` même) n'est accédé avant
son retour. Le gold n'est chargé — via `evaluation/data/cquae_agent_gold.jsonl`,
par identifiant — que dans `noter_cas()`, appelée après coup avec le résultat
déjà produit. `tests/evaluation/test_cquae_multicapacite.py::test_aucune_fuite_gold`
vérifie ceci dynamiquement (agent mocké, aucun appel réseau).

Ce module ne modifie ni n'importe de logique de `src/rag/`, `src/tools/` ou
`src/agent/` : il consomme leurs points d'entrée publics déjà utilisés par
`evaluation/evaluate_agent.py` (`construire_graphe`, `construire_session`,
`EtatGraphe`) et par les autres scripts d'évaluation (`evaluation.common`).

Sécurité d'exécution
---------------------
`main()` vérifie TOUJOURS les préconditions (profils actifs, collection
Qdrant, couverture du gold, exclusions) et les affiche. Il ne déclenche un
appel agent QUE si `--executer` est passé explicitement : lancé sans cet
argument, ce module est un simple contrôle à blanc, jamais un run réel — un
garde-fou indépendant de toute discipline humaine à la CLI.

Exemple (une fois `--executer` autorisé par vous)
--------------------------------------------------
    QDRANT_PATH=data/vectordb/qdrant_cquae_eval \\
    ACTIVE_PROFILE=generic \\
    ACTIVE_DOMAIN_PROFILE=histoire-culture-humaines \\
    python -m evaluation._runner_config \\
        --surcharges '{"qdrant.nom_collection": "cquae_eval"}' \\
        -- evaluation.cquae_multicapacite --executer --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evaluation.common import (
    DOSSIER_DONNEES,
    DOSSIER_RAPPORTS,
    Enregistrement,
    cle_document,
    charger_enregistrements,
    configurer_logs,
    ensemble_jetons,
    horodatage,
    mesurer_couverture,
)

logger = logging.getLogger("evaluation.cquae_multicapacite")

# ===========================================================================
# Constantes de préconditions — jamais utilisées pour AJUSTER un résultat,
# seulement pour vérifier que l'environnement est celui attendu avant de
# lancer quoi que ce soit contre l'agent.
# ===========================================================================

FICHIER_CAS_PAR_DEFAUT = DOSSIER_DONNEES / "cquae_smoke_cases.jsonl"
FICHIER_GOLD_PAR_DEFAUT = DOSSIER_DONNEES / "cquae_agent_gold.jsonl"

PROFIL_TECHNIQUE_ATTENDU = "generic"
PROFIL_DOMAINE_ATTENDU = "histoire-culture-humaines"
COLLECTION_ATTENDUE = "cquae_eval"
FRAGMENT_QDRANT_PATH_ATTENDU = "qdrant_cquae_eval"
DOCUMENT_MANQUANT_CONNU = "cquae_doc_2262.txt"

SEUIL_RECOUVREMENT_VALEUR = 0.3  # part des jetons du gold retrouvés dans une valeur extraite/résumée
SEUIL_RECOUVREMENT_SEMANTIQUE = 0.6  # part des jetons PORTEURS du gold retrouvés dans une réponse SEARCH rédigée

VERDICTS = (
    "PASS",
    "ANSWER_ONLY",
    "RETRIEVAL_ONLY",
    "UNSUPPORTED",
    "WRONG",
    "ABSTAIN_CORRECT",
    "TECHNICAL_ERROR",
)

FAILURE_CATEGORIES = (
    "ROUTING_FAILURE",
    "DOCUMENT_RETRIEVAL_FAILURE",
    "EVIDENCE_RETRIEVAL_FAILURE",
    "GENERATION_FAILURE",
    "PROVENANCE_FAILURE",
    "EXTRACTION_FAILURE",
    "CLASSIFICATION_FAILURE",
    "SUMMARIZATION_FAITHFULNESS_FAILURE",
    "CORRECT_ABSTENTION",
    "TECHNICAL_FAILURE",
    "MISSING_DOCUMENT_EXCLUDED",
    "UNKNOWN",
)


# ===========================================================================
# 1. Cas de test (manifeste, jamais le gold lui-même)
# ===========================================================================


@dataclass
class CasSmoke:
    """
    Un cas du smoke benchmark multi-capacités.

    Ne contient AUCUNE valeur attendue en clair : seulement des références
    (`gold_qids`, `champs[].gold_qid`) vers `cquae_agent_gold.jsonl`, résolues
    uniquement au moment du scoring — jamais avant l'appel agent.
    """

    test_id: str
    capability: str
    query: str
    expected_tool: str
    native_gold: bool
    evaluation_type: str
    source_document: str | None
    gold_qids: list[str] = field(default_factory=list)
    champs: list[dict[str, Any]] | None = None
    notes: str = ""


def charger_cas(chemin: Path = FICHIER_CAS_PAR_DEFAUT) -> list[CasSmoke]:
    cas: list[CasSmoke] = []
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        ligne = ligne.strip()
        if not ligne:
            continue
        donnees = json.loads(ligne)
        cas.append(CasSmoke(**donnees))
    return cas


# ===========================================================================
# 2. Résultat observable (schéma imposé, section 9 de la mission)
# ===========================================================================


@dataclass
class ResultatCas:
    test_id: str
    capability: str
    query: str
    native_gold: bool
    expected_tool: str
    detected_tool: str | None
    tools_executed: list[str]
    source_document: str | None
    agent_success: bool
    agent_result: Any
    retrieved_sources: list[str]
    citations: list[str]
    routing_status: str
    retrieval_status: str
    answer_status: str
    provenance_status: str
    final_verdict: str
    failure_category: str | None
    latency_seconds: float
    error: str
    details: dict[str, Any] = field(default_factory=dict)
    judge_secondaire: dict[str, Any] | None = None

    def vers_dict(self) -> dict[str, Any]:
        return asdict(self)


# ===========================================================================
# 3. Préconditions — vérifiées, jamais supposées
# ===========================================================================


def verifier_preconditions(
    cas: list[CasSmoke],
    fichier_gold: Path = FICHIER_GOLD_PAR_DEFAUT,
) -> tuple[bool, list[str]]:
    """
    Vérifie profils actifs, collection Qdrant, couverture gold, exclusions.

    Ne lance AUCUN appel agent. Retourne (ok, messages) — `ok=False` doit
    interrompre `main()` avant tout `--executer`.
    """
    messages: list[str] = []
    ok = True

    from src.config import get_config_technique, get_settings

    settings = get_settings()
    technique = get_config_technique()

    if settings.active_profile != PROFIL_TECHNIQUE_ATTENDU:
        ok = False
        messages.append(
            f"ACTIVE_PROFILE={settings.active_profile!r} != {PROFIL_TECHNIQUE_ATTENDU!r} attendu."
        )
    else:
        messages.append(f"ACTIVE_PROFILE = {settings.active_profile!r} (OK).")

    if settings.active_domain_profile != PROFIL_DOMAINE_ATTENDU:
        ok = False
        messages.append(
            f"ACTIVE_DOMAIN_PROFILE={settings.active_domain_profile!r} != "
            f"{PROFIL_DOMAINE_ATTENDU!r} attendu. STOP requis (mission, point 1)."
        )
    else:
        messages.append(f"ACTIVE_DOMAIN_PROFILE = {settings.active_domain_profile!r} (OK).")

    if technique.qdrant.nom_collection != COLLECTION_ATTENDUE:
        ok = False
        messages.append(
            f"Collection configurée={technique.qdrant.nom_collection!r} != "
            f"{COLLECTION_ATTENDUE!r} attendu. "
            "Utiliser evaluation._runner_config --surcharges "
            f'\'{{"qdrant.nom_collection": "{COLLECTION_ATTENDUE}"}}\'.'
        )
    else:
        messages.append(f"Collection configurée = {technique.qdrant.nom_collection!r} (OK).")

    if FRAGMENT_QDRANT_PATH_ATTENDU not in str(settings.qdrant_path):
        ok = False
        messages.append(
            f"QDRANT_PATH={settings.qdrant_path} ne contient pas "
            f"{FRAGMENT_QDRANT_PATH_ATTENDU!r} — risque de pointer vers un "
            "corpus autre que CQuAE."
        )
    else:
        messages.append(f"QDRANT_PATH = {settings.qdrant_path} (OK).")

    # --- Collection réellement accessible, sans lancer de test agent -------
    try:
        from src.rag.vectorstore import get_client, info_collection

        get_client()
        infos = info_collection()
        messages.append(f"Collection Qdrant : {infos}")
        if not infos.get("existe"):
            ok = False
            messages.append("Collection absente ou vide.")
    except Exception as exc:  # noqa: BLE001
        ok = False
        messages.append(f"Qdrant inaccessible : {exc}")

    # --- Gold : couverture et exclusion connue ------------------------------
    if not fichier_gold.exists():
        ok = False
        messages.append(f"Gold introuvable : {fichier_gold}")
    else:
        tous = charger_enregistrements(fichier_gold)
        excluent = [
            e for e in tous if cle_document(e.expected_document or "") == cle_document(DOCUMENT_MANQUANT_CONNU)
        ]
        evaluables = len(tous) - len(excluent)
        messages.append(
            f"Gold : {len(tous)} questions, {len(excluent)} exclue(s) "
            f"(document manquant {DOCUMENT_MANQUANT_CONNU}), {evaluables} évaluables."
        )
        if evaluables != 239:
            messages.append(
                f"ATTENTION : {evaluables} évaluables, 239 attendues d'après l'audit précédent."
            )
        ids_excluent = {e.id for e in excluent}
        if ids_excluent != {"cquae:test:8298"}:
            messages.append(
                f"ATTENTION : exclusion(s) inattendue(s) — {ids_excluent} au lieu de "
                "{'cquae:test:8298'}."
            )

    # --- Cas smoke : références gold résolvables ----------------------------
    if fichier_gold.exists():
        index_gold = {e.id: e for e in charger_enregistrements(fichier_gold)}
        for c in cas:
            for qid in c.gold_qids:
                if qid not in index_gold:
                    ok = False
                    messages.append(f"{c.test_id} : gold_qid {qid!r} introuvable dans le gold.")
            if c.champs:
                for champ in c.champs:
                    qid = champ.get("gold_qid")
                    if qid and qid not in index_gold:
                        ok = False
                        messages.append(
                            f"{c.test_id} : champ {champ['label']!r} référence "
                            f"gold_qid {qid!r} introuvable."
                        )

    messages.append(f"Cas smoke chargés : {len(cas)} (28 attendus).")
    if len(cas) != 28:
        messages.append("ATTENTION : nombre de cas différent de 28.")

    return ok, messages


# ===========================================================================
# 4. Exécution — FRONTIÈRE anti-fuite gold
# ===========================================================================


def executer_cas(cas: CasSmoke) -> dict[str, Any]:
    """
    Exécute le graphe agentique sur `cas.query` UNIQUEMENT.

    Réutilise exactement le même point d'entrée que `evaluate_agent.py`
    (`construire_graphe`, `construire_session`, `EtatGraphe`) — aucune
    réimplémentation du graphe. Ne lit ni `cas.gold_qids`, ni `cas.champs`,
    ni `cas.source_document`, ni aucun champ autre que `cas.query` : c'est la
    garantie structurelle vérifiée par
    `tests/evaluation/test_cquae_multicapacite.py::test_aucune_fuite_gold`.
    """
    from src.agent.graph import construire_graphe
    from src.agent.graph_state import EtatGraphe
    from src.agent.session import construire_session

    graphe = construire_graphe()
    debut = time.perf_counter()

    try:
        session = construire_session(cas.query)
        limite = max(25, session.etat.max_tentatives * 3 + 5)
        resultat = graphe.invoke(
            EtatGraphe(session=session), config={"recursion_limit": limite}
        )
        return {
            "session": resultat["session"],
            "sortie": resultat["reponse"],
            "duree_secondes": round(time.perf_counter() - debut, 4),
            "erreur": None,
        }
    except Exception as exc:  # noqa: BLE001 — un cas en échec ne doit jamais arrêter le smoke run
        logger.exception("Erreur d'exécution sur %s", cas.test_id)
        return {
            "session": None,
            "sortie": None,
            "duree_secondes": round(time.perf_counter() - debut, 4),
            "erreur": f"{type(exc).__name__}: {exc}",
        }


# ===========================================================================
# 5. Utilitaires de scoring partagés
# ===========================================================================


def _intention_detectee(session: Any) -> tuple[str | None, bool]:
    for etape in session.etat.trace:
        if etape.nom == "intention":
            return etape.donnees.get("intention"), bool(etape.donnees.get("desambiguisation_llm"))
    return None, False


def _trace_outil(session: Any, nom: str) -> dict[str, Any]:
    for etape in session.etat.trace:
        if etape.nom == nom:
            return dict(etape.donnees)
    return {}


def _jaccard(a: str, b: str) -> float:
    ta, tb = ensemble_jetons(a), ensemble_jetons(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _recouvrement_gold(gold_texte: str, valeur: str) -> float:
    """
    RAPPEL (recall) : part des jetons du texte gold retrouvés dans `valeur`.

    Adapté à SUMMARIZE : `valeur` y est un résumé LONG censé couvrir un fait
    gold court — on veut savoir si ce fait y est repris.
    """
    gold_jetons = ensemble_jetons(gold_texte)
    if not gold_jetons:
        return 0.0
    return len(gold_jetons & ensemble_jetons(valeur)) / len(gold_jetons)


def _precision_valeur(valeur: str, reference: str) -> float:
    """
    PRÉCISION : part des jetons de `valeur` retrouvés dans `reference`.

    Adapté à EXTRACT : `valeur` y est une chaîne COURTE et précise (ex. « 20
    septembre 1792 ») comparée à une référence gold potentiellement longue
    (réponse + evidence). Le rappel serait trompeur ici : une valeur exacte
    mais courte aurait mécaniquement un faible recouvrement face à un texte
    de référence volumineux. La précision — la valeur est-elle intégralement
    retrouvée dans la référence ? — est le bon sens de comparaison.
    """
    valeur_jetons = ensemble_jetons(valeur)
    if not valeur_jetons:
        return 0.0
    return len(valeur_jetons & ensemble_jetons(reference)) / len(valeur_jetons)


def _reponse_couvre_gold(reponse: str, gold: Enregistrement) -> bool:
    """
    Vrai si `reponse` contient bien l'information de la réponse gold.

    Deux voies, dans l'ordre :

    1. `test_rag.comparer_reponse` (inchangé) — suffisant pour un gold court,
       canonique ou numérique et pour les `answer_variants`. Une réponse
       rédigée qui reprend la formule gold quasi mot à mot y passe déjà.

    2. Voie tolérante à la paraphrase, nécessaire quand le gold est une
       PHRASE reformulée (« La bataille de Valmy s'est déroulée le 20
       septembre 1792. ») là où l'agent répond « … a eu lieu le 20 septembre
       1792 … ». On isole les *jetons porteurs de la réponse* — ceux du gold
       présents verbatim dans l'evidence et absents de la question, donc ni
       reformulation propre au gold ni simple écho de la question — et on
       exige qu'ils soient majoritairement présents dans la réponse. Tout
       jeton purement numérique de cet ensemble (date, quantité) doit être
       présent : un millésime faux dans la bonne phrase n'est pas correct.

    Aucune liste de réponses codée en dur : seuls `gold.expected_answer`,
    `gold.evidence_text` et `gold.question` sont utilisés.
    """
    from test_rag import TOLERANCE_RELATIVE, comparer_reponse

    for candidat in (gold.expected_answer, *gold.answer_variants):
        if candidat and comparer_reponse(reponse, candidat, tolerance=TOLERANCE_RELATIVE)[0]:
            return True

    jetons_gold = ensemble_jetons(gold.expected_answer)
    jetons_question = ensemble_jetons(gold.question)
    porteurs = (jetons_gold & ensemble_jetons(gold.evidence_text)) - jetons_question
    if not porteurs:
        # Pas d'evidence exploitable : repli sur les jetons du gold absents de
        # la question (l'information que la réponse ajoute à la question).
        porteurs = jetons_gold - jetons_question
    if not porteurs:
        return False

    jetons_reponse = ensemble_jetons(reponse)
    chiffres = {jeton for jeton in porteurs if jeton.isdigit()}
    if chiffres and not chiffres <= jetons_reponse:
        return False
    return len(porteurs & jetons_reponse) / len(porteurs) >= SEUIL_RECOUVREMENT_SEMANTIQUE


def _resultat_routing_failure(cas: CasSmoke, brut: dict[str, Any], detected: str | None, desambiguisation: bool) -> ResultatCas:
    """
    Construit un `ResultatCas` de routage incorrect SANS supposer la forme de
    `brut["sortie"]`.

    Un mauvais routage signifie que l'outil réellement exécuté N'EST PAS
    celui attendu (ex. SU-02 routé vers `search` au lieu de `summarize`) :
    `brut["sortie"]` est alors potentiellement un `ReponseRAG` (SEARCH) là où
    un `ResultatOutil` (SUMMARIZE/CLASSIFY/EXTRACT) était attendu, ou
    inversement — deux formes structurellement différentes (voir docstring de
    module). Accéder à un attribut spécifique à l'une des deux formes ferait
    planter le scoring exactement dans le cas qu'il doit mesurer. Seul
    `getattr(..., défaut)` est utilisé ici.
    """
    session = brut["session"]
    sortie = brut["sortie"]
    texte = getattr(sortie, "reponse", None)
    if texte is None:
        donnees = getattr(sortie, "donnees", None)
        texte = donnees.get("resume") if isinstance(donnees, dict) else None
    return ResultatCas(
        test_id=cas.test_id,
        capability=cas.capability,
        query=cas.query,
        native_gold=cas.native_gold,
        expected_tool=cas.expected_tool,
        detected_tool=detected,
        tools_executed=session.outils_utilises(),
        source_document=cas.source_document,
        agent_success=bool(getattr(sortie, "succes", getattr(sortie, "contexte_suffisant", False))),
        agent_result=texte,
        retrieved_sources=[
            getattr(s, "nom_fichier", None) or getattr(s, "document", "") for s in getattr(sortie, "sources", [])
        ],
        citations=[],
        routing_status="ROUTING_FAILURE",
        retrieval_status="INCONNU",
        answer_status="INCONNU",
        provenance_status="INCONNU",
        final_verdict="WRONG",
        failure_category="ROUTING_FAILURE",
        latency_seconds=brut["duree_secondes"],
        error="",
        details={
            "desambiguisation_llm": desambiguisation,
            "note": f"Attendu={cas.expected_tool!r}, détecté={detected!r} — sortie non interprétable dans le schéma attendu.",
        },
    )


def _associer_champ(label_attendu: str, donnees_extract: dict[str, Any]) -> str | None:
    """Apparie un label attendu au champ réellement renvoyé par EXTRACT (paraphrasé par le LLM)."""
    candidats = [
        (cle, _jaccard(label_attendu, cle))
        for cle in donnees_extract
        if isinstance(donnees_extract.get(cle), dict)
    ]
    candidats = [c for c in candidats if c[1] > 0]
    if not candidats:
        return None
    candidats.sort(key=lambda c: -c[1])
    return candidats[0][0]


# ===========================================================================
# 6A. Scoring — SEARCH
# ===========================================================================


def noter_search(cas: CasSmoke, brut: dict[str, Any], gold_index: dict[str, Enregistrement]) -> ResultatCas:
    erreur = brut["erreur"]
    if erreur:
        return _resultat_technique(cas, brut, erreur)

    session = brut["session"]
    detected, desambiguisation = _intention_detectee(session)
    if detected != "search":
        return _resultat_routing_failure(cas, brut, detected, desambiguisation)

    reponse = brut["sortie"]  # ReponseRAG — la vérification ci-dessus garantit ce type
    routing_status = "OK"

    gold = gold_index.get(cas.gold_qids[0]) if cas.gold_qids else None

    sources_citees = [s.nom_fichier or s.document or s.source for s in reponse.sources]
    doc_attendu_cite = (
        bool(gold and gold.expected_document)
        and any(cle_document(s) == cle_document(gold.expected_document) for s in sources_citees)
    )

    couverture = None
    if gold is not None and gold.evidence_text and reponse.recherche is not None:
        couverture = mesurer_couverture(gold.evidence_text, reponse.recherche.passages)

    exactitude = None
    if gold is not None and gold.expected_answer and reponse.reponse:
        exactitude = _reponse_couvre_gold(reponse.reponse, gold)

    from evaluation.evaluate_end_to_end import calculer_groundedness

    groundedness = calculer_groundedness(reponse) if reponse.sources else 0.0
    refus = (not reponse.contexte_suffisant) or (not reponse.sources)

    retrieval_status = "OK" if doc_attendu_cite else "FAIL"
    answer_status = "OK" if exactitude else ("FAIL" if exactitude is not None else "INCONNU")
    provenance_status = "OK" if (reponse.citations_valides and groundedness >= 0.5) else "FAIL"

    failure_category = None
    if refus:
        verdict = "WRONG"
        failure_category = (
            "DOCUMENT_RETRIEVAL_FAILURE" if not doc_attendu_cite else "EVIDENCE_RETRIEVAL_FAILURE"
        )
    elif exactitude and doc_attendu_cite and provenance_status == "OK":
        verdict = "PASS"
    elif exactitude and (not doc_attendu_cite or provenance_status != "OK"):
        verdict = "ANSWER_ONLY"
        failure_category = "PROVENANCE_FAILURE"
    elif not exactitude and doc_attendu_cite and (couverture is None or couverture.couverture_relative >= 0.3):
        verdict = "RETRIEVAL_ONLY"
        failure_category = "GENERATION_FAILURE"
    elif not exactitude and groundedness < 0.3:
        verdict = "UNSUPPORTED"
        failure_category = "PROVENANCE_FAILURE"
    else:
        verdict = "WRONG"
        failure_category = "GENERATION_FAILURE"

    return ResultatCas(
        test_id=cas.test_id,
        capability=cas.capability,
        query=cas.query,
        native_gold=cas.native_gold,
        expected_tool=cas.expected_tool,
        detected_tool=detected,
        tools_executed=session.outils_utilises(),
        source_document=cas.source_document,
        agent_success=not refus,
        agent_result=reponse.reponse,
        retrieved_sources=sources_citees,
        citations=[s.citation for s in reponse.sources],
        routing_status=routing_status,
        retrieval_status=retrieval_status,
        answer_status=answer_status,
        provenance_status=provenance_status,
        final_verdict=verdict,
        failure_category=failure_category,
        latency_seconds=brut["duree_secondes"],
        error="",
        details={
            "desambiguisation_llm": desambiguisation,
            "exactitude": exactitude,
            "groundedness": round(groundedness, 4),
            "couverture_relative": (
                round(couverture.couverture_relative, 4) if couverture is not None else None
            ),
            "refus": refus,
        },
    )


# ===========================================================================
# 6B. Scoring — EXTRACT
# ===========================================================================


def noter_extract(cas: CasSmoke, brut: dict[str, Any], gold_index: dict[str, Enregistrement]) -> ResultatCas:
    erreur = brut["erreur"]
    if erreur:
        return _resultat_technique(cas, brut, erreur)

    session = brut["session"]
    detected, desambiguisation = _intention_detectee(session)
    if detected != "extract":
        return _resultat_routing_failure(cas, brut, detected, desambiguisation)

    resultat = brut["sortie"]  # ResultatOutil — garanti par la vérification ci-dessus
    routing_status = "OK"

    sources = resultat.sources if resultat.succes else []
    sources_ok = (
        all(cle_document(s.nom_fichier) == cle_document(cas.source_document) for s in sources)
        if cas.source_document
        else True
    )

    details: dict[str, Any] = {}
    tout_ok = resultat.succes and sources_ok
    au_moins_un_desaccord_trouve = False

    for champ in cas.champs or []:
        cle = _associer_champ(champ["label"], resultat.donnees) if resultat.succes else None
        entree = resultat.donnees.get(cle, {}) if cle else {}
        trouve = bool(entree.get("trouve"))
        detail: dict[str, Any] = {
            "champ_retourne": cle,
            "trouve": trouve,
            "trouve_attendu": champ["trouve_attendu"],
        }
        if trouve != champ["trouve_attendu"]:
            tout_ok = False
            au_moins_un_desaccord_trouve = True
        elif champ["trouve_attendu"] and champ.get("gold_qid"):
            gold = gold_index.get(champ["gold_qid"])
            valeur = entree.get("valeur") or ""
            reference = f"{gold.expected_answer} {gold.evidence_text}" if gold else ""
            precision = _precision_valeur(valeur, reference)
            detail["precision_valeur"] = round(precision, 3)
            if precision < SEUIL_RECOUVREMENT_VALEUR:
                tout_ok = False
        details[champ["label"]] = detail

    if not sources_ok:
        details["provenance"] = "sources hors du document demandé"

    if not resultat.succes:
        verdict, failure_category = "WRONG", "EXTRACTION_FAILURE"
    elif tout_ok:
        verdict, failure_category = "PASS", None
    elif au_moins_un_desaccord_trouve:
        verdict, failure_category = "WRONG", "EXTRACTION_FAILURE"
    else:
        verdict, failure_category = "UNSUPPORTED", "PROVENANCE_FAILURE"

    return ResultatCas(
        test_id=cas.test_id,
        capability=cas.capability,
        query=cas.query,
        native_gold=cas.native_gold,
        expected_tool=cas.expected_tool,
        detected_tool=detected,
        tools_executed=session.outils_utilises(),
        source_document=cas.source_document,
        agent_success=resultat.succes,
        agent_result=resultat.donnees,
        retrieved_sources=[s.nom_fichier for s in sources],
        citations=[s.doc_id for s in sources],
        routing_status=routing_status,
        retrieval_status="OK" if sources_ok else "FAIL",
        answer_status="OK" if tout_ok else "FAIL",
        provenance_status="OK" if sources_ok else "FAIL",
        final_verdict=verdict,
        failure_category=failure_category,
        latency_seconds=brut["duree_secondes"],
        error="" if resultat.succes else resultat.message,
        details={"desambiguisation_llm": desambiguisation, "champs": details},
    )


# ===========================================================================
# 6C. Scoring — CLASSIFY (contract_validation, PAS de catégorie gold)
# ===========================================================================


def noter_classify(
    cas: CasSmoke, brut: dict[str, Any], categories_taxonomie: list[str]
) -> ResultatCas:
    erreur = brut["erreur"]
    if erreur:
        return _resultat_technique(cas, brut, erreur)

    session = brut["session"]
    detected, desambiguisation = _intention_detectee(session)
    if detected != "classify":
        return _resultat_routing_failure(cas, brut, detected, desambiguisation)

    resultat = brut["sortie"]  # ResultatOutil — garanti par la vérification ci-dessus
    routing_status = "OK"
    trace = _trace_outil(session, "classify")
    mode = trace.get("mode")
    outils = session.outils_utilises()
    aucun_search_interne = "search" not in outils

    details: dict[str, Any] = {"mode": mode, "outils_executes": outils}

    if cas.test_id == "CL-03":
        # Document confirmé absent de l'index : succès attendu = refus explicite,
        # jamais une classification produite sur un contenu de repli.
        if not resultat.succes and mode == "document_vise_non_resolu":
            verdict, failure_category = "ABSTAIN_CORRECT", "MISSING_DOCUMENT_EXCLUDED"
            retrieval_status = provenance_status = answer_status = "OK"
        elif resultat.succes and mode == "contexte_existant":
            # Repli silencieux sur un contenu sans rapport avec le document demandé —
            # exactement le risque documenté dans src/agent/nodes.py (statut="aucun" ambigu).
            verdict, failure_category = "WRONG", "CLASSIFICATION_FAILURE"
            retrieval_status = "FAIL"
            provenance_status = answer_status = "FAIL"
            details["alerte"] = (
                "Le document demandé est absent de l'index mais l'agent a "
                "produit une classification via le mode contextuel (repli "
                "sur des sources sans rapport)."
            )
        else:
            verdict, failure_category = "TECHNICAL_ERROR", "TECHNICAL_FAILURE"
            retrieval_status = provenance_status = answer_status = "INCONNU"

    else:
        doc_demande_ok = cle_document(trace.get("document_demande") or "") == cle_document(
            cas.source_document or ""
        )
        categorie = resultat.donnees.get("categorie") if resultat.succes else None
        categorie_ok = categorie is None or categorie in categories_taxonomie
        citations = resultat.donnees.get("citations") or [] if resultat.succes else []
        sources = resultat.sources if resultat.succes else []
        provenance_ok = True
        if categorie is not None:
            provenance_ok = bool(citations) and bool(sources) and all(
                cle_document(s.nom_fichier) == cle_document(cas.source_document or "") for s in sources
            )

        details.update(
            {
                "document_demande_ok": doc_demande_ok,
                "aucun_search_interne": aucun_search_interne,
                "categorie": categorie,
                "categorie_dans_taxonomie_ou_none": categorie_ok,
                "provenance_ok": provenance_ok,
            }
        )

        retrieval_status = "OK" if (mode == "document_complet" and doc_demande_ok) else "FAIL"
        provenance_status = "OK" if provenance_ok else "FAIL"
        answer_status = "OK" if categorie_ok else "FAIL"

        if mode != "document_complet" or not doc_demande_ok:
            verdict, failure_category = "WRONG", "DOCUMENT_RETRIEVAL_FAILURE"
        elif not aucun_search_interne:
            verdict, failure_category = "WRONG", "CLASSIFICATION_FAILURE"
        elif not categorie_ok:
            verdict, failure_category = "WRONG", "CLASSIFICATION_FAILURE"
        elif not provenance_ok:
            verdict, failure_category = "UNSUPPORTED", "PROVENANCE_FAILURE"
        elif categorie is None:
            verdict, failure_category = "ABSTAIN_CORRECT", "CORRECT_ABSTENTION"
        else:
            verdict, failure_category = "PASS", None

    return ResultatCas(
        test_id=cas.test_id,
        capability=cas.capability,
        query=cas.query,
        native_gold=cas.native_gold,
        expected_tool=cas.expected_tool,
        detected_tool=detected,
        tools_executed=outils,
        source_document=cas.source_document,
        agent_success=resultat.succes,
        agent_result=resultat.donnees if resultat.succes else resultat.message,
        retrieved_sources=[s.nom_fichier for s in (resultat.sources if resultat.succes else [])],
        citations=list(resultat.donnees.get("citations") or []) if resultat.succes else [],
        routing_status=routing_status,
        retrieval_status=retrieval_status,
        answer_status=answer_status,
        provenance_status=provenance_status,
        final_verdict=verdict,
        failure_category=failure_category,
        latency_seconds=brut["duree_secondes"],
        error="" if resultat.succes else resultat.message,
        details={**details, "desambiguisation_llm": desambiguisation, "evaluation_type": "contract_validation"},
    )


# ===========================================================================
# 6D. Scoring — SUMMARIZE (structurel déterministe + checkpoints factuels)
# ===========================================================================


def noter_summarize(
    cas: CasSmoke,
    brut: dict[str, Any],
    gold_index: dict[str, Enregistrement],
    *,
    llm_judge: bool = False,
) -> ResultatCas:
    erreur = brut["erreur"]
    if erreur:
        return _resultat_technique(cas, brut, erreur)

    session = brut["session"]
    detected, desambiguisation = _intention_detectee(session)
    if detected != "summarize":
        # C'est exactement le cas SU-02 attendu (routage lexical probable
        # vers 'search') : on le mesure explicitement, sans planter sur la
        # forme ReponseRAG qui revient dans ce cas au lieu d'un ResultatOutil.
        return _resultat_routing_failure(cas, brut, detected, desambiguisation)

    resultat = brut["sortie"]  # ResultatOutil — garanti par la vérification ci-dessus
    routing_status = "OK"
    trace = _trace_outil(session, "summarize")
    outils = session.outils_utilises()

    # --- A. Validation structurelle (déterministe) --------------------------
    document_demande = trace.get("documents_demandes") or []
    doc_ok = bool(document_demande) and any(
        cle_document(d) == cle_document(cas.source_document or "") for d in document_demande
    )
    aucun_search_interne = "search" not in outils
    sources = resultat.sources if resultat.succes else []
    sources_ok = bool(sources) and all(
        cle_document(s.nom_fichier) == cle_document(cas.source_document or "") for s in sources
    )

    structural_ok = resultat.succes and doc_ok and aucun_search_interne and sources_ok

    # --- B. Fidélité factuelle (secondaire, jamais pénalisée pour omission) -
    resume = resultat.donnees.get("resume", "") if resultat.succes else ""
    checkpoints: list[dict[str, Any]] = []
    for qid in cas.gold_qids:
        gold = gold_index.get(qid)
        if gold is None:
            continue
        recouvrement = _recouvrement_gold(gold.expected_answer, resume)
        checkpoints.append(
            {
                "gold_qid": qid,
                "fait_touche": recouvrement >= SEUIL_RECOUVREMENT_VALEUR,
                "recouvrement": round(recouvrement, 3),
            }
        )

    # Heuristique déterministe, volontairement étroite : ne signale une
    # contradiction potentielle QUE sur un désaccord numérique explicite
    # (un nombre du checkpoint absent du résumé alors que le résumé cite un
    # AUTRE nombre dans un contexte proche) — jamais une simple omission.
    contradiction_potentielle = False
    if resume:
        jetons_resume_nombres = {t for t in ensemble_jetons(resume) if t.isdigit()}
        for qid in cas.gold_qids:
            gold = gold_index.get(qid)
            if gold is None:
                continue
            jetons_gold_nombres = {t for t in ensemble_jetons(gold.expected_answer) if t.isdigit()}
            if jetons_gold_nombres and jetons_resume_nombres and not (
                jetons_gold_nombres & jetons_resume_nombres
            ) and jetons_resume_nombres:
                # Le résumé avance un ou plusieurs nombres dans une réponse
                # censée couvrir ce fait, sans qu'aucun ne corresponde au
                # nombre gold : signal faible, à revue humaine — jamais
                # décisif seul.
                contradiction_potentielle = True

    judge_secondaire = None
    if llm_judge and resume and cas.gold_qids:
        judge_secondaire = _juger_contradiction_llm(resume, cas.gold_qids, gold_index)

    if not resultat.succes:
        verdict, failure_category = "WRONG", "PROVENANCE_FAILURE"
    elif not structural_ok:
        verdict, failure_category = "WRONG", (
            "DOCUMENT_RETRIEVAL_FAILURE" if not doc_ok else "PROVENANCE_FAILURE"
        )
    elif contradiction_potentielle or (judge_secondaire and judge_secondaire.get("contradiction")):
        verdict, failure_category = "UNSUPPORTED", "SUMMARIZATION_FAITHFULNESS_FAILURE"
    else:
        verdict, failure_category = "PASS", None

    return ResultatCas(
        test_id=cas.test_id,
        capability=cas.capability,
        query=cas.query,
        native_gold=cas.native_gold,
        expected_tool=cas.expected_tool,
        detected_tool=detected,
        tools_executed=outils,
        source_document=cas.source_document,
        agent_success=resultat.succes,
        agent_result=resume,
        retrieved_sources=[s.nom_fichier for s in sources],
        citations=[s.doc_id for s in sources],
        routing_status=routing_status,
        retrieval_status="OK" if doc_ok else "FAIL",
        answer_status="OK" if not contradiction_potentielle else "FAIL",
        provenance_status="OK" if sources_ok else "FAIL",
        final_verdict=verdict,
        failure_category=failure_category,
        latency_seconds=brut["duree_secondes"],
        error="" if resultat.succes else resultat.message,
        details={
            "desambiguisation_llm": desambiguisation,
            "structural_ok": structural_ok,
            "aucun_search_interne": aucun_search_interne,
            "checkpoints_factuels": checkpoints,
            "contradiction_potentielle_heuristique": contradiction_potentielle,
            "evaluation_type": "structural_and_faithfulness",
        },
        judge_secondaire=judge_secondaire,
    )


def _juger_contradiction_llm(
    resume: str, gold_qids: list[str], gold_index: dict[str, Enregistrement]
) -> dict[str, Any]:
    """
    Jugement LLM SECONDAIRE, optionnel (`--llm-judge`), jamais décisionnaire
    seul (mission, point 3B). Le gold n'est montré qu'APRÈS génération du
    résumé — jamais avant ou pendant — donc aucune fuite vers l'agent : ce
    jugement a lieu hors du chemin agent, en scoring pur.
    """
    from src.llm.common import extraire_json_objet, invoquer_llm
    from src.llm.factory import construire_llm

    faits = "\n".join(
        f"- {gold_index[q].expected_answer}" for q in gold_qids if q in gold_index
    )
    systeme = (
        "Tu compares un résumé à une liste de faits vérifiés. Réponds "
        'uniquement {"contradiction": true|false, "raison": "..."}. '
        "\"contradiction\"=true UNIQUEMENT si le résumé affirme explicitement "
        "quelque chose qui contredit un fait listé. L'absence d'un fait dans "
        "le résumé n'est JAMAIS une contradiction."
    )
    utilisateur = f"FAITS VÉRIFIÉS\n{faits}\n\nRÉSUMÉ\n{resume}"
    try:
        llm = construire_llm()
        texte = invoquer_llm(llm, systeme=systeme, utilisateur=utilisateur)
        objet = extraire_json_objet(texte)
        return {
            "contradiction": bool(objet.get("contradiction")),
            "raison": str(objet.get("raison", "")),
            "input": utilisateur,
            "output_brut": texte,
        }
    except Exception as exc:  # noqa: BLE001 — jugement secondaire, jamais bloquant
        return {"contradiction": False, "raison": f"juge indisponible : {exc}", "input": utilisateur}


# ===========================================================================
# 6E. Scoring — Anti-hallucination (AH-01)
# ===========================================================================


def noter_anti_hallucination(cas: CasSmoke, brut: dict[str, Any]) -> ResultatCas:
    """
    AH-01 : question hors-corpus, vise le grounding, pas l'exactitude du fait.

    ABSTAIN_CORRECT si refus faute de preuve ; UNSUPPORTED si une réponse est
    produite sans preuve issue du corpus (même correcte) ; jamais PASS.
    """
    erreur = brut["erreur"]
    if erreur:
        return _resultat_technique(cas, brut, erreur)

    session = brut["session"]
    detected, desambiguisation = _intention_detectee(session)
    if detected != "search":
        return _resultat_routing_failure(cas, brut, detected, desambiguisation)

    reponse = brut["sortie"]  # ReponseRAG — garanti par la vérification ci-dessus
    refus = (not reponse.contexte_suffisant) or (not reponse.sources)

    if refus:
        verdict, failure_category = "ABSTAIN_CORRECT", "CORRECT_ABSTENTION"
    else:
        # Une réponse a été produite : par construction (fait absent du
        # corpus CQuAE), elle ne peut pas être authentiquement fondée sur des
        # preuves pertinentes — quelle que soit sa justesse factuelle réelle.
        verdict, failure_category = "UNSUPPORTED", "PROVENANCE_FAILURE"

    return ResultatCas(
        test_id=cas.test_id,
        capability=cas.capability,
        query=cas.query,
        native_gold=cas.native_gold,
        expected_tool=cas.expected_tool,
        detected_tool=detected,
        tools_executed=session.outils_utilises(),
        source_document=cas.source_document,
        agent_success=not refus,
        agent_result=reponse.reponse,
        retrieved_sources=[s.nom_fichier for s in reponse.sources],
        citations=[s.citation for s in reponse.sources],
        routing_status="OK" if detected == "search" else "ROUTING_FAILURE",
        retrieval_status="N/A",
        answer_status="N/A",
        provenance_status="OK" if refus else "FAIL",
        final_verdict=verdict,
        failure_category=failure_category,
        latency_seconds=brut["duree_secondes"],
        error="",
        details={"desambiguisation_llm": desambiguisation, "refus": refus},
    )


def _resultat_technique(cas: CasSmoke, brut: dict[str, Any], erreur: str) -> ResultatCas:
    return ResultatCas(
        test_id=cas.test_id,
        capability=cas.capability,
        query=cas.query,
        native_gold=cas.native_gold,
        expected_tool=cas.expected_tool,
        detected_tool=None,
        tools_executed=[],
        source_document=cas.source_document,
        agent_success=False,
        agent_result=None,
        retrieved_sources=[],
        citations=[],
        routing_status="INCONNU",
        retrieval_status="INCONNU",
        answer_status="INCONNU",
        provenance_status="INCONNU",
        final_verdict="TECHNICAL_ERROR",
        failure_category="TECHNICAL_FAILURE",
        latency_seconds=brut["duree_secondes"],
        error=erreur,
        details={},
    )


# ===========================================================================
# 7. Orchestration d'un cas
# ===========================================================================


def noter_cas(
    cas: CasSmoke,
    brut: dict[str, Any],
    gold_index: dict[str, Enregistrement],
    categories_taxonomie: list[str],
    *,
    llm_judge: bool = False,
) -> ResultatCas:
    if cas.capability == "SEARCH":
        return noter_search(cas, brut, gold_index)
    if cas.capability == "EXTRACT":
        return noter_extract(cas, brut, gold_index)
    if cas.capability == "CLASSIFY":
        return noter_classify(cas, brut, categories_taxonomie)
    if cas.capability == "SUMMARIZE":
        return noter_summarize(cas, brut, gold_index, llm_judge=llm_judge)
    if cas.capability == "ANTI_HALLUCINATION":
        return noter_anti_hallucination(cas, brut)
    raise ValueError(f"Capability inconnue : {cas.capability!r}")


# ===========================================================================
# 8. CLI
# ===========================================================================


def construire_parseur() -> argparse.ArgumentParser:
    parseur = argparse.ArgumentParser(
        description="Smoke benchmark E2E multi-capacités CQuAE (SEARCH/SUMMARIZE/CLASSIFY/EXTRACT)."
    )
    parseur.add_argument("--cas", type=Path, default=FICHIER_CAS_PAR_DEFAUT)
    parseur.add_argument("--gold", type=Path, default=FICHIER_GOLD_PAR_DEFAUT)
    parseur.add_argument("--sortie", type=Path, default=DOSSIER_RAPPORTS / "cquae_multicapacite")
    parseur.add_argument("--nom", default=None)
    parseur.add_argument(
        "--executer",
        action="store_true",
        help="Sans cet argument : préconditions vérifiées et affichées, AUCUN appel agent.",
    )
    parseur.add_argument(
        "--llm-judge",
        action="store_true",
        help="Active le juge LLM secondaire pour SUMMARIZE (jamais décisionnaire seul).",
    )
    parseur.add_argument("--limite", type=int, default=None, help="tronque les cas (smoke test du smoke test)")
    parseur.add_argument("--verbose", action="store_true")
    return parseur


def main(argv: list[str] | None = None) -> int:
    args = construire_parseur().parse_args(argv)
    configurer_logs(args.verbose)

    cas = charger_cas(args.cas)
    if args.limite is not None:
        cas = cas[: args.limite]

    print("=" * 78)
    print("PRÉCONDITIONS — vérification (aucun appel agent à ce stade)")
    print("=" * 78)
    ok, messages = verifier_preconditions(cas, args.gold)
    for m in messages:
        print(f"  {m}")

    if not ok:
        print("\nSTOP — préconditions non satisfaites. Aucun test ne sera lancé.")
        return 1

    if not args.executer:
        print(
            "\nPréconditions satisfaites. Rien n'a été exécuté "
            "(passer --executer pour lancer les cas contre l'agent)."
        )
        return 0

    from evaluation.common import attendre_client_qdrant
    from src.config import get_profil
    from src.rag.vectorstore import fermer_client

    attendre_client_qdrant()
    gold_index = {e.id: e for e in charger_enregistrements(args.gold)}
    categories_taxonomie = get_profil().classification.noms()

    resultats: list[ResultatCas] = []
    try:
        for i, c in enumerate(cas, start=1):
            logger.info("[%d/%d] %s (%s)", i, len(cas), c.test_id, c.capability)
            brut = executer_cas(c)
            resultats.append(
                noter_cas(c, brut, gold_index, categories_taxonomie, llm_judge=args.llm_judge)
            )
    finally:
        fermer_client()

    resume = {
        "total": len(resultats),
        "par_verdict": {v: sum(1 for r in resultats if r.final_verdict == v) for v in VERDICTS},
        "par_capability": sorted({r.capability for r in resultats}),
    }
    nom = args.nom or f"cquae_multicapacite_{horodatage()}"
    from evaluation.common import ecrire_rapport

    ecrire_rapport(args.sortie, nom, resume=resume, details=[r.vers_dict() for r in resultats])

    print("\nRésumé :", json.dumps(resume, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
