"""
Capacité COMPARE (étape P1.5).

Compare explicitement 2 à 4 documents ciblés par l'utilisateur — ou, depuis
P1.9 (`portee="intra"`), plusieurs éléments au sein d'UN même document
(sections, périodes, options…). MAP par document (via `multidoc_pipeline`) -> REDUCE inter-document ->
`ResultatOutil` avec provenance par document.

Ne réalise AUCUN search global : le REDUCE ne voit que les sorties MAP
validées, jamais le corpus. Les désaccords entre documents sont conservés
explicitement.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

from src.llm.common import bloc_profil_domaine, extraire_json_objet, invoquer_llm
from src.tools.base import ResultatOutil, SourceOutil

from src.agent.multidoc import PORTEE_INTER, PORTEE_INTRA
from src.agent.multidoc_pipeline import (
    bloc_maps_pour_reduce,
    budget_caracteres_entree_llm,
    citations_autorisees,
    diagnostic_maps,
    executer_maps,
    resoudre_cibles,
    retirer_citations_invalides,
    sources_par_citation,
    valider_citations,
)

logger = logging.getLogger(__name__)

_OUTIL = "compare"

#: `donnees["statut"]` — REDUCE MINIMAL pour réduire les refus globaux (audit
#: long-documents, section D) : "complet" si TOUS les documents résolus ont
#: apporté des éléments exploitables, "partiel" si au moins un document a
#: contribué mais qu'au moins un autre n'a rien apporté (sans_evidence ou
#: échec). Le refus dur (`ResultatOutil.echec`) reste réservé à ZÉRO document
#: exploitable — jamais de conclusion comparative fabriquée pour un document
#: sans preuve, jamais de provenance inventée : la validation par citation
#: ci-dessous (inchangée) s'applique identiquement aux deux statuts.
STATUT_COMPLET = "complet"
STATUT_PARTIEL = "partiel"


@dataclass
class ResultatCompare:
    question: str
    documents: list[str]
    points_communs: list[str] = field(default_factory=list)
    differences: list[str] = field(default_factory=list)
    positions_par_document: dict[str, str] = field(default_factory=dict)
    contradictions: list[str] = field(default_factory=list)
    conclusion: str | None = None
    documents_sans_evidence: list[str] = field(default_factory=list)
    documents_en_echec: list[str] = field(default_factory=list)
    portee: str = PORTEE_INTER


_SYSTEME_REDUCE = """Tu construis une COMPARAISON entre plusieurs documents à partir d'analyses déjà réalisées, une par document.

RÈGLES ABSOLUES
- Utilise UNIQUEMENT les analyses par document fournies ci-dessous. Tu n'as pas accès aux documents complets ni à aucune autre source.
- N'invente aucun fait, chiffre, position ou conclusion.
- Chaque point (commun, différence, position, contradiction, conclusion) DOIT porter au moins une citation [D_S_] reprise des analyses.
- Ne fusionne jamais deux documents dans une même affirmation si une seule analyse la soutient : attribue-la au bon document.
- Si les documents divergent ou se contredisent, conserve la divergence EXPLICITEMENT dans "contradictions" ou "differences" — ne la lisse pas.
- Si un document n'apporte aucune information pertinente, ne fabrique rien pour lui.
- Si UN SEUL document apporte des éléments exploitables (les autres n'apportant rien), ne fabrique AUCUNE comparaison : laisse "points_communs", "differences" et "contradictions" VIDES, et utilise uniquement "positions_par_document" pour décrire ce que ce document apporte, avec ses citations.
- La "conclusion" est facultative : ne la fournis que si les analyses la soutiennent réellement, sinon mets null.

Réponds UNIQUEMENT avec un objet JSON strict :
{
  "points_communs": ["... [D1S2][D2S1]", ...],
  "differences": ["... [D1S3]", ...],
  "positions_par_document": {"<libellé doc>": "... [D1S1]"},
  "contradictions": ["... [D1S2] vs [D2S4]", ...],
  "conclusion": "... [D1S1][D2S2]" | null
}"""


_SYSTEME_REDUCE_INTRA = """Tu construis une COMPARAISON entre plusieurs éléments d'UN SEUL document (sections, périodes, options, approches…) à partir d'une analyse déjà réalisée de ce document, organisée par axe.

RÈGLES ABSOLUES
- Utilise UNIQUEMENT l'analyse fournie ci-dessous. Tu n'as pas accès au document complet ni à aucune autre source.
- N'invente aucun fait, chiffre, position ou conclusion.
- Chaque point (commun, différence, position, contradiction, conclusion) DOIT porter au moins une citation [D_S_] reprise de l'analyse.
- Compare les éléments demandés ENTRE EUX : attribue chaque affirmation à l'élément qu'elle concerne, sans en fusionner deux si un seul passage la soutient.
- Si le document se contredit d'une partie à l'autre, conserve la divergence EXPLICITEMENT dans "contradictions" ou "differences" — ne la lisse pas.
- Si le document ne dit rien de l'un des éléments demandés, ne fabrique rien pour lui et ne fabrique AUCUNE comparaison le concernant.
- "positions_par_document" : une entrée par ÉLÉMENT comparé (clé = libellé de l'élément), avec ses citations.
- La "conclusion" est facultative : ne la fournis que si l'analyse la soutient réellement, sinon mets null.

Réponds UNIQUEMENT avec un objet JSON strict :
{
  "points_communs": ["... [D1S2][D1S7]", ...],
  "differences": ["... [D1S3]", ...],
  "positions_par_document": {"<élément comparé>": "... [D1S1]"},
  "contradictions": ["... [D1S2] vs [D1S9]", ...],
  "conclusion": "... [D1S1][D1S4]" | null
}"""


def _systeme_reduce(profil_domaine: Any | None, portee: str = PORTEE_INTER) -> str:
    base = _SYSTEME_REDUCE_INTRA if portee == PORTEE_INTRA else _SYSTEME_REDUCE
    bloc = bloc_profil_domaine(profil_domaine)
    return f"{base}\n\n{bloc}" if bloc else base


def _liste_str(valeur: Any) -> list[str]:
    if isinstance(valeur, list):
        return [" ".join(str(x).split()) for x in valeur if str(x).strip()]
    if isinstance(valeur, str) and valeur.strip():
        return [" ".join(valeur.split())]
    return []


def _filtrer_par_citation(elements: list[str], autorisees: set[str]) -> tuple[list[str], list[str]]:
    """Garde les éléments portant >=1 citation valide (jetons hors périmètre
    retirés du texte) ; renvoie (gardés nettoyés, rejetés bruts)."""
    gardes, rejetes = [], []
    for e in elements:
        valides, _ = valider_citations(e, autorisees)
        if valides:
            gardes.append(retirer_citations_invalides(e, autorisees))
        else:
            rejetes.append(e)
    return gardes, rejetes


def comparer(
    question: str,
    references: Sequence[str],
    *,
    llm: Any,
    profil_domaine: Any | None = None,
    corpus_id: str = "default",
    strategy: Literal["map_reduce", "contextual", "hybrid"] = "map_reduce",
    portee: str = PORTEE_INTER,
) -> ResultatOutil:
    """
    Point d'entrée COMPARE. `references` = noms de fichiers explicites du
    signal multi-document (P1.4), résolus DANS LE CORPUS `corpus_id`
    uniquement. Abstention déterministe si la résolution n'est pas fiable —
    jamais de repli vers un search global, jamais un mélange de corpus.

    `strategy` (additif, défaut inchangé) : `"map_reduce"` est le chemin de
    production ci-dessous (PLAN -> MAP par document -> REDUCE, couverture
    intégrale garantie). `"contextual"` délègue à
    `src.agent.contextual_strategy.executer_contextual` — retrieval borné +
    UN SEUL appel LLM ; voir ce module et `report.md` pour l'évaluation
    comparative. `"hybrid"` tente `"contextual"` d'abord et ne se replie sur
    `"map_reduce"` que sur un motif déterministe précis — voir
    `_tenter_contextual_puis_repli` ci-dessous. Non routé par l'agent
    (`nodes.py` appelle toujours `comparer()` sans `strategy=`) : n'affecte
    jamais le comportement par défaut.
    """
    question = " ".join(str(question).split())

    if strategy == "contextual":
        from src.agent.contextual_strategy import executer_contextual

        return executer_contextual(
            operation="compare",
            question=question,
            corpus_id=corpus_id,
            llm=llm,
            documents=references,
            profil_domaine=profil_domaine,
        )

    if strategy == "hybrid":
        from src.agent.contextual_strategy import tenter_avec_repli

        return tenter_avec_repli(
            operation="compare",
            question=question,
            corpus_id=corpus_id,
            llm=llm,
            documents=references,
            profil_domaine=profil_domaine,
            repli=lambda: comparer(
                question, references, llm=llm, profil_domaine=profil_domaine,
                corpus_id=corpus_id, strategy="map_reduce", portee=portee,
            ),
        )

    intra = portee == PORTEE_INTRA
    resolution = resoudre_cibles(references, corpus_id=corpus_id, portee=portee)
    if resolution.refus is not None:
        return ResultatOutil.echec(_OUTIL, resolution.refus, motif=resolution.motif)

    if llm is None:
        return ResultatOutil.echec(_OUTIL, "Aucun LLM disponible pour la comparaison.")

    maps = executer_maps(
        resolution.documents,
        question,
        llm=llm,
        profil_domaine=profil_domaine,
        operation="compare",
        corpus_id=corpus_id,
        portee=portee,
    )
    utilisables, sans_evidence, echecs = diagnostic_maps(maps)

    if not utilisables:
        detail = []
        if sans_evidence:
            detail.append("sans information pertinente : " + ", ".join(sans_evidence))
        if echecs:
            detail.append("analyse impossible : " + ", ".join(echecs))
        return ResultatOutil.echec(
            _OUTIL,
            "Comparaison impossible : aucun document ne fournit d'élément "
            "exploitable pour la question"
            + (" (" + " ; ".join(detail) + ")" if detail else "")
            + ".",
            documents_sans_evidence=sans_evidence,
            documents_en_echec=echecs,
        )

    # Au moins UNE preuve exploitable existe : la comparaison peut être
    # tentée. "partiel" si un ou plusieurs des documents demandés n'ont rien
    # apporté (sans_evidence/échec) — le REDUCE ci-dessous est explicitement
    # instruit (voir `_SYSTEME_REDUCE`) de ne jamais fabriquer de comparaison
    # pour un document sans preuve.
    statut = STATUT_PARTIEL if (sans_evidence or echecs) else STATUT_COMPLET

    autorisees = citations_autorisees(maps)
    systeme = _systeme_reduce(profil_domaine, portee)
    utilisateur = (
        f"QUESTION DE COMPARAISON\n{question}\n\n"
        + ("ANALYSE DU DOCUMENT (par axe)\n" if intra else "ANALYSES PAR DOCUMENT\n")
        + f"{bloc_maps_pour_reduce(maps)}\n\n"
        "Produis maintenant l'objet JSON de comparaison."
    )

    # 2.5 — contrôle de taille du prompt REDUCE AVANT tout envoi. Dépassement
    # -> refus déterministe, jamais de troncature, jamais de compaction
    # pré-REDUCE (reportée à P1), jamais de repli SEARCH.
    budget = budget_caracteres_entree_llm()
    taille = len(systeme) + len(utilisateur)
    if taille > budget:
        return ResultatOutil.echec(
            _OUTIL,
            "Les analyses par document dépassent ce que le modèle peut traiter "
            f"en un seul appel ({taille} > {budget} caractères). Restreins la "
            "demande : moins de documents, ou des documents plus courts.",
            motif="budget_reduce_depasse",
        )

    try:
        # `reasoning=False` : même correctif que PLAN/MAP (voir
        # `multidoc_pipeline.py`, diagnostic Mode B) — REDUCE consomme un
        # JSON déjà structuré par les MAP, aucun raisonnement libre n'est
        # nécessaire ; sans ce paramètre, `think` reste au défaut du modèle
        # sous-jacent (activé pour qwen3), ce qui peut consommer tout
        # `num_predict` en raisonnement avant d'émettre le JSON de sortie et
        # renvoyer une réponse vide (voir report.md / diagnostic REDUCE).
        brut = invoquer_llm(llm, systeme=systeme, utilisateur=utilisateur, reasoning=False)
        objet = extraire_json_objet(brut)
    except Exception as exc:  # noqa: BLE001 — REDUCE raté => abstention, jamais hallucination
        logger.warning("REDUCE COMPARE échoué : %s", exc)
        return ResultatOutil.echec(_OUTIL, f"Synthèse comparative impossible : {exc}")

    points_communs, rej_pc = _filtrer_par_citation(_liste_str(objet.get("points_communs")), autorisees)
    differences, rej_diff = _filtrer_par_citation(_liste_str(objet.get("differences")), autorisees)
    contradictions, rej_contra = _filtrer_par_citation(_liste_str(objet.get("contradictions")), autorisees)

    positions_brutes = objet.get("positions_par_document") or {}
    positions: dict[str, str] = {}
    if isinstance(positions_brutes, dict):
        for libelle, texte in positions_brutes.items():
            texte = " ".join(str(texte).split())
            valides, _ = valider_citations(texte, autorisees)
            if valides:
                positions[str(libelle)] = retirer_citations_invalides(texte, autorisees)

    conclusion = objet.get("conclusion")
    conclusion = " ".join(str(conclusion).split()) if conclusion else None
    if conclusion:
        valides, _ = valider_citations(conclusion, autorisees)
        conclusion = retirer_citations_invalides(conclusion, autorisees) if valides else None

    if not (points_communs or differences or positions or contradictions):
        return ResultatOutil.echec(
            _OUTIL,
            "La comparaison produite n'était rattachable à aucune source : "
            "aucune provenance fiable.",
        )

    table_sources = sources_par_citation(maps)
    citations_utilisees: list[str] = []
    elements_cites = [
        *points_communs,
        *differences,
        *contradictions,
        *positions.values(),
        conclusion or "",
    ]
    for element in elements_cites:
        for citation in valider_citations(element, autorisees)[0]:
            if citation not in citations_utilisees:
                citations_utilisees.append(citation)
    sources: list[SourceOutil] = [
        table_sources[c] for c in citations_utilisees if c in table_sources
    ]

    resultat = ResultatCompare(
        question=question,
        documents=[m.cible.libelle for m in maps],
        points_communs=points_communs,
        differences=differences,
        positions_par_document=positions,
        contradictions=contradictions,
        conclusion=conclusion,
        documents_sans_evidence=sans_evidence,
        documents_en_echec=echecs,
        portee=portee,
    )

    avertissements: list[str] = []
    for m in maps:
        avertissements.extend(m.avertissements)
    if sans_evidence:
        avertissements.append(
            "Aucun élément pertinent trouvé dans : " + ", ".join(sans_evidence) + "."
        )
    if echecs:
        avertissements.append("Analyse indisponible pour : " + ", ".join(echecs) + ".")
    rejets = rej_pc + rej_diff + rej_contra
    if rejets:
        avertissements.append(
            f"{len(rejets)} affirmation(s) sans citation valide écartée(s)."
        )

    if intra:
        message = f"Comparaison au sein du document « {maps[0].cible.libelle} »."
    elif statut == STATUT_PARTIEL:
        message = (
            f"Comparaison partielle : {len(utilisables)}/{len(maps)} document(s) "
            "apportent des éléments exploitables ; voir les limitations."
        )
    else:
        message = (
            f"Comparaison de {len(maps)} documents "
            f"({len(utilisables)} avec des éléments pertinents)."
        )

    return ResultatOutil(
        outil=_OUTIL,
        succes=True,
        message=message,
        donnees={
            "statut": statut,
            "comparaison": asdict(resultat),
            "par_document": {
                m.cible.libelle: {
                    "citations": m.citations_valides,
                    "sans_evidence": m.sans_evidence,
                    "echec": m.echec,
                    "nombre_lots": m.nombre_lots,
                    "lots_en_echec": m.lots_en_echec,
                }
                for m in maps
            },
            "citations_utilisees": citations_utilisees,
        },
        sources=sources,
        avertissements=avertissements,
    )


def resultat_compare_depuis_donnees(donnees: dict[str, Any]) -> ResultatCompare | None:
    bloc = donnees.get("comparaison")
    if not isinstance(bloc, dict):
        return None
    try:
        return ResultatCompare(**bloc)
    except TypeError:
        return None


__all__ = ["ResultatCompare", "comparer", "resultat_compare_depuis_donnees"]
