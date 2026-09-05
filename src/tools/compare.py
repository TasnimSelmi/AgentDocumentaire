"""
Capacité COMPARE (étape P1.5).

Compare explicitement 2 à 4 documents ciblés par l'utilisateur.
MAP par document (via `multidoc_pipeline`) -> REDUCE inter-document ->
`ResultatOutil` avec provenance par document.

Ne réalise AUCUN search global : le REDUCE ne voit que les sorties MAP
validées, jamais le corpus. Les désaccords entre documents sont conservés
explicitement.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from src.llm.common import bloc_profil_domaine, extraire_json_objet, invoquer_llm
from src.tools.base import ResultatOutil, SourceOutil

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


def _systeme_reduce(profil_domaine: Any | None) -> str:
    bloc = bloc_profil_domaine(profil_domaine)
    return f"{_SYSTEME_REDUCE}\n\n{bloc}" if bloc else _SYSTEME_REDUCE


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
) -> ResultatOutil:
    """
    Point d'entrée COMPARE. `references` = noms de fichiers explicites du
    signal multi-document (P1.4). Abstention déterministe si la résolution
    n'est pas fiable — jamais de repli vers un search global.
    """
    question = " ".join(str(question).split())

    resolution = resoudre_cibles(references)
    if resolution.refus is not None:
        return ResultatOutil.echec(_OUTIL, resolution.refus, motif=resolution.motif)

    if llm is None:
        return ResultatOutil.echec(_OUTIL, "Aucun LLM disponible pour la comparaison.")

    maps = executer_maps(
        resolution.documents, question, llm=llm, profil_domaine=profil_domaine
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
    systeme = _systeme_reduce(profil_domaine)
    utilisateur = (
        f"QUESTION DE COMPARAISON\n{question}\n\n"
        f"ANALYSES PAR DOCUMENT\n{bloc_maps_pour_reduce(maps)}\n\n"
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
        brut = invoquer_llm(llm, systeme=systeme, utilisateur=utilisateur)
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

    if statut == STATUT_PARTIEL:
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
