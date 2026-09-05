"""
Capacité SYNTHESIZE (étape P1.5).

Produit une synthèse transversale de 2 à 4 documents ciblés par
l'utilisateur. Même structure que COMPARE : MAP par document (via
`multidoc_pipeline`) -> REDUCE inter-document -> `ResultatOutil` avec
provenance par document.

Une synthèse n'efface JAMAIS les désaccords : si A dit X et B dit Y, la
divergence est conservée explicitement. Aucun search global : le REDUCE ne
voit que les sorties MAP validées.
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

_OUTIL = "synthesize"

#: `donnees["statut"]` — voir `src.tools.compare.STATUT_COMPLET/STATUT_PARTIEL`
#: (même principe, même seuil, module volontairement découplé).
STATUT_COMPLET = "complet"
STATUT_PARTIEL = "partiel"


@dataclass
class ResultatSynthese:
    question: str
    documents: list[str]
    themes_communs: list[str] = field(default_factory=list)
    elements_complementaires: list[str] = field(default_factory=list)
    divergences: list[str] = field(default_factory=list)
    synthese_transversale: str | None = None
    documents_sans_evidence: list[str] = field(default_factory=list)
    documents_en_echec: list[str] = field(default_factory=list)


_SYSTEME_REDUCE = """Tu produis une SYNTHÈSE TRANSVERSALE de plusieurs documents à partir d'analyses déjà réalisées, une par document.

RÈGLES ABSOLUES
- Utilise UNIQUEMENT les analyses par document fournies ci-dessous. Tu n'as pas accès aux documents complets ni à aucune autre source.
- N'invente aucun fait, chiffre, recommandation ou conclusion.
- Chaque élément (thème commun, complément, divergence, phrase de synthèse) DOIT porter au moins une citation [D_S_] reprise des analyses.
- Ne fusionne jamais deux documents dans une même affirmation si une seule analyse la soutient.
- Si les documents divergent, CONSERVE la divergence dans "divergences" — ne la lisse pas, ne choisis pas un camp.
- Si un document n'apporte rien de pertinent, ne fabrique rien pour lui.
- Si UN SEUL document apporte des éléments exploitables (les autres n'apportant rien), ne fabrique AUCUN theme_commun ni AUCUNE divergence : décris uniquement le contenu de ce document (via "elements_complementaires" et/ou "synthese_transversale"), avec ses citations.
- "synthese_transversale" : un paragraphe court qui articule les points ci-dessus, avec citations. Mets null si les analyses ne permettent pas une synthèse honnête.

Réponds UNIQUEMENT avec un objet JSON strict :
{
  "themes_communs": ["... [D1S1][D2S2]", ...],
  "elements_complementaires": ["... [D2S3]", ...],
  "divergences": ["D1 indique ... [D1S1] alors que D2 indique ... [D2S1]", ...],
  "synthese_transversale": "... [D1S1][D2S2]" | null
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
    gardes, rejetes = [], []
    for e in elements:
        valides, _ = valider_citations(e, autorisees)
        if valides:
            gardes.append(retirer_citations_invalides(e, autorisees))
        else:
            rejetes.append(e)
    return gardes, rejetes


def synthetiser_documents(
    question: str,
    references: Sequence[str],
    *,
    llm: Any,
    profil_domaine: Any | None = None,
) -> ResultatOutil:
    """
    Point d'entrée SYNTHESIZE. `references` = noms de fichiers explicites du
    signal multi-document (P1.4). Abstention déterministe si la résolution
    n'est pas fiable — jamais de repli vers un search global.
    """
    question = " ".join(str(question).split())

    resolution = resoudre_cibles(references)
    if resolution.refus is not None:
        return ResultatOutil.echec(_OUTIL, resolution.refus, motif=resolution.motif)

    if llm is None:
        return ResultatOutil.echec(_OUTIL, "Aucun LLM disponible pour la synthèse.")

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
            "Synthèse impossible : aucun document ne fournit d'élément "
            "exploitable pour la question"
            + (" (" + " ; ".join(detail) + ")" if detail else "")
            + ".",
            documents_sans_evidence=sans_evidence,
            documents_en_echec=echecs,
        )

    # Au moins UNE preuve exploitable existe : la synthèse peut être tentée.
    # "partiel" si un ou plusieurs des documents demandés n'ont rien apporté
    # (sans_evidence/échec) — le REDUCE ci-dessous est explicitement instruit
    # (voir `_SYSTEME_REDUCE`) de ne jamais fabriquer de theme_commun ni de
    # divergence pour un document sans preuve.
    statut = STATUT_PARTIEL if (sans_evidence or echecs) else STATUT_COMPLET

    autorisees = citations_autorisees(maps)
    systeme = _systeme_reduce(profil_domaine)
    utilisateur = (
        f"QUESTION / OBJECTIF DE SYNTHÈSE\n{question}\n\n"
        f"ANALYSES PAR DOCUMENT\n{bloc_maps_pour_reduce(maps)}\n\n"
        "Produis maintenant l'objet JSON de synthèse transversale."
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
        logger.warning("REDUCE SYNTHESIZE échoué : %s", exc)
        return ResultatOutil.echec(_OUTIL, f"Synthèse transversale impossible : {exc}")

    themes, rej_t = _filtrer_par_citation(_liste_str(objet.get("themes_communs")), autorisees)
    complements, rej_c = _filtrer_par_citation(
        _liste_str(objet.get("elements_complementaires")), autorisees
    )
    divergences, rej_d = _filtrer_par_citation(_liste_str(objet.get("divergences")), autorisees)

    synthese = objet.get("synthese_transversale")
    synthese = " ".join(str(synthese).split()) if synthese else None
    if synthese:
        valides, _ = valider_citations(synthese, autorisees)
        synthese = retirer_citations_invalides(synthese, autorisees) if valides else None

    if not (themes or complements or divergences or synthese):
        return ResultatOutil.echec(
            _OUTIL,
            "La synthèse produite n'était rattachable à aucune source : "
            "aucune provenance fiable.",
        )

    table_sources = sources_par_citation(maps)
    citations_utilisees: list[str] = []
    for element in [*themes, *complements, *divergences, synthese or ""]:
        for citation in valider_citations(element, autorisees)[0]:
            if citation not in citations_utilisees:
                citations_utilisees.append(citation)
    sources: list[SourceOutil] = [
        table_sources[c] for c in citations_utilisees if c in table_sources
    ]

    resultat = ResultatSynthese(
        question=question,
        documents=[m.cible.libelle for m in maps],
        themes_communs=themes,
        elements_complementaires=complements,
        divergences=divergences,
        synthese_transversale=synthese,
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
    rejets = rej_t + rej_c + rej_d
    if rejets:
        avertissements.append(
            f"{len(rejets)} affirmation(s) sans citation valide écartée(s)."
        )

    if statut == STATUT_PARTIEL:
        message = (
            f"Synthèse partielle : {len(utilisables)}/{len(maps)} document(s) "
            "apportent des éléments exploitables ; voir les limitations."
        )
    else:
        message = (
            f"Synthèse transversale de {len(maps)} documents "
            f"({len(utilisables)} avec des éléments pertinents)."
        )

    return ResultatOutil(
        outil=_OUTIL,
        succes=True,
        message=message,
        donnees={
            "statut": statut,
            "synthese": asdict(resultat),
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


def resultat_synthese_depuis_donnees(donnees: dict[str, Any]) -> ResultatSynthese | None:
    bloc = donnees.get("synthese")
    if not isinstance(bloc, dict):
        return None
    try:
        return ResultatSynthese(**bloc)
    except TypeError:
        return None


__all__ = ["ResultatSynthese", "synthetiser_documents", "resultat_synthese_depuis_donnees"]
