"""
Validation de la suffisance du contexte documentaire.

Ce module ne modifie pas le classement du retrieval.
Il analyse les passages déjà récupérés et fournit une décision structurée
avant la génération finale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.rag.retrieval import RapportRecherche


@dataclass
class ValidationContexte:
    suffisant: bool
    raison: str
    score_maximum: float | None
    nombre_passages: int
    nombre_documents: int
    avertissements: list[str] = field(default_factory=list)


def valider_contexte(
    recherche: RapportRecherche,
    *,
    minimum_passages: int = 1,
    exiger_texte_non_vide: bool = True,
) -> ValidationContexte:
    """
    Validation déterministe et prudente.

    Cette première version n'utilise aucun seuil numérique universel.
    Elle vérifie seulement que le retrieval a fourni un contexte réellement
    exploitable avant d'appeler le LLM.
    """
    passages = recherche.passages

    if not passages:
        return ValidationContexte(
            suffisant=False,
            raison="Aucun passage n'a été récupéré.",
            score_maximum=None,
            nombre_passages=0,
            nombre_documents=0,
        )

    passages_valides = [
        passage
        for passage in passages
        if not exiger_texte_non_vide or passage.texte.strip()
    ]

    if len(passages_valides) < minimum_passages:
        return ValidationContexte(
            suffisant=False,
            raison=(
                "Le nombre de passages exploitables est inférieur "
                "au minimum requis."
            ),
            score_maximum=max(
                (p.score_final for p in passages),
                default=None,
            ),
            nombre_passages=len(passages_valides),
            nombre_documents=len(
                {
                    p.doc_id or p.nom_fichier or p.source
                    for p in passages_valides
                }
            ),
        )

    documents = {
        p.doc_id or p.nom_fichier or p.source
        for p in passages_valides
    }

    score_maximum = max(
        (p.score_final for p in passages_valides),
        default=None,
    )

    avertissements: list[str] = []

    if len(documents) == 1 and len(passages_valides) > 1:
        avertissements.append(
            "Tous les passages proviennent du même document."
        )

    return ValidationContexte(
        suffisant=True,
        raison="Le contexte contient au moins un passage exploitable.",
        score_maximum=score_maximum,
        nombre_passages=len(passages_valides),
        nombre_documents=len(documents),
        avertissements=avertissements,
    )