"""
Validation de la suffisance du contexte documentaire et des citations.

Ce module ne modifie pas le classement du retrieval.
Il analyse les passages déjà récupérés et la réponse produite, puis fournit
des décisions structurées :

    - avant la génération : le contexte est-il exploitable, et contient-il
      réellement des passages du périmètre documentaire demandé ?
    - après la génération : les citations sont-elles connues, et surtout
      renvoient-elles au document sur lequel portait la question ?

La seconde vérification est le cœur du module. Une citation syntaxiquement
valide qui renvoie au mauvais document produit une réponse fausse présentée
comme sourcée : c'est le pire des cas, car rien ne signale l'erreur à
l'utilisateur. Une citation est donc jugée sur sa provenance, pas seulement
sur sa forme.

Toutes les vérifications s'appuient sur les identifiants génériques du
périmètre documentaire : aucun nom de corpus, de société ou de fichier n'est
écrit en dur.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from src.rag.retrieval import Passage, PerimetreDocumentaire, RapportRecherche

# Identifiants de citation acceptés dans une réponse : [S1], [S2], ...
RE_CITATION = re.compile(r"\[S(\d+)\]")


# ===========================================================================
# Structures publiques
# ===========================================================================


@dataclass
class ValidationContexte:
    suffisant: bool
    raison: str
    score_maximum: float | None
    nombre_passages: int
    nombre_documents: int
    avertissements: list[str] = field(default_factory=list)
    # Diagnostic de cloisonnement : renseigné dès qu'un périmètre est imposé.
    perimetre_respecte: bool = True
    passages_dans_perimetre: int = 0
    passages_hors_perimetre: int = 0


@dataclass
class ValidationCitations:
    """Verdict complet sur les citations d'une réponse."""

    valides: bool
    citations: list[str] = field(default_factory=list)
    inconnues: list[str] = field(default_factory=list)
    hors_perimetre: list[str] = field(default_factory=list)
    citations_absentes: bool = False
    raison: str = ""
    avertissements: list[str] = field(default_factory=list)

    @property
    def citations_retenues(self) -> list[str]:
        """Citations exploitables : connues et dans le périmètre demandé."""
        rejetees = set(self.inconnues) | set(self.hors_perimetre)
        return [citation for citation in self.citations if citation not in rejetees]


# ===========================================================================
# Utilitaires de provenance
# ===========================================================================


def citations_dans_texte(texte: str) -> list[str]:
    """Renvoie les citations dans leur ordre d'apparition, sans doublon."""
    resultat: list[str] = []
    for numero in RE_CITATION.findall(str(texte)):
        citation = f"S{int(numero)}"
        if citation not in resultat:
            resultat.append(citation)
    return resultat


def _dans_perimetre(
    passage: Passage,
    perimetre: PerimetreDocumentaire | None,
) -> bool:
    """Vrai si le passage appartient au périmètre demandé.

    En l'absence de périmètre contraignant, tout passage est acceptable :
    la question ne visait aucun document en particulier.
    """
    if perimetre is None or not perimetre.contraignant:
        return True
    return perimetre.contient(*passage.identifiants)


def repartition_documents(passages: Sequence[Passage]) -> dict[str, int]:
    """Nombre de passages par document, pour le diagnostic."""
    repartition: dict[str, int] = {}
    for passage in passages:
        cle = passage.doc_id or passage.nom_fichier or passage.source or "inconnu"
        repartition[cle] = repartition.get(cle, 0) + 1
    return repartition


# ===========================================================================
# Validation du contexte
# ===========================================================================


def valider_contexte(
    recherche: RapportRecherche,
    *,
    minimum_passages: int = 1,
    exiger_texte_non_vide: bool = True,
) -> ValidationContexte:
    """
    Validation déterministe et prudente.

    Cette version n'utilise aucun seuil numérique universel. Elle vérifie que
    le retrieval a fourni un contexte réellement exploitable avant d'appeler
    le LLM, et — lorsqu'un périmètre documentaire a été imposé — que ce
    contexte contient bien des passages de ce périmètre.

    Un contexte composé uniquement de passages hors périmètre est déclaré
    insuffisant : générer à partir de lui reviendrait à répondre depuis un
    autre document que celui demandé.
    """
    passages = recherche.passages
    perimetre = getattr(recherche, "perimetre", None)

    if not passages:
        return ValidationContexte(
            suffisant=False,
            raison=(
                "Aucun passage n'a été récupéré"
                + (
                    f" dans le périmètre demandé ({', '.join(perimetre.libelles)})."
                    if perimetre is not None and perimetre.contraignant
                    else "."
                )
            ),
            score_maximum=None,
            nombre_passages=0,
            nombre_documents=0,
            perimetre_respecte=perimetre is None or not perimetre.contraignant,
        )

    passages_valides = [
        passage
        for passage in passages
        if not exiger_texte_non_vide or passage.texte.strip()
    ]

    # Cloisonnement : un passage étranger au périmètre ne peut pas servir de
    # base à la réponse, même s'il est pertinent sur le fond.
    dans_perimetre = [p for p in passages_valides if _dans_perimetre(p, perimetre)]
    nombre_hors = len(passages_valides) - len(dans_perimetre)
    exploitables = dans_perimetre

    if perimetre is not None and perimetre.contraignant and not exploitables:
        return ValidationContexte(
            suffisant=False,
            raison=(
                "Aucun passage récupéré n'appartient au périmètre demandé "
                f"({', '.join(perimetre.libelles)})."
            ),
            score_maximum=max((p.score_final for p in passages), default=None),
            nombre_passages=0,
            nombre_documents=0,
            perimetre_respecte=False,
            passages_dans_perimetre=0,
            passages_hors_perimetre=nombre_hors,
        )

    if len(exploitables) < minimum_passages:
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
            nombre_passages=len(exploitables),
            nombre_documents=len(repartition_documents(exploitables)),
            perimetre_respecte=nombre_hors == 0,
            passages_dans_perimetre=len(exploitables),
            passages_hors_perimetre=nombre_hors,
        )

    documents = repartition_documents(exploitables)
    score_maximum = max((p.score_final for p in exploitables), default=None)

    avertissements: list[str] = []

    if nombre_hors:
        # Anormal quand un filtre Qdrant a été posé : le signaler permet de
        # détecter un index de payload manquant plutôt que de le subir.
        avertissements.append(
            f"{nombre_hors} passage(s) hors périmètre écarté(s) du contexte."
        )

    # Un contexte mono-document n'est un signal que si personne ne l'a
    # demandé. Lorsque la question visait un document unique, c'est le
    # résultat attendu et non une anomalie.
    perimetre_unique = (
        perimetre is not None and perimetre.contraignant and perimetre.unique
    )
    if len(documents) == 1 and len(exploitables) > 1 and not perimetre_unique:
        avertissements.append(
            "Tous les passages proviennent du même document."
        )

    return ValidationContexte(
        suffisant=True,
        raison="Le contexte contient au moins un passage exploitable.",
        score_maximum=score_maximum,
        nombre_passages=len(exploitables),
        nombre_documents=len(documents),
        avertissements=avertissements,
        perimetre_respecte=nombre_hors == 0,
        passages_dans_perimetre=len(exploitables),
        passages_hors_perimetre=nombre_hors,
    )


# ===========================================================================
# Validation sémantique des citations
# ===========================================================================


def valider_citations(
    texte: str,
    passages: Sequence[Passage],
    *,
    perimetre: PerimetreDocumentaire | None = None,
    citations_obligatoires: bool = True,
) -> ValidationCitations:
    """
    Valide les citations d'une réponse sur la forme *et* sur la provenance.

    Trois motifs de rejet, du plus évident au plus insidieux :

    1. citation inconnue : l'identifiant n'existe pas dans le contexte ;
    2. citation absente alors qu'elle est obligatoire ;
    3. citation hors périmètre : l'identifiant existe et le passage cité
       existe, mais il provient d'un autre document que celui demandé.

    Le troisième cas est celui qu'une vérification purement syntaxique laisse
    passer. C'est pourtant le seul qui produise une valeur fausse attribuée
    à la bonne question, sans aucun signal d'erreur.
    """
    par_citation = {passage.citation: passage for passage in passages}
    citees = citations_dans_texte(texte)

    inconnues = [citation for citation in citees if citation not in par_citation]
    hors_perimetre = [
        citation
        for citation in citees
        if citation in par_citation
        and not _dans_perimetre(par_citation[citation], perimetre)
    ]

    citations_absentes = citations_obligatoires and not citees

    avertissements: list[str] = []
    motifs: list[str] = []

    if inconnues:
        motifs.append(
            "citations inconnues : " + ", ".join(f"[{c}]" for c in inconnues)
        )
    if citations_absentes:
        motifs.append("aucune citation alors qu'elles sont obligatoires")
    if hors_perimetre:
        libelles = (
            ", ".join(perimetre.libelles)
            if perimetre is not None and perimetre.libelles
            else "périmètre demandé"
        )
        motifs.append(
            "citations renvoyant à un document hors périmètre ("
            + libelles
            + ") : "
            + ", ".join(f"[{c}]" for c in hors_perimetre)
        )
        avertissements.append(
            "Une valeur a été citée depuis un document qui n'est pas celui "
            "demandé : la réponse ne peut pas être considérée comme sourcée."
        )

    valides = not inconnues and not hors_perimetre and not citations_absentes

    return ValidationCitations(
        valides=valides,
        citations=citees,
        inconnues=inconnues,
        hors_perimetre=hors_perimetre,
        citations_absentes=citations_absentes,
        raison=(
            "Citations valides et cohérentes avec le périmètre demandé."
            if valides
            else "Citations rejetées : " + " ; ".join(motifs) + "."
        ),
        avertissements=avertissements,
    )