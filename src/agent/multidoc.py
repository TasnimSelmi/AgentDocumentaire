"""
Détecteur générique multi-document (étape P1.4).

Fonction PURE et DÉTERMINISTE : à partir du seul texte d'une requête, décide
si celle-ci vise réellement **plusieurs documents distincts**, et — le cas
échéant — si le signal correspond à une **comparaison** ou à une **synthèse**
inter-documents.

Ce module ne fait que **produire un signal**. Il ne route pas, n'exécute
aucune comparaison, ne touche pas à la topologie du graphe. Le routing réel
et la topologie relèvent de P1.5.

Contraintes respectées :
- 0 appel LLM, 0 retrieval vectoriel, 0 génération, pas de planner ;
- aucun nom de document en dur ;
- logique bilingue FR/EN, aucune règle métier ;
- `src/rag/**`, `src/tools/**`, `src/agent/graph.py` non touchés ;
- un `resolveur` documentaire optionnel (lecture seule) peut être injecté,
  mais le détecteur reste pleinement fonctionnel et testable sans lui, avec
  des références fictives.

Principe de discrimination (le point délicat) :

    « Compare rapport_alpha.pdf et rapport_beta.pdf. »
        -> 2 références de fichier distinctes  => multi-doc COMPARE

    « Compare les deux méthodes décrites dans rapport_alpha.pdf. »
        -> « les deux <nom-non-documentaire> » + 1 seul fichier => MONO

    « Quels sont les points communs entre les deux approches présentées
      dans ce document ? »
        -> « ce document » (déixis singulière), 0 fichier => MONO

    « Dans ces documents, quelle est la date limite de dépôt ? »
        -> « ces documents » => multi-doc, mais aucune opération => hint none
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Iterable

# --------------------------------------------------------------------------
# Valeurs d'`operation_hint`
# --------------------------------------------------------------------------

HINT_COMPARE = "compare"
HINT_SYNTHESIZE = "synthesize"
HINT_AUCUN = "none"

# --------------------------------------------------------------------------
# Vocabulaire (générique, bilingue, sans terme métier)
# --------------------------------------------------------------------------

# Extensions de fichiers documentaires courantes — sert uniquement à repérer
# des RÉFÉRENCES explicites, jamais un nom de document précis.
_MOTIF_FICHIER = re.compile(
    r"\b[\w-]+\.(?:pdf|txt|docx?|csv|json|md|xlsx?|pptx?|rtf|odt|html?)\b",
    re.IGNORECASE,
)

# Noms communs désignant un CONTENANT documentaire (pas un contenu).
_NOMS_DOCUMENT = (
    "document", "documents", "fichier", "fichiers",
    "rapport", "rapports", "note", "notes",
    "contrat", "contrats", "texte", "textes",
    "courrier", "courriers", "lettre", "lettres",
    "memo", "memos", "papier", "papiers",
    "report", "reports", "file", "files", "doc", "docs",
    "paper", "papers", "letter", "letters", "record", "records",
)
_NOMS_DOCUMENT_PLURIEL = tuple(n for n in _NOMS_DOCUMENT if n.endswith("s"))

_NOMBRES = (
    "deux", "trois", "quatre", "plusieurs",
    "two", "three", "four", "several",
)
_VALEUR_NOMBRE = {
    "deux": 2, "two": 2, "trois": 3, "three": 3, "quatre": 4, "four": 4,
    "plusieurs": 2, "several": 2, "both": 2,
}

_ALT_DOC = "|".join(re.escape(n) for n in _NOMS_DOCUMENT)
_ALT_DOC_PLURIEL = "|".join(re.escape(n) for n in _NOMS_DOCUMENT_PLURIEL)
_ALT_NOMBRE = "|".join(re.escape(n) for n in _NOMBRES)

# Marqueurs « plusieurs documents » — DÉMONSTRATIF / NOMBRE + nom documentaire.
# Volontairement conservateur : « les deux méthodes » ne matche pas
# (« méthodes » n'est pas un nom de document).
_MOTIFS_PLURIEL_DOC = (
    re.compile(rf"\b(?:ces|les|those|these)\s+(?:{_ALT_NOMBRE}|\d+)\s+(?:{_ALT_DOC})\b"),
    re.compile(rf"\b(?:ces|these|those)\s+(?:{_ALT_DOC_PLURIEL})\b"),
    re.compile(rf"\bboth\s+(?:{_ALT_DOC})\b"),
    re.compile(rf"\bles\s+deux\s+(?:{_ALT_DOC})\b"),
    re.compile(rf"\b(?:the\s+)?(?:{_ALT_NOMBRE}|\d+)\s+(?:{_ALT_DOC_PLURIEL})\b"),
)

# Déixis SINGULIÈRE : « ce document », « this report », « le fichier »… —
# signal fort de mono-document.
_MOTIFS_DEIXIS_SINGULIER = (
    re.compile(rf"\b(?:ce|cet|cette|this|le|la|the)\s+(?:{_ALT_DOC})\b"),
    re.compile(rf"\b(?:du|de\s+ce|of\s+this|dans\s+ce|in\s+this)\s+(?:{_ALT_DOC})\b"),
)

# Marqueurs de COMPARAISON (sous-chaînes sur texte normalisé).
_MARQUEURS_COMPARE = (
    "compare", "comparer", "comparez", "comparons", "comparaison",
    "compares", "comparing", "compared", "comparative", "comparatif",
    "difference", "differences", "differe", "different", "differents",
    "differ", "differs", "differing", "en quoi",
    "ecart", "ecarts", "divergence", "diverge", "divergent",
    "points communs", "point commun", "in common", "common points",
    "similarite", "similarites", "similarities", "similitude", "similitudes",
    "versus", " vs ", " vs.", "par rapport a", "mettre en regard",
    "which of", "lequel de", "laquelle de", "lequel des", "laquelle des",
    "stricter", "strictest", "plus strict", "moins strict",
    "avantages et inconvenients", "pour et contre", "pros and cons",
    "oppose", "opposition entre",
)

# Marqueurs de SYNTHÈSE inter-documents (sous-chaînes sur texte normalisé).
_MARQUEURS_SYNTHESE = (
    "synthese", "syntheses", "synthetise", "synthetiser", "synthetisez",
    "synthesize", "synthesise", "synthesizing", "synthesis",
    "consolide", "consolider", "consolidez", "consolidation", "consolidate",
    "combine", "combiner", "combinez", "combined", "combining",
    "fusionne", "fusionner", "fusion des",
    "regroupe", "regrouper", "rassemble", "rassembler",
    "unified summary", "single summary", "one summary", "unified view",
    "summary of", "overview of", "vue d'ensemble",
    "key findings from", "key takeaways from", "principaux enseignements",
    "resume commun", "synthese commune", "bilan croise",
)


# --------------------------------------------------------------------------
# Résultat
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalMultiDoc:
    """Signal préparatoire — consommé plus tard par le routing (P1.5)."""

    is_multidoc: bool
    operation_hint: str  # HINT_COMPARE | HINT_SYNTHESIZE | HINT_AUCUN
    nombre_documents: int  # meilleure estimation ; 0 si vague, 1 si mono nommé
    references_detectees: tuple[str, ...]  # noms de fichiers distincts, normalisés
    marqueur_pluriel: str | None
    marqueurs_compare: tuple[str, ...] = field(default_factory=tuple)
    marqueurs_synthese: tuple[str, ...] = field(default_factory=tuple)
    confiance: str = "faible"  # "haute" | "moyenne" | "faible"
    raison: str = ""

    def vers_dict(self) -> dict:
        return {
            "is_multidoc": self.is_multidoc,
            "operation_hint": self.operation_hint,
            "nombre_documents": self.nombre_documents,
            "references_detectees": list(self.references_detectees),
            "marqueur_pluriel": self.marqueur_pluriel,
            "marqueurs_compare": list(self.marqueurs_compare),
            "marqueurs_synthese": list(self.marqueurs_synthese),
            "confiance": self.confiance,
            "raison": self.raison,
        }


# --------------------------------------------------------------------------
# Normalisation (locale — pas de dépendance au Core)
# --------------------------------------------------------------------------


def _normaliser(texte: str) -> str:
    brut = unicodedata.normalize("NFKD", str(texte))
    brut = "".join(c for c in brut if not unicodedata.combining(c))
    return brut.lower()


def _references_fichiers(texte_original: str) -> tuple[str, ...]:
    """Noms de fichiers distincts cités, dans l'ordre d'apparition."""
    vus: list[str] = []
    for correspondance in _MOTIF_FICHIER.finditer(texte_original):
        nom = correspondance.group(0).lower()
        if nom not in vus:
            vus.append(nom)
    return tuple(vus)


def _marqueur_pluriel(normalisee: str) -> tuple[str | None, int]:
    """Retourne (extrait matché, nombre estimé) ou (None, 0)."""
    for motif in _MOTIFS_PLURIEL_DOC:
        m = motif.search(normalisee)
        if not m:
            continue
        extrait = m.group(0)
        nombre = 2
        for mot, valeur in _VALEUR_NOMBRE.items():
            if mot in extrait:
                nombre = valeur
                break
        chiffres = re.search(r"\d+", extrait)
        if chiffres:
            try:
                nombre = max(2, int(chiffres.group(0)))
            except ValueError:
                pass
        return extrait, nombre
    return None, 0


def _sous_chaines_presentes(normalisee: str, marqueurs: Iterable[str]) -> tuple[str, ...]:
    return tuple(m for m in marqueurs if m in normalisee)


# --------------------------------------------------------------------------
# API publique
# --------------------------------------------------------------------------


def detecter_multidoc(
    query: str,
    *,
    resolveur: Callable[[str], Iterable[str]] | None = None,
) -> SignalMultiDoc:
    """
    Analyse `query` et renvoie un `SignalMultiDoc`.

    Args:
        query: requête utilisateur brute.
        resolveur: OPTIONNEL. Fonction lecture seule `query -> identifiants de
            documents résolus` (p. ex. un adaptateur du catalogue RAG). Si elle
            renvoie ≥ 2 identifiants distincts, cela **renforce** le signal
            multi-doc quand aucune référence explicite n'a été trouvée. Jamais
            appelée par défaut ; le détecteur fonctionne entièrement sans elle.

    Le détecteur ne combine JAMAIS un seul mot isolé : il croise
    (1) références de fichiers explicites, (2) marqueurs pluriels/pronominaux
    portant sur un nom de document, (3) marqueurs comparatifs,
    (4) marqueurs de synthèse — avec un garde-fou mono-document (déixis
    singulière + absence de références multiples).
    """
    original = str(query)
    normalisee = _normaliser(original)

    references = _references_fichiers(original)
    extrait_pluriel, nombre_pluriel = _marqueur_pluriel(normalisee)
    deixis_singuliere = any(m.search(normalisee) for m in _MOTIFS_DEIXIS_SINGULIER)

    refs_resolveur: tuple[str, ...] = ()
    if resolveur is not None and len(references) < 2:
        try:
            refs_resolveur = tuple(dict.fromkeys(str(r) for r in resolveur(original) if r))
        except Exception:  # noqa: BLE001 — un résolveur défaillant ne casse rien
            refs_resolveur = ()

    # --- Décision is_multidoc -------------------------------------------------
    raisons: list[str] = []
    if len(references) >= 2:
        is_multidoc = True
        nombre_documents = len(references)
        confiance = "haute"
        raisons.append(f"{len(references)} références explicites : {', '.join(references)}")
    elif extrait_pluriel and not deixis_singuliere:
        is_multidoc = True
        nombre_documents = nombre_pluriel
        confiance = "moyenne"
        raisons.append(f"marqueur pluriel « {extrait_pluriel} »")
    elif len(refs_resolveur) >= 2:
        is_multidoc = True
        nombre_documents = len(refs_resolveur)
        confiance = "moyenne"
        raisons.append(f"{len(refs_resolveur)} documents résolus par le résolveur injecté")
    else:
        is_multidoc = False
        nombre_documents = 1 if (len(references) == 1 or deixis_singuliere) else 0
        confiance = "faible"
        if deixis_singuliere:
            raisons.append("déixis singulière (« ce document »/« this report »…)")
        elif len(references) == 1:
            raisons.append(f"une seule référence : {references[0]}")
        else:
            raisons.append("aucune référence ni marqueur pluriel")

    # --- operation_hint (uniquement si multi-doc) ---------------------------
    marqueurs_c = _sous_chaines_presentes(normalisee, _MARQUEURS_COMPARE)
    marqueurs_s = _sous_chaines_presentes(normalisee, _MARQUEURS_SYNTHESE)

    if not is_multidoc:
        operation_hint = HINT_AUCUN
    elif marqueurs_c:
        operation_hint = HINT_COMPARE
        raisons.append(f"marqueur(s) comparatif(s) : {', '.join(marqueurs_c)}")
    elif marqueurs_s:
        operation_hint = HINT_SYNTHESIZE
        raisons.append(f"marqueur(s) de synthèse : {', '.join(m.strip() for m in marqueurs_s)}")
    else:
        operation_hint = HINT_AUCUN
        raisons.append("multi-doc sans marqueur d'opération (lecture factuelle probable)")

    return SignalMultiDoc(
        is_multidoc=is_multidoc,
        operation_hint=operation_hint,
        nombre_documents=nombre_documents,
        references_detectees=references,
        marqueur_pluriel=extrait_pluriel,
        marqueurs_compare=marqueurs_c,
        marqueurs_synthese=marqueurs_s,
        confiance=confiance,
        raison=" ; ".join(raisons),
    )
