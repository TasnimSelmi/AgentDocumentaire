"""
Contrat de sortie PUBLIC et UNIQUE de l'agent documentaire — `AgentResponse`.

P1.3 : couche de normalisation MINIMALE, placée à la FRONTIÈRE de sortie de
l'agent. Elle ne modifie ni le routage, ni les capacités (SEARCH / SUMMARIZE
/ CLASSIFY / EXTRACT / COMPARE / SYNTHESIZE), ni le cœur RAG. Les tools
continuent de renvoyer leurs structures internes (`ResultatOutil`,
`ReponseRAG`) telles quelles ; `normaliser_reponse_agent` les transpose de
façon **100 % déterministe** (aucun appel LLM, aucune reconstruction de
provenance) vers `AgentResponse`.

`AgentResponse` est destiné à un consommateur externe (API / UI à venir). Il
ne transporte donc JAMAIS : prompts LLM, raisonnement, contenu complet des
documents, dump d'état du graphe, objets non sérialisables, configuration
secrète. Les traces détaillées relèvent d'une étape ultérieure (P2).

Dépendances : stdlib uniquement. Le typage des résultats internes est résolu
par duck-typing, ce qui garde ce module léger et testable avec des doublures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

# --- Valeurs contrôlées de `status` ---------------------------------------
STATUT_SUCCES = "success"
STATUT_REFUS = "refusal"
STATUT_ERREUR = "error"
STATUTS = (STATUT_SUCCES, STATUT_REFUS, STATUT_ERREUR)

# --- Capacités reconnues -------------------------------------------------
CAPACITES = ("search", "summarize", "classify", "extract", "compare", "synthesize")

#: Code d'erreur d'un refus SEARCH (preuves jugées non pertinentes ou non
#: suffisantes par la couche agentique). Les 5 autres capacités réutilisent le
#: `motif` déjà présent dans `ResultatOutil.donnees` quand il existe.
CODE_REFUS_SEARCH = "evidence_insuffisante"


@dataclass
class AgentResponse:
    """
    Sortie normalisée, unique et sérialisable de l'agent.

    Champs :
    - `status`      : `success` | `refusal` | `error`. `refusal` = abstention
                      FONCTIONNELLE attendue (jamais une exception). `error` =
                      vrai échec technique non prévu.
    - `capability`  : capacité réellement exécutée (`CAPACITES`), ou `""` si
                      une erreur est survenue avant le routage.
    - `answer`      : texte principal destiné au consommateur (réponse,
                      résumé, conclusion, message de refus…).
    - `sources`     : provenance déjà VALIDÉE par les capacités, au format
                      externe commun `{citation, document, page, categorie,
                      extrait, hors_perimetre}`. Jamais reconstruite.
    - `citations`   : liste ordonnée des identifiants de citation réellement
                      utilisés (`S1`, `D1S2`…). Vue uniforme, utile surtout
                      pour COMPARE/SYNTHESIZE où `sources` est dédupliquée.
    - `warnings`    : avertissements non bloquants, repris tels quels.
    - `data`        : payload structuré spécifique à la capacité (miroir
                      fidèle du résultat interne ; le gros texte déjà porté
                      par `answer` en est retiré).
    - `metadata`    : métadonnées légères déjà disponibles (durée, nombre de
                      sources, documents résolus, profil).
    - `error`       : `{"code", "message"}` si `status != success`, sinon
                      `None`.
    """

    status: str
    capability: str
    answer: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None

    def vers_dict(self) -> dict[str, Any]:
        """Représentation dict/JSON — uniquement des types natifs."""
        return asdict(self)


# =========================================================================
# Normalisation déterministe
# =========================================================================


def _texte(valeur: Any) -> str:
    return valeur if isinstance(valeur, str) else ""


def _source_vers_dict(src: Any) -> dict[str, Any]:
    """
    Normalise un `SourceCitee` (SEARCH) ou un `SourceOutil` (5 autres
    capacités) vers un dict externe commun — sans perte de provenance.
    Duck-typé : les deux structures exposent un sous-ensemble de ces attributs.
    """
    return {
        "citation": getattr(src, "citation", None) or None,
        "document": (
            getattr(src, "document", "")
            or getattr(src, "nom_fichier", "")
            or getattr(src, "source", "")
            or getattr(src, "doc_id", "")
            or ""
        ),
        "page": getattr(src, "page", None),
        "categorie": getattr(src, "categorie", "") or "",
        "extrait": getattr(src, "extrait", "") or "",
        "hors_perimetre": bool(getattr(src, "hors_perimetre", False)),
    }


def _est_reponse_rag(r: Any) -> bool:
    return (
        hasattr(r, "reponse")
        and hasattr(r, "contexte_suffisant")
        and hasattr(r, "sources")
        and not hasattr(r, "outil")
    )


def _est_resultat_outil(r: Any) -> bool:
    return hasattr(r, "outil") and hasattr(r, "succes") and hasattr(r, "donnees")


# `answer` : par défaut le `message`. Petite adaptation par capacité pour
# extraire le vrai texte utile — 100 % déterministe (lecture de dict).
_ANSWER_PAR_CAPACITE: dict[str, Callable[[dict[str, Any], str], str]] = {
    "summarize": lambda d, m: _texte(d.get("resume")) or m,
    "synthesize": lambda d, m: _texte((d.get("synthese") or {}).get("synthese_transversale")) or m,
    "compare": lambda d, m: _texte((d.get("comparaison") or {}).get("conclusion")) or m,
}

# `citations` : liste plate d'identifiants déjà présents dans le résultat.
_CITATIONS_PAR_CAPACITE: dict[str, Callable[[dict[str, Any]], list[str]]] = {
    "summarize": lambda d: [str(c) for c in (d.get("citations_valides") or [])],
    "classify": lambda d: [str(c) for c in (d.get("citations") or [])],
    "compare": lambda d: [str(c) for c in (d.get("citations_utilisees") or [])],
    "synthesize": lambda d: [str(c) for c in (d.get("citations_utilisees") or [])],
}


def _normaliser_search(
    r: Any,
    *,
    capability: str,
    preuves_pertinentes: bool | None,
    preuves_suffisantes: bool | None,
) -> AgentResponse:
    # La couche agentique tranche succès/refus AVANT génération : une réponse
    # n'est produite que si les deux niveaux valident les preuves.
    succes = bool(preuves_pertinentes) and bool(preuves_suffisantes)

    sources = [_source_vers_dict(s) for s in (getattr(r, "sources", None) or [])]
    citations = [s["citation"] for s in sources if s["citation"]]
    warnings = list(getattr(r, "avertissements", None) or [])
    answer = _texte(getattr(r, "reponse", ""))

    data: dict[str, Any] = {
        "contexte_suffisant": bool(getattr(r, "contexte_suffisant", False)),
        "citations_valides": bool(getattr(r, "citations_valides", False)),
        "citations_reparees": bool(getattr(r, "citations_reparees", False)),
        "citations_hors_perimetre": list(getattr(r, "citations_hors_perimetre", None) or []),
    }
    metadata: dict[str, Any] = {
        "duree_secondes": float(getattr(r, "duree_secondes", 0.0) or 0.0),
        "nombre_sources": len(sources),
        "documents_resolus": [],
        "profil": getattr(r, "profil", None) or None,
        "profil_domaine": getattr(r, "profil_domaine", None) or None,
    }

    return AgentResponse(
        status=STATUT_SUCCES if succes else STATUT_REFUS,
        capability=capability,
        answer=answer,
        sources=sources,
        citations=citations,
        warnings=warnings,
        data=data,
        metadata=metadata,
        error=None if succes else {"code": CODE_REFUS_SEARCH, "message": answer},
    )


def _normaliser_resultat_outil(
    r: Any,
    *,
    capability: str,
    documents_resolus: Sequence[str],
) -> AgentResponse:
    donnees = dict(getattr(r, "donnees", None) or {})
    succes = bool(getattr(r, "succes", False))
    message = _texte(getattr(r, "message", ""))
    sources = [_source_vers_dict(s) for s in (getattr(r, "sources", None) or [])]
    warnings = list(getattr(r, "avertissements", None) or [])

    answer = _ANSWER_PAR_CAPACITE.get(capability, lambda d, m: m)(donnees, message)
    citations = _CITATIONS_PAR_CAPACITE.get(
        capability,
        lambda d: [s["citation"] for s in sources if s["citation"]],
    )(donnees)

    # `data` : miroir fidèle du résultat structuré ; on retire uniquement le
    # gros texte déjà porté par `answer`.
    data = {
        cle: valeur
        for cle, valeur in donnees.items()
        if not (capability == "summarize" and cle == "resume")
    }

    metadata: dict[str, Any] = {
        "duree_secondes": float(getattr(r, "duree_secondes", 0.0) or 0.0),
        "nombre_sources": len(sources),
        "documents_resolus": [str(x) for x in (documents_resolus or [])],
        "profil": None,
        "profil_domaine": None,
    }

    return AgentResponse(
        status=STATUT_SUCCES if succes else STATUT_REFUS,
        capability=capability,
        answer=answer,
        sources=sources,
        citations=citations,
        warnings=warnings,
        data=data,
        metadata=metadata,
        error=None if succes else {"code": donnees.get("motif"), "message": message},
    )


def normaliser_reponse_agent(
    resultat_interne: Any,
    *,
    capability: str,
    preuves_pertinentes: bool | None = None,
    preuves_suffisantes: bool | None = None,
    documents_resolus: Sequence[str] = (),
) -> AgentResponse:
    """
    Transpose le résultat final déjà présent dans l'état du graphe vers
    `AgentResponse`.

    - `ReponseRAG` (branche SEARCH) : succès/refus décidé par
      `preuves_pertinentes` ET `preuves_suffisantes` (le graphe ne génère une
      réponse que si les deux valident).
    - `ResultatOutil` (SUMMARIZE / CLASSIFY / EXTRACT / COMPARE / SYNTHESIZE) :
      succès/refus = `ResultatOutil.succes` ; le code de refus réutilise
      `donnees["motif"]` quand il existe.

    Aucun appel LLM, aucune reconstruction de provenance : `sources` et
    `citations` ne font que transporter ce que les capacités ont déjà validé.
    """
    capability = str(capability or "").lower()

    if _est_reponse_rag(resultat_interne):
        return _normaliser_search(
            resultat_interne,
            capability=capability or "search",
            preuves_pertinentes=preuves_pertinentes,
            preuves_suffisantes=preuves_suffisantes,
        )

    if _est_resultat_outil(resultat_interne):
        return _normaliser_resultat_outil(
            resultat_interne,
            capability=capability or str(getattr(resultat_interne, "outil", "")).lower(),
            documents_resolus=documents_resolus,
        )

    # Le graphe n'a pas produit de résultat exploitable -> erreur technique.
    return AgentResponse(
        status=STATUT_ERREUR,
        capability=capability,
        error={
            "code": "resultat_interne_inattendu",
            "message": f"type de résultat non reconnu : {type(resultat_interne).__name__}",
        },
    )


__all__ = [
    "AgentResponse",
    "normaliser_reponse_agent",
    "STATUT_SUCCES",
    "STATUT_REFUS",
    "STATUT_ERREUR",
    "STATUTS",
    "CAPACITES",
    "CODE_REFUS_SEARCH",
]
