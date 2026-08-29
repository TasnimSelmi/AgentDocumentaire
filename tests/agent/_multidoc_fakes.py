"""
Doublures partagées pour les tests des branches COMPARE / SYNTHESIZE (P1.5).

Aucun Ollama, aucun Qdrant : `catalogue` / `charger_document` de
`src.agent.multidoc_pipeline` sont remplacés, et le LLM est une doublure
scriptée qui ne cite QUE ce qu'elle voit réellement dans le prompt.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from langchain_core.messages import AIMessage

from src.rag.retrieval import DocumentInconnu, FicheDocument, Passage

_MOTIF_CIT = re.compile(r"\[(D\d+S\d+)\]")

# Marqueur qu'un passage de fixture "ne contient rien de pertinent".
SANS_EVIDENCE = "AUCUNE INFORMATION PERTINENTE"


def EST_MAP_LOT(systeme: str) -> bool:
    return "analyses un extrait d'UN seul document" in systeme


def EST_AGREGATION_INTRA_DOC(systeme: str) -> bool:
    return "portant TOUTES sur le MÊME document" in systeme


def passage(doc_id: str, index: int, texte: str, *, page: int | None = None) -> Passage:
    return Passage(
        citation=f"S{index}",
        rang=index,
        point_id=f"{doc_id}-{index}",
        doc_id=doc_id,
        chunk_index=index,
        texte=texte,
        source=f"{doc_id}.pdf",
        nom_fichier=f"{doc_id}.pdf",
        page=page,
        categorie="",
        score_recherche=0.0,
        score_reranking=None,
    )


class FauxCatalogue:
    """`par_identifiant` : nom de fichier -> FicheDocument, sinon None."""

    def __init__(self, fiches_par_nom: dict[str, str]) -> None:
        # {nom_fichier: doc_id}
        self._m = fiches_par_nom

    def par_identifiant(self, identifiant: str) -> FicheDocument | None:
        cible = str(identifiant).strip()
        doc_id = self._m.get(cible)
        if doc_id is None:
            return None
        return FicheDocument(
            document_id=doc_id,
            champ_id="nom_fichier",
            nom_fichier=cible,
            titre="",
        )


def cabler_corpus(
    monkeypatch,
    module,
    *,
    fiches: dict[str, str],
    passages_par_doc: dict[str, list[Passage]],
) -> None:
    """Câble résolution + chargement pour un ensemble de documents fictifs.

    `module` = `src.agent.multidoc_pipeline` (là où `catalogue` /
    `charger_document` / `get_profil` sont importés)."""
    monkeypatch.setattr(module, "get_profil", lambda: None)
    monkeypatch.setattr(module, "catalogue", lambda profil=None: FauxCatalogue(fiches))

    def _charger(doc_id: str) -> list[Passage]:
        if doc_id in passages_par_doc:
            return passages_par_doc[doc_id]
        raise DocumentInconnu(doc_id)

    monkeypatch.setattr(module, "charger_document", _charger)


class LLMScripte:
    """LLM déterministe : MAP renvoie les faits pertinents cités du document ;
    REDUCE renvoie un JSON dont chaque élément porte des citations réellement
    vues dans le prompt. Ne cite jamais hors de ce qui lui est montré, sauf
    si `citation_hors_scope` est fournie (pour tester le filtrage)."""

    def __init__(self, *, citation_hors_scope: str | None = None) -> None:
        self.appels: list[tuple[str, str]] = []
        self.citation_hors_scope = citation_hors_scope

    def invoke(self, messages: Any) -> AIMessage:
        systeme, utilisateur = messages[0].content, messages[1].content
        self.appels.append((systeme, utilisateur))
        cites = list(dict.fromkeys(_MOTIF_CIT.findall(utilisateur)))

        if EST_MAP_LOT(systeme):
            # MAP d'un lot : si un passage du lot porte le marqueur, pas d'évidence.
            if SANS_EVIDENCE in utilisateur:
                return AIMessage(content=SANS_EVIDENCE)
            corps = " ".join(f"[{c}]" for c in cites[:3]) or "(rien)"
            return AIMessage(content=f"Élément pertinent du document. {corps}")

        if EST_AGREGATION_INTRA_DOC(systeme):
            # Consolidation des maps de lots d'un même document : conserve les
            # citations vues.
            corps = " ".join(f"[{c}]" for c in cites) or "(rien)"
            return AIMessage(content=f"Liste consolidée du document. {corps}")

        # REDUCE inter-document (COMPARE ou SYNTHESE)
        par_doc: dict[str, list[str]] = {}
        for c in cites:
            par_doc.setdefault(c[:2], []).append(c)  # "D1", "D2", ...
        docs = sorted(par_doc)
        premier = par_doc[docs[0]][0] if docs else ""
        second = par_doc[docs[1]][0] if len(docs) > 1 else premier
        # Cite aussi la DERNIÈRE citation de chaque document : quand un
        # document est réparti en plusieurs lots, cela fait remonter une
        # provenance issue d'un lot au-delà du premier.
        d1 = f"[{premier}][{par_doc[docs[0]][-1]}]" if docs else ""
        d2 = f"[{second}][{par_doc[docs[1]][-1]}]" if len(docs) > 1 else d1
        hs = f"[{self.citation_hors_scope}]" if self.citation_hors_scope else ""

        if "COMPARAISON" in systeme:
            objet = {
                "points_communs": [f"Constat commun. {d1}{d2}"],
                "differences": [
                    f"Le premier document indique une valeur {d1} "
                    f"tandis que le second indique une autre valeur {d2}. {hs}"
                ],
                "positions_par_document": {},
                "contradictions": [
                    f"Contradiction conservée : [{premier}] contre [{second}]."
                ],
                "conclusion": None,
            }
        else:
            objet = {
                "themes_communs": [f"Thème partagé. {d1}{d2}"],
                "elements_complementaires": [f"Apport propre au second. {d2} {hs}"],
                "divergences": [
                    f"D1 affirme X {d1} alors que D2 affirme Y {d2}."
                ],
                "synthese_transversale": (
                    f"Synthèse articulant les points ci-dessus. {d1}{d2}"
                ),
            }
        return AIMessage(content=json.dumps(objet, ensure_ascii=False))


class LLMExplose:
    def invoke(self, messages: Any) -> AIMessage:  # pragma: no cover
        raise RuntimeError("LLM injoignable")


def make_llm_map_echoue(sur_libelle: str) -> Callable:
    class _LLM:
        def __init__(self) -> None:
            self.appels: list[tuple[str, str]] = []

        def invoke(self, messages: Any) -> AIMessage:
            systeme, utilisateur = messages[0].content, messages[1].content
            self.appels.append((systeme, utilisateur))
            if EST_MAP_LOT(systeme) and sur_libelle in utilisateur:
                raise RuntimeError("MAP KO")
            base = LLMScripte()
            return base.invoke(messages)

    return _LLM()
