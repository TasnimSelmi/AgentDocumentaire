"""
Tests d'intégration du graphe agentique (rechercher/évaluer/reformuler/répondre).

`src.rag.generation.generer_depuis_recherche` est systématiquement remplacé
par une doublure : ces tests vérifient le câblage et le routage du graphe,
pas la génération RAG elle-même (déjà testée ailleurs). Aucun serveur Ollama
ni collection Qdrant n'est nécessaire.

Les scores factices encadrent volontairement `nodes.SEUIL_PERTINENCE_MINIMALE`
(0.15) — voir le commentaire de calibrage dans `src/agent/nodes.py`.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from src.agent import nodes
from src.agent.graph import construire_graphe
from src.agent.graph_state import EtatGraphe
from src.agent.session import construire_session
from src.rag.generation import ReponseRAG
from src.rag.retrieval import Passage, PerimetreDocumentaire, RapportRecherche
from src.tools.base import DefinitionOutil, ResultatOutil, SourceOutil

_SCORE_EVIDENCE_FORTE = 0.90
_SCORE_EVIDENCE_FAIBLE = 0.03


class _ArgsSearchFactice(BaseModel):
    requete: str = Field(default="", description="Requête factice.")


def _reponse_factice(question: str) -> ReponseRAG:
    return ReponseRAG(
        question=question,
        reponse="Réponse factice.",
        profil="generic",
        contexte_suffisant=True,
        citations_valides=True,
        citations_reparees=False,
        sources=[],
    )


def _fabrique_generation_factice(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    appels: list[dict[str, Any]] = []

    def _fausse_generation(**kwargs: Any) -> ReponseRAG:
        appels.append(kwargs)
        return _reponse_factice(kwargs["question"])

    monkeypatch.setattr(nodes, "generer_depuis_recherche", _fausse_generation)
    return appels


def _un_passage(score: float) -> Passage:
    return Passage(
        citation="S1",
        rang=1,
        point_id="p1",
        doc_id="d1",
        chunk_index=0,
        texte="Extrait.",
        source="doc.pdf",
        nom_fichier="doc.pdf",
        page=1,
        categorie="",
        score_recherche=score,
        score_reranking=score,
    )


def _un_rapport(score: float | None) -> RapportRecherche:
    return RapportRecherche(
        requete="peu importe",
        profil="generic",
        filtres={},
        passages=[] if score is None else [_un_passage(score)],
        candidats_recuperes=0 if score is None else 1,
        reranking_utilise=True,
        seuil_applique=None,
        duree_secondes=0.01,
    )


def _outil_search_sequence(
    scores: Sequence[float | None],
) -> tuple[Callable[[], DefinitionOutil], list[int]]:
    """
    Fabrique de recherche factice à scores contrôlés (un par appel, le
    dernier répété au-delà). Renvoie aussi un compteur d'appels partagé,
    pour vérifier explicitement l'absence de double retrieval.
    """
    compteur: list[int] = [0]

    def _fonction(*, contexte=None, **kw) -> ResultatOutil:
        index = min(compteur[0], len(scores) - 1)
        compteur[0] += 1
        score = scores[index]

        rapport = _un_rapport(score)
        if contexte is not None:
            contexte.dernier_rapport_recherche = rapport

        if score is None:
            return ResultatOutil(outil="search", succes=True, message="Rien trouvé.")
        return ResultatOutil(
            outil="search",
            succes=True,
            message="1 passage trouvé.",
            sources=[
                SourceOutil(
                    doc_id="d1",
                    source="doc.pdf",
                    nom_fichier="doc.pdf",
                    page=1,
                    score=score,
                    extrait="Extrait.",
                )
            ],
        )

    def _fabrique() -> DefinitionOutil:
        return DefinitionOutil(
            nom="search",
            description="Search factice à scores contrôlés.",
            schema_arguments=_ArgsSearchFactice,
            fonction=_fonction,
        )

    return _fabrique, compteur


def _outil_search(score: float | None):
    fabrique, _ = _outil_search_sequence([score])
    return fabrique()


class _ArgsSummarizeFactice(BaseModel):
    objectif: str | None = Field(default=None)
    documents: list[str] | None = Field(default=None)
    format: str = Field(default="court")


def _outil_summarize_capture(
    resultat: ResultatOutil,
) -> tuple[Callable[[], DefinitionOutil], list[dict[str, Any]]]:
    """Fabrique factice de `summarize` qui enregistre les arguments reçus."""
    appels: list[dict[str, Any]] = []

    def _fonction(*, contexte=None, **kw) -> ResultatOutil:
        appels.append(kw)
        return resultat

    def _fabrique() -> DefinitionOutil:
        return DefinitionOutil(
            nom="summarize",
            description="Summarize factice.",
            schema_arguments=_ArgsSummarizeFactice,
            fonction=_fonction,
        )

    return _fabrique, appels


class _ArgsClassifyFactice(BaseModel):
    categories: list[str] = Field(default_factory=list)
    document: str | None = Field(default=None)
    critere: str | None = Field(default=None)
    instruction: str | None = Field(default=None)


def _outil_classify_capture(
    resultat: ResultatOutil,
) -> tuple[Callable[[], DefinitionOutil], list[dict[str, Any]]]:
    """Fabrique factice de `classify` qui enregistre les arguments reçus."""
    appels: list[dict[str, Any]] = []

    def _fonction(*, contexte=None, **kw) -> ResultatOutil:
        appels.append(kw)
        return resultat

    def _fabrique() -> DefinitionOutil:
        return DefinitionOutil(
            nom="classify",
            description="Classify factice.",
            schema_arguments=_ArgsClassifyFactice,
            fonction=_fonction,
        )

    return _fabrique, appels


class _ArgsExtractFactice(BaseModel):
    champs: list[str] = Field(default_factory=list)
    document: str | None = Field(default=None)
    documents: list[str] | None = Field(default=None)
    instruction: str | None = Field(default=None)


def _outil_extract_capture(
    resultat: ResultatOutil,
) -> tuple[Callable[[], DefinitionOutil], list[dict[str, Any]]]:
    """Fabrique factice de `extract` qui enregistre les arguments reçus."""
    appels: list[dict[str, Any]] = []

    def _fonction(*, contexte=None, **kw) -> ResultatOutil:
        appels.append(kw)
        return resultat

    def _fabrique() -> DefinitionOutil:
        return DefinitionOutil(
            nom="extract",
            description="Extract factice.",
            schema_arguments=_ArgsExtractFactice,
            fonction=_fonction,
        )

    return _fabrique, appels


class _LLMNonSollicite:
    def invoke(self, messages: Any):  # pragma: no cover - non sollicité ici
        raise AssertionError("Aucun LLM ne devrait être invoqué dans ce test.")


class _LLMChamps:
    """Répond au parsing des champs d'extraction (`noeud_extract`).

    N'est jamais sollicité pour la détection d'intention tant que la
    requête déclenche EXTRACT de façon déterministe (voir
    `_JETONS_EXTRACT_SURS`/`_MOTIF_CHAMP_VALEUR`) : aucune ambiguïté à
    lever dans ce cas, un seul type de message système est donc attendu.
    """

    def __init__(self, champs: list[str]) -> None:
        self._champs = champs

    def invoke(self, messages: Any) -> AIMessage:
        return AIMessage(content=json.dumps({"champs": self._champs}))


class _LLMExtractPipeline:
    """
    Distingue désambiguïsation SEARCH/EXTRACT et parsing des champs par le
    contenu du message système — nécessaire lorsque la requête est
    volontairement ambiguë (zone grise EXTRACT implicite) : les deux appels
    LLM bornés de `noeud_detecter_intention` puis `noeud_extract`
    interviennent l'un après l'autre dans le même test.
    """

    def __init__(self, *, intention: str = "EXTRACT", champs: list[str] | None = None) -> None:
        self._intention = intention
        self._champs = champs or []

    def invoke(self, messages: Any) -> AIMessage:
        systeme = messages[0].content
        if "EXTRACT :" in systeme and "distingues deux intentions" in systeme:
            return AIMessage(content=json.dumps({"intention": self._intention}))
        if "identifies la liste des informations" in systeme:
            return AIMessage(content=json.dumps({"champs": self._champs}))
        raise AssertionError(f"Message système inattendu : {systeme[:120]!r}")


class _LLMScripte:
    """Distingue reformulation et jugement de suffisance par le message système."""

    def __init__(
        self,
        *,
        reformulations: Sequence[str] = (),
        verdicts_suffisance: Sequence[str] = (),
    ) -> None:
        self._reformulations = list(reformulations)
        self._verdicts = list(verdicts_suffisance)
        self.appels_reformulation = 0
        self.appels_suffisance = 0

    def invoke(self, messages: Any) -> AIMessage:
        systeme = messages[0].content

        if "reformules une requête" in systeme:
            self.appels_reformulation += 1
            if self._reformulations:
                return AIMessage(content=self._reformulations.pop(0))
            return AIMessage(
                content=f'{{"requete_reformulee": "reformulation {self.appels_reformulation}"}}'
            )

        if "juges si des passages" in systeme:
            self.appels_suffisance += 1
            if self._verdicts:
                return AIMessage(content=self._verdicts.pop(0))
            return AIMessage(content='{"suffisant": false, "raison": "défaut de test"}')

        raise AssertionError(f"Message système inattendu : {systeme[:80]!r}")


def _invoquer(session) -> ReponseRAG:
    graphe = construire_graphe()
    limite = max(25, session.etat.max_tentatives * 3 + 5)
    resultat = graphe.invoke(EtatGraphe(session=session), config={"recursion_limit": limite})
    return resultat["reponse"]


# ---------------------------------------------------------------------------


def test_evidence_forte_et_suffisante_du_premier_coup_ne_reformule_pas(monkeypatch):
    appels_generation = _fabrique_generation_factice(monkeypatch)
    fabrique, compteur_recherche = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    llm = _LLMScripte(verdicts_suffisance=['{"suffisant": true, "raison": "ok"}'])

    session = construire_session(
        "Quelles sont les conditions de résiliation ?",
        llm=llm,
        charger_profil_domaine=False,
        fabriques=[fabrique],
    )
    reponse = _invoquer(session)

    assert isinstance(reponse, ReponseRAG)
    assert len(appels_generation) == 1
    assert compteur_recherche[0] == 1
    assert llm.appels_reformulation == 0
    assert llm.appels_suffisance == 1
    assert session.etat.tentatives == 1


def test_reformulation_ameliore_la_suffisance_et_permet_de_repondre(monkeypatch):
    """
    Cas positif explicite : preuves pertinentes mais jugées insuffisantes au
    premier essai, suffisantes après une reformulation. Le budget de
    recherche n'est pas épuisé, la génération est bien appelée avec le
    second rapport.
    """
    appels_generation = _fabrique_generation_factice(monkeypatch)
    fabrique, compteur_recherche = _outil_search_sequence(
        [_SCORE_EVIDENCE_FORTE, _SCORE_EVIDENCE_FORTE]
    )
    llm = _LLMScripte(
        reformulations=['{"requete_reformulee": "reformulation plus précise"}'],
        verdicts_suffisance=[
            '{"suffisant": false, "raison": "sujet correct mais valeur absente"}',
            '{"suffisant": true, "raison": "valeur trouvée", "elements_support": ["S1"]}',
        ],
    )

    session = construire_session(
        "Quel est le niveau exact de la métrique X pour l'année Y ?",
        llm=llm,
        charger_profil_domaine=False,
        fabriques=[fabrique],
        max_tentatives=6,
    )
    reponse = _invoquer(session)

    assert isinstance(reponse, ReponseRAG)
    assert compteur_recherche[0] == 2
    assert llm.appels_reformulation == 1
    assert llm.appels_suffisance == 2
    assert len(appels_generation) == 1
    assert session.etat.a_ete_reformulee


def test_evidence_non_pertinente_stagnante_stoppe_avant_le_budget(monkeypatch):
    """
    Score de pertinence constant d'une tentative à l'autre (le cas mesuré en
    smoke test réel) : le garde-fou anti-stagnation doit interrompre la
    boucle avant `max_tentatives`, sans jamais appeler le LLM de génération.
    """
    appels_generation = _fabrique_generation_factice(monkeypatch)
    fabrique, compteur_recherche = _outil_search_sequence([_SCORE_EVIDENCE_FAIBLE])
    llm = _LLMScripte()

    session = construire_session(
        "Question dont l'évidence est toujours faible",
        llm=llm,
        charger_profil_domaine=False,
        fabriques=[fabrique],
        max_tentatives=6,
    )
    reponse = _invoquer(session)

    assert isinstance(reponse, ReponseRAG)
    assert reponse.contexte_suffisant is False
    assert len(appels_generation) == 0
    assert llm.appels_suffisance == 0  # jamais pertinent, niveau 2 jamais atteint
    # Stagnation détectée dès la 2e évaluation : une seule reformulation.
    assert llm.appels_reformulation == 1
    assert compteur_recherche[0] == 2
    assert session.etat.tentatives < session.etat.max_tentatives


def test_evidence_non_pertinente_variable_epuise_le_budget(monkeypatch):
    """
    Sans stagnation (le score varie mais reste sous le seuil), la boucle va
    bien jusqu'au bout du budget avant de refuser — la garde stagnation ne
    doit pas interrompre un cas qui progresse simplement trop lentement.
    """
    appels_generation = _fabrique_generation_factice(monkeypatch)
    fabrique, compteur_recherche = _outil_search_sequence([0.02, 0.05, 0.03])
    llm = _LLMScripte()

    session = construire_session(
        "Question dont l'évidence reste faible mais varie",
        llm=llm,
        charger_profil_domaine=False,
        fabriques=[fabrique],
        max_tentatives=3,
    )
    reponse = _invoquer(session)

    assert isinstance(reponse, ReponseRAG)
    assert reponse.contexte_suffisant is False
    assert len(appels_generation) == 0
    assert compteur_recherche[0] == 3
    assert llm.appels_reformulation == 2
    assert session.etat.tentatives == 3
    assert not session.etat.peut_reessayer


def test_resultat_vide_route_vers_reformuler_puis_refuse(monkeypatch):
    appels_generation = _fabrique_generation_factice(monkeypatch)
    fabrique, compteur_recherche = _outil_search_sequence([None])
    llm = _LLMScripte()

    session = construire_session(
        "Question introuvable dans le corpus",
        llm=llm,
        charger_profil_domaine=False,
        fabriques=[fabrique],
        max_tentatives=6,
    )
    reponse = _invoquer(session)

    assert isinstance(reponse, ReponseRAG)
    assert reponse.contexte_suffisant is False
    assert len(appels_generation) == 0
    # Un RapportRecherche (vide) a bien été construit à chaque tentative :
    # le refus final passe par `refuser_sans_generation`, pas par le repli
    # « aucun rapport du tout » (réservé au cas où `search` échoue
    # techniquement, jamais exercé ici).
    assert "Corpus indisponible" not in reponse.reponse
    # Score constant à 0.0 (aucun passage) : stagnation dès la 2e évaluation.
    assert llm.appels_reformulation == 1
    assert compteur_recherche[0] == 2


def test_double_retrieval_ne_se_reproduit_plus(monkeypatch):
    """
    Test dédié au correctif principal de cette tâche : le nombre d'appels à
    l'outil `search` doit être strictement égal au nombre de tentatives
    consommées, jamais un de plus pour la génération finale.
    """
    appels_generation = _fabrique_generation_factice(monkeypatch)
    fabrique, compteur_recherche = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    llm = _LLMScripte(verdicts_suffisance=['{"suffisant": true, "raison": "ok"}'])

    session = construire_session(
        "Question answerable dès le premier essai",
        llm=llm,
        charger_profil_domaine=False,
        fabriques=[fabrique],
    )
    _invoquer(session)

    assert compteur_recherche[0] == session.etat.tentatives == 1
    assert len(appels_generation) == 1


# ===========================================================================
# Action 03B — routage SEARCH vs SUMMARIZE (détecter_intention)
# ===========================================================================


def test_A_routage_qa_reste_search(monkeypatch):
    """Une question factuelle continue de suivre la boucle QA existante."""
    appels_generation = _fabrique_generation_factice(monkeypatch)
    fabrique_search, compteur_recherche = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    resultat_summarize = ResultatOutil(
        outil="summarize", succes=True, message="Ne devrait jamais être appelé."
    )
    fabrique_summarize, appels_summarize = _outil_summarize_capture(resultat_summarize)
    llm = _LLMScripte(verdicts_suffisance=['{"suffisant": true, "raison": "ok"}'])

    session = construire_session(
        "Who is the CEO of company X?",
        llm=llm,
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_summarize],
    )
    reponse = _invoquer(session)

    assert isinstance(reponse, ReponseRAG)
    assert compteur_recherche[0] == 1
    assert appels_summarize == []
    etapes_intention = [e for e in session.etat.trace if e.nom == "intention"]
    assert len(etapes_intention) == 1
    assert etapes_intention[0].donnees["intention"] == "search"


def test_B_routage_resume_francais(monkeypatch):
    """« Résume le rapport CNIL 2023. » route vers SUMMARIZE."""
    perimetre = PerimetreDocumentaire(
        statut="exact", valeurs_filtre=("cnil-44e-rapport-annuel-2023.pdf",), libelles=("CNIL 2023",)
    )
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    fabrique_search, compteur_recherche = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    resultat_summarize = ResultatOutil(
        outil="summarize",
        succes=True,
        message="Résumé produit à partir du document complet (199 passage(s)).",
        donnees={"resume": "Le rapport CNIL 2023 couvre..."},
    )
    fabrique_summarize, appels_summarize = _outil_summarize_capture(resultat_summarize)

    session = construire_session(
        "Résume le rapport CNIL 2023.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_summarize],
    )
    reponse = _invoquer(session)

    assert reponse is resultat_summarize
    assert compteur_recherche[0] == 0
    assert len(appels_summarize) == 1
    assert appels_summarize[0]["documents"] == ["cnil-44e-rapport-annuel-2023.pdf"]


def test_C_routage_resume_anglais(monkeypatch):
    """« Summarize the CNIL annual report. » route aussi vers SUMMARIZE."""
    perimetre = PerimetreDocumentaire(
        statut="exact", valeurs_filtre=("cnil-44e-rapport-annuel-2023.pdf",), libelles=("CNIL 2023",)
    )
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    fabrique_search, compteur_recherche = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    resultat_summarize = ResultatOutil(
        outil="summarize", succes=True, message="Résumé produit.", donnees={"resume": "..."}
    )
    fabrique_summarize, appels_summarize = _outil_summarize_capture(resultat_summarize)

    session = construire_session(
        "Summarize the CNIL annual report.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_summarize],
    )
    reponse = _invoquer(session)

    assert reponse is resultat_summarize
    assert compteur_recherche[0] == 0
    assert len(appels_summarize) == 1


def test_D_summarize_appele_une_fois_search_jamais(monkeypatch):
    """Pour SUMMARIZE : summarize appelé exactement une fois, search jamais."""
    perimetre = PerimetreDocumentaire(statut="exact", valeurs_filtre=("doc-a",), libelles=("Doc A",))
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    fabrique_search, compteur_recherche = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    resultat_summarize = ResultatOutil(outil="summarize", succes=True, message="ok")
    fabrique_summarize, appels_summarize = _outil_summarize_capture(resultat_summarize)

    session = construire_session(
        "Fais-moi une synthèse du document A.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_summarize],
    )
    _invoquer(session)

    assert len(appels_summarize) == 1
    assert compteur_recherche[0] == 0
    assert session.outils_utilises() == ["summarize"]


def test_E_documents_transmis_au_tool(monkeypatch):
    """La désignation résolue de façon UNIQUE arrive dans documents=[id]."""
    perimetre = PerimetreDocumentaire(
        statut="exact",
        valeurs_filtre=("doc-a",),
        libelles=("Rapport A",),
    )
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    fabrique_search, _ = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    resultat_summarize = ResultatOutil(outil="summarize", succes=True, message="ok")
    fabrique_summarize, appels_summarize = _outil_summarize_capture(resultat_summarize)

    session = construire_session(
        "Résume le rapport A en mettant l'accent sur les engagements climatiques.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_summarize],
    )
    _invoquer(session)

    assert appels_summarize[0]["documents"] == ["doc-a"]
    assert "climatiques" in appels_summarize[0]["objectif"]


def test_E2_perimetre_multi_documents_refuse_sans_appeler_summarize(monkeypatch):
    """`compatible` multi-doc : refus déterministe, le map-reduce n'est jamais lancé."""
    perimetre = PerimetreDocumentaire(
        statut="compatible",
        valeurs_filtre=("doc-a", "doc-b"),
        libelles=("Rapport A", "Rapport B"),
    )
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    fabrique_search, compteur_recherche = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    resultat_summarize = ResultatOutil(outil="summarize", succes=True, message="ne doit pas être appelé")
    fabrique_summarize, appels_summarize = _outil_summarize_capture(resultat_summarize)

    session = construire_session(
        "Résume le document cquae_doc_219.txt.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_summarize],
    )
    reponse = _invoquer(session)

    assert appels_summarize == []
    assert compteur_recherche[0] == 0
    assert reponse.succes is False
    assert "Rapport A" in reponse.message


def test_F_succes_devient_la_reponse_finale(monkeypatch):
    """`ResultatOutil.succes=True` -> réponse finale de l'agent correctement remplie."""
    perimetre = PerimetreDocumentaire(statut="exact", valeurs_filtre=("doc-a",), libelles=("Doc A",))
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    fabrique_search, _ = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    resultat_summarize = ResultatOutil(
        outil="summarize",
        succes=True,
        message="Résumé produit.",
        donnees={"resume": "Contenu du résumé.", "citations_valides": ["S1", "S2"]},
        sources=[SourceOutil(doc_id="doc-a", source="a.pdf", nom_fichier="a.pdf", page=1, extrait="x")],
    )
    fabrique_summarize, _ = _outil_summarize_capture(resultat_summarize)

    session = construire_session(
        "Résume le document A.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_summarize],
    )
    reponse = _invoquer(session)

    assert isinstance(reponse, ResultatOutil)
    assert reponse.succes is True
    assert reponse.donnees["resume"] == "Contenu du résumé."
    assert len(reponse.sources) == 1


def test_G_echec_tool_termine_proprement_sans_boucle_qa(monkeypatch):
    """`succes=False` -> le graphe termine proprement, sans crash ni boucle QA."""
    perimetre = PerimetreDocumentaire(statut="exact", valeurs_filtre=("doc-x",), libelles=("Doc X",))
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    fabrique_search, compteur_recherche = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    resultat_echec = ResultatOutil(
        outil="summarize", succes=False, message="Document inconnu dans la collection."
    )
    fabrique_summarize, appels_summarize = _outil_summarize_capture(resultat_echec)
    llm = _LLMNonSollicite()

    session = construire_session(
        "Résume le document X.",
        llm=llm,
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_summarize],
    )
    reponse = _invoquer(session)

    assert reponse is resultat_echec
    assert reponse.succes is False
    assert "Document inconnu" in reponse.message
    assert compteur_recherche[0] == 0  # aucune boucle QA déclenchée
    assert len(appels_summarize) == 1  # pas de retry


def test_H_sans_document_explicite_reutilise_le_contexte_existant(monkeypatch):
    """
    « Résume les éléments trouvés. » (pas de document nommé) : documents=None,
    cohérent avec le mode historique de summarize (résume ContexteOutil.sources).
    """
    perimetre = PerimetreDocumentaire(statut="aucun", raison="aucune_correspondance")
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    fabrique_search, compteur_recherche = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    resultat_summarize = ResultatOutil(
        outil="summarize", succes=True, message="Résumé produit à partir du contexte existant."
    )
    fabrique_summarize, appels_summarize = _outil_summarize_capture(resultat_summarize)

    session = construire_session(
        "Résume les éléments trouvés.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_summarize],
    )
    # Simule un `search` déjà exécuté plus tôt dans la même session.
    session.contexte.sources.append(
        SourceOutil(doc_id="d1", source="doc.pdf", nom_fichier="doc.pdf", page=1, extrait="Extrait.")
    )

    reponse = _invoquer(session)

    assert reponse is resultat_summarize
    assert appels_summarize[0]["documents"] is None
    assert compteur_recherche[0] == 0


def test_I_ambiguite_documentaire_conserve_le_refus(monkeypatch):
    """
    Désignation ambiguë : le refus de `summarize` est conservé tel quel, sans
    document choisi arbitrairement et sans bascule silencieuse vers search.
    """
    perimetre = PerimetreDocumentaire(
        statut="ambigu", raison="marge_insuffisante", libelles=("Rapport A", "Rapport B")
    )
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    fabrique_search, compteur_recherche = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    resultat_ne_doit_pas_servir = ResultatOutil(
        outil="summarize", succes=False, message="ne doit pas être appelé",
    )
    fabrique_summarize, appels_summarize = _outil_summarize_capture(resultat_ne_doit_pas_servir)

    session = construire_session(
        "Résume le rapport.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_summarize],
    )
    reponse = _invoquer(session)

    # Le refus est désormais construit dans le nœud : `summarize` n'est jamais
    # appelé, `search` non plus, aucun document n'est choisi arbitrairement.
    assert reponse.succes is False
    assert appels_summarize == []
    assert compteur_recherche[0] == 0
    assert "Rapport A" in reponse.message


# ===========================================================================
# Action « classify dans le graphe » — routage vers CLASSIFY
# ===========================================================================


def test_classify_A_qa_reste_search(monkeypatch):
    """Une question factuelle continue de suivre la boucle QA, classify jamais appelé."""
    appels_generation = _fabrique_generation_factice(monkeypatch)
    fabrique_search, compteur_recherche = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    fabrique_summarize, appels_summarize = _outil_summarize_capture(
        ResultatOutil(outil="summarize", succes=True, message="ne devrait jamais être appelé")
    )
    fabrique_classify, appels_classify = _outil_classify_capture(
        ResultatOutil(outil="classify", succes=True, message="ne devrait jamais être appelé")
    )
    llm = _LLMScripte(verdicts_suffisance=['{"suffisant": true, "raison": "ok"}'])

    session = construire_session(
        "Who is the CEO of company X?",
        llm=llm,
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_summarize, fabrique_classify],
    )
    reponse = _invoquer(session)

    assert isinstance(reponse, ReponseRAG)
    assert compteur_recherche[0] == 1
    assert appels_summarize == []
    assert appels_classify == []


def test_classify_B_resume_reste_summarize(monkeypatch):
    """« Résume le rapport CNIL. » reste SUMMARIZE, classify jamais appelé."""
    perimetre = PerimetreDocumentaire(statut="exact", valeurs_filtre=("doc-cnil",), libelles=("CNIL",))
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    fabrique_search, compteur_recherche = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    fabrique_summarize, appels_summarize = _outil_summarize_capture(
        ResultatOutil(outil="summarize", succes=True, message="ok")
    )
    fabrique_classify, appels_classify = _outil_classify_capture(
        ResultatOutil(outil="classify", succes=True, message="ne devrait jamais être appelé")
    )

    session = construire_session(
        "Résume le rapport CNIL.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_summarize, fabrique_classify],
    )
    reponse = _invoquer(session)

    assert isinstance(reponse, ResultatOutil)
    assert reponse.outil == "summarize"
    assert len(appels_summarize) == 1
    assert appels_classify == []
    assert compteur_recherche[0] == 0


def test_classify_C_classification_francaise(monkeypatch):
    """
    « Classe le rapport CNIL 2023. » route vers CLASSIFY.

    Document résolu de façon fiable (périmètre exact) : classify passe en
    mode document complet (Option E) et n'a plus besoin d'un search interne
    pour s'alimenter — contrairement à l'ancien comportement top-k.
    """
    perimetre = PerimetreDocumentaire(
        statut="exact", valeurs_filtre=("cnil-44e-rapport-annuel-2023.pdf",), libelles=("CNIL 2023",)
    )
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    fabrique_search, compteur_recherche = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    resultat_classify = ResultatOutil(
        outil="classify",
        succes=True,
        message="Document classifié dans la catégorie « rapport ».",
        donnees={"categorie": "rapport", "confiance": 0.9},
    )
    fabrique_classify, appels_classify = _outil_classify_capture(resultat_classify)

    session = construire_session(
        "Classe le rapport CNIL 2023.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_classify],
    )
    reponse = _invoquer(session)

    assert reponse is resultat_classify
    assert len(appels_classify) == 1
    assert compteur_recherche[0] == 0  # document résolu -> mode document complet, aucun search


def test_classify_D_classification_anglaise(monkeypatch):
    """« Classify the CNIL annual report. » route aussi vers CLASSIFY."""
    perimetre = PerimetreDocumentaire(
        statut="exact", valeurs_filtre=("cnil-44e-rapport-annuel-2023.pdf",), libelles=("CNIL 2023",)
    )
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    fabrique_search, _ = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    resultat_classify = ResultatOutil(outil="classify", succes=True, message="ok")
    fabrique_classify, appels_classify = _outil_classify_capture(resultat_classify)

    session = construire_session(
        "Classify the CNIL annual report.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_classify],
    )
    reponse = _invoquer(session)

    assert reponse is resultat_classify
    assert len(appels_classify) == 1


def test_classify_E_bon_tool_search_au_plus_une_fois(monkeypatch):
    """
    classify appelé exactement une fois, summarize jamais. `search` est
    appelé au plus une fois — désormais zéro fois quand le document se
    résout de façon fiable (mode document complet, Option E), et au plus
    une fois dans le cas contraire (mode historique sur
    `ContexteOutil.sources`) — jamais davantage, jamais pour un fallback
    vers la boucle QA.
    """
    perimetre = PerimetreDocumentaire(statut="exact", valeurs_filtre=("doc-a",), libelles=("Doc A",))
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    fabrique_search, compteur_recherche = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    resultat_classify = ResultatOutil(outil="classify", succes=True, message="ok")
    fabrique_classify, appels_classify = _outil_classify_capture(resultat_classify)
    fabrique_summarize, appels_summarize = _outil_summarize_capture(
        ResultatOutil(outil="summarize", succes=True, message="ne devrait jamais être appelé")
    )

    session = construire_session(
        "Classe le document A.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_classify, fabrique_summarize],
    )
    _invoquer(session)

    assert len(appels_classify) == 1
    assert compteur_recherche[0] <= 1
    assert appels_summarize == []
    assert session.outils_utilises()[-1] == "classify"


def test_classify_F_document_transmis(monkeypatch):
    """Le document résolu arrive correctement dans documents=[...] (mode document complet)."""
    perimetre = PerimetreDocumentaire(
        statut="exact", valeurs_filtre=("cnil-44e-rapport-annuel-2023.pdf",), libelles=("CNIL 2023",)
    )
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    fabrique_search, _ = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    resultat_classify = ResultatOutil(outil="classify", succes=True, message="ok")
    fabrique_classify, appels_classify = _outil_classify_capture(resultat_classify)

    session = construire_session(
        "Classe le rapport CNIL 2023.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_classify],
    )
    _invoquer(session)

    assert appels_classify[0]["documents"] == ["cnil-44e-rapport-annuel-2023.pdf"]
    assert "document" not in appels_classify[0]
    assert appels_classify[0]["categories"]  # catégories du profil technique, non vides


def test_classify_G_succes_devient_la_reponse_finale(monkeypatch):
    """`ResultatOutil.succes=True` -> réponse finale de l'agent correctement remplie."""
    perimetre = PerimetreDocumentaire(statut="exact", valeurs_filtre=("doc-a",), libelles=("Doc A",))
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    fabrique_search, _ = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    resultat_classify = ResultatOutil(
        outil="classify",
        succes=True,
        message="Document classifié dans la catégorie « rapport ».",
        donnees={"categorie": "rapport", "confiance": 0.87, "citations": ["S1"]},
        sources=[SourceOutil(doc_id="doc-a", source="a.pdf", nom_fichier="a.pdf", page=1, extrait="x")],
    )
    fabrique_classify, _ = _outil_classify_capture(resultat_classify)

    session = construire_session(
        "Classe le document A.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_classify],
    )
    reponse = _invoquer(session)

    assert isinstance(reponse, ResultatOutil)
    assert reponse.succes is True
    assert reponse.donnees["categorie"] == "rapport"
    assert len(reponse.sources) == 1


def test_classify_H_echec_termine_proprement_sans_boucle_qa(monkeypatch):
    """`succes=False` -> le graphe termine proprement, sans crash ni boucle QA."""
    perimetre = PerimetreDocumentaire(statut="exact", valeurs_filtre=("doc-x",), libelles=("Doc X",))
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    fabrique_search, compteur_recherche = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    resultat_echec = ResultatOutil(
        outil="classify",
        succes=False,
        message="Plusieurs documents sont présents dans le contexte. Précise le document à classifier.",
    )
    fabrique_classify, appels_classify = _outil_classify_capture(resultat_echec)

    session = construire_session(
        "Classe le document X.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_classify],
    )
    reponse = _invoquer(session)

    assert reponse is resultat_echec
    assert reponse.succes is False
    assert "Précise le document" in reponse.message
    assert len(appels_classify) == 1  # pas de retry
    assert compteur_recherche[0] <= 1  # aucune boucle QA déclenchée


def test_classify_I_ambiguite_documentaire_conserve_le_refus(monkeypatch):
    """
    Désignation ambiguë : `classify` n'est PAS appelé (aucun choix implicite
    entre les candidats), le nœud construit lui-même un refus déterministe,
    et aucun `search` de repli n'est déclenché.
    """
    perimetre = PerimetreDocumentaire(
        statut="ambigu", raison="marge_insuffisante", libelles=("Rapport A", "Rapport B")
    )
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    fabrique_search, compteur_recherche = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    resultat_jamais_appele = ResultatOutil(
        outil="classify", succes=True, message="ne devrait jamais être appelé"
    )
    fabrique_classify, appels_classify = _outil_classify_capture(resultat_jamais_appele)

    session = construire_session(
        "Classe le rapport.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_classify],
    )
    reponse = _invoquer(session)

    assert reponse is not resultat_jamais_appele
    assert appels_classify == []
    assert compteur_recherche[0] == 0
    assert reponse.succes is False


# ===========================================================================
# EXTRACT (Action 04)
# ===========================================================================


def test_extract_A_qa_reste_search(monkeypatch):
    """Une question factuelle simple continue de suivre la boucle QA, extract jamais appelé."""
    appels_generation = _fabrique_generation_factice(monkeypatch)
    fabrique_search, compteur_recherche = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    fabrique_extract, appels_extract = _outil_extract_capture(
        ResultatOutil(outil="extract", succes=True, message="ne devrait jamais être appelé")
    )
    llm = _LLMScripte(verdicts_suffisance=['{"suffisant": true, "raison": "ok"}'])

    session = construire_session(
        "What is the invoice date?",
        llm=llm,
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_extract],
    )
    reponse = _invoquer(session)

    assert isinstance(reponse, ReponseRAG)
    assert compteur_recherche[0] == 1
    assert appels_extract == []


def test_extract_B_summarize_reste_summarize(monkeypatch):
    """« Résume le rapport CNIL. » reste SUMMARIZE, extract jamais appelé."""
    perimetre = PerimetreDocumentaire(statut="exact", valeurs_filtre=("doc-cnil",), libelles=("CNIL",))
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    fabrique_search, _ = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    fabrique_summarize, appels_summarize = _outil_summarize_capture(
        ResultatOutil(outil="summarize", succes=True, message="ok")
    )
    fabrique_extract, appels_extract = _outil_extract_capture(
        ResultatOutil(outil="extract", succes=True, message="ne devrait jamais être appelé")
    )

    session = construire_session(
        "Résume le rapport CNIL.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_summarize, fabrique_extract],
    )
    reponse = _invoquer(session)

    assert isinstance(reponse, ResultatOutil)
    assert reponse.outil == "summarize"
    assert appels_extract == []


def test_extract_C_classify_reste_classify(monkeypatch):
    """« Classe le rapport CNIL 2023. » reste CLASSIFY, extract jamais appelé."""
    perimetre = PerimetreDocumentaire(
        statut="exact", valeurs_filtre=("cnil-44e-rapport-annuel-2023.pdf",), libelles=("CNIL 2023",)
    )
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    fabrique_search, _ = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    fabrique_classify, appels_classify = _outil_classify_capture(
        ResultatOutil(outil="classify", succes=True, message="ok")
    )
    fabrique_extract, appels_extract = _outil_extract_capture(
        ResultatOutil(outil="extract", succes=True, message="ne devrait jamais être appelé")
    )

    session = construire_session(
        "Classe le rapport CNIL 2023.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_classify, fabrique_extract],
    )
    reponse = _invoquer(session)

    assert reponse.outil == "classify"
    assert appels_extract == []


def test_extract_D_explicite_route_vers_extract_document_complet(monkeypatch):
    """« Extrais le fournisseur, la date et le montant du rapport CNIL 2023. » route vers EXTRACT."""
    perimetre = PerimetreDocumentaire(
        statut="exact", valeurs_filtre=("cnil-44e-rapport-annuel-2023.pdf",), libelles=("CNIL 2023",)
    )
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    fabrique_search, compteur_recherche = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    resultat_extract = ResultatOutil(outil="extract", succes=True, message="ok")
    fabrique_extract, appels_extract = _outil_extract_capture(resultat_extract)

    session = construire_session(
        "Extrais le fournisseur, la date et le montant du rapport CNIL 2023.",
        llm=_LLMChamps(["fournisseur", "date", "montant"]),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_extract],
    )
    reponse = _invoquer(session)

    assert reponse is resultat_extract
    assert appels_extract[0]["documents"] == ["cnil-44e-rapport-annuel-2023.pdf"]
    assert appels_extract[0]["champs"] == ["fournisseur", "date", "montant"]
    assert compteur_recherche[0] == 0  # document résolu -> mode document complet, aucun search


def test_extract_E_implicite_route_vers_extract(monkeypatch):
    """
    « Donne-moi le fournisseur, la date et le montant total. » (aucun verbe
    d'extraction) doit être routée vers EXTRACT via la désambiguïsation
    bornée, exactement comme le veut la mission.
    """
    monkeypatch.setattr(
        nodes, "resoudre_document", lambda requete: PerimetreDocumentaire(statut="aucun", raison="aucune")
    )

    fabrique_search, compteur_recherche = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    resultat_extract = ResultatOutil(outil="extract", succes=True, message="ok")
    fabrique_extract, appels_extract = _outil_extract_capture(resultat_extract)

    llm = _LLMExtractPipeline(intention="EXTRACT", champs=["fournisseur", "date", "montant total"])

    session = construire_session(
        "Donne-moi le fournisseur, la date et le montant total.",
        llm=llm,
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_extract],
    )
    reponse = _invoquer(session)

    assert reponse is resultat_extract
    assert appels_extract[0]["champs"] == ["fournisseur", "date", "montant total"]
    # Aucun document explicitement visé : mode contextuel, search interne
    # au plus une fois (comportement historique, contexte vide au départ).
    assert compteur_recherche[0] == 1


def test_extract_F_ambiguite_documentaire_conserve_le_refus(monkeypatch):
    """
    Désignation ambiguë : `extract` n'est PAS appelé (aucun choix implicite
    entre les candidats), le nœud construit lui-même un refus déterministe,
    et aucun `search` de repli n'est déclenché — même garantie que CLASSIFY.
    """
    perimetre = PerimetreDocumentaire(
        statut="ambigu", raison="marge_insuffisante", libelles=("Rapport A", "Rapport B")
    )
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    fabrique_search, compteur_recherche = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    resultat_jamais_appele = ResultatOutil(
        outil="extract", succes=True, message="ne devrait jamais être appelé"
    )
    fabrique_extract, appels_extract = _outil_extract_capture(resultat_jamais_appele)

    session = construire_session(
        "Extrais le montant du rapport.",
        llm=_LLMChamps(["montant"]),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_extract],
    )
    reponse = _invoquer(session)

    assert reponse is not resultat_jamais_appele
    assert appels_extract == []
    assert compteur_recherche[0] == 0
    assert reponse.succes is False


def test_extract_G_echec_termine_proprement_sans_boucle_qa(monkeypatch):
    """`succes=False` -> le graphe termine proprement, sans crash ni boucle QA."""
    perimetre = PerimetreDocumentaire(statut="exact", valeurs_filtre=("doc-x",), libelles=("Doc X",))
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    fabrique_search, compteur_recherche = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    resultat_echec = ResultatOutil(
        outil="extract", succes=False, message="Aucun champ d'extraction valide n'a été fourni."
    )
    fabrique_extract, appels_extract = _outil_extract_capture(resultat_echec)

    session = construire_session(
        "Extrais le document X.",
        llm=_LLMChamps([]),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_extract],
    )
    reponse = _invoquer(session)

    assert reponse is resultat_echec
    assert reponse.succes is False
    assert len(appels_extract) == 1  # pas de retry
    assert compteur_recherche[0] == 0


def test_extract_H_succes_devient_la_reponse_finale(monkeypatch):
    """`ResultatOutil.succes=True` -> réponse finale de l'agent correctement remplie."""
    perimetre = PerimetreDocumentaire(statut="exact", valeurs_filtre=("doc-a",), libelles=("Doc A",))
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    fabrique_search, _ = _outil_search_sequence([_SCORE_EVIDENCE_FORTE])
    resultat_extract = ResultatOutil(
        outil="extract",
        succes=True,
        message="1 information(s) extraite(s) sur 1 champ(s) demandé(s).",
        donnees={"extractions": {"montant": {"trouve": True, "valeur": "100 EUR"}}},
        sources=[SourceOutil(doc_id="doc-a", source="a.pdf", nom_fichier="a.pdf", page=1, extrait="x")],
    )
    fabrique_extract, _ = _outil_extract_capture(resultat_extract)

    session = construire_session(
        "Extrais le montant du document A.",
        llm=_LLMChamps(["montant"]),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_extract],
    )
    reponse = _invoquer(session)

    assert isinstance(reponse, ResultatOutil)
    assert reponse.succes is True
    assert reponse.donnees["extractions"]["montant"]["valeur"] == "100 EUR"
    assert len(reponse.sources) == 1
