"""
Tests des nœuds du graphe agentique.

Comme pour `tests/agent/test_session.py`, aucun test ici n'exige de serveur
Ollama ni de collection Qdrant : le LLM est une doublure et l'outil `search`
est une fabrique factice enregistrée à la place de la vraie
(`src.tools.search`). La fabrique factice peuple `ContexteOutil.dernier_
rapport_recherche` exactement comme le fait `src.tools.search._executer_search`,
avec de vrais `Passage`/`RapportRecherche` (src.rag.retrieval) — pas
seulement des `SourceOutil` — puisque c'est ce rapport que `noeud_evaluer_
preuves` lit désormais, et que `noeud_generer_reponse` doit réutiliser tel
quel (voir la section "double retrieval" ci-dessous).

`noeud_generer_reponse` est testé en isolant
`src.rag.generation.generer_depuis_recherche` (monkeypatch), puisque ce
nœud délègue entièrement à cette fonction et ne doit donc jamais l'appeler
réellement ici — ni, surtout, jamais appeler une seconde recherche.

Les scores factices ci-dessous encadrent volontairement
`nodes.SEUIL_PERTINENCE_MINIMALE` (0.15) : voir le commentaire de calibrage
dans `src/agent/nodes.py` pour les scores réels observés (reranker
BAAI/bge-reranker-v2-m3) qui justifient ce seuil.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from src.agent import nodes
from src.agent.graph_state import EtatGraphe
from src.agent.session import construire_session
from src.rag.generation import ReponseRAG
from src.rag.retrieval import Passage, PerimetreDocumentaire, RapportRecherche
from src.tools.base import DefinitionOutil, ResultatOutil, SourceOutil

# Au-dessus du seuil (0.15) : cas net observé en smoke test réel (0.31 à 0.9991).
_SCORE_EVIDENCE_FORTE = 0.90
# Sous le seuil mais non nul : c'est exactement le cas qui échappait à
# l'ancienne heuristique (nombre_preuves >= 1) — un passage existe, mais son
# score de reranking le désigne comme non pertinent (0.01 à 0.09 observés en
# smoke test réel sur des requêtes hors-corpus).
_SCORE_EVIDENCE_FAIBLE = 0.03


class _ArgsSearchFactice(BaseModel):
    requete: str = Field(default="", description="Requête factice.")


def _un_passage(score: float, citation: str = "S1") -> Passage:
    return Passage(
        citation=citation,
        rang=1,
        point_id="p1",
        doc_id="d1",
        chunk_index=0,
        texte="Un extrait.",
        source="doc.pdf",
        nom_fichier="doc.pdf",
        page=1,
        categorie="",
        score_recherche=score,
        score_reranking=score,
    )


def _un_rapport(passages: list[Passage], requete: str = "peu importe") -> RapportRecherche:
    return RapportRecherche(
        requete=requete,
        profil="generic",
        filtres={},
        passages=passages,
        candidats_recuperes=len(passages),
        reranking_utilise=True,
        seuil_applique=None,
        duree_secondes=0.01,
    )


def _une_source(score: float) -> SourceOutil:
    return SourceOutil(
        doc_id="d1",
        source="doc.pdf",
        nom_fichier="doc.pdf",
        page=1,
        score=score,
        extrait="Un extrait.",
    )


def _outil_search_sequence(scores: Sequence[float | None]) -> Callable[[], DefinitionOutil]:
    """
    Fabrique de recherche factice à scores contrôlés, un par appel successif
    (le dernier est répété si la séquence est épuisée).

    Peuple `contexte.dernier_rapport_recherche` avec un vrai `RapportRecherche`,
    exactement comme le fait `src.tools.search._executer_search` : c'est ce
    champ, pas `contexte.sources`, que `noeud_evaluer_preuves` lit désormais.
    """
    etat_compteur = {"i": 0}

    def _fonction(*, contexte=None, **kw) -> ResultatOutil:
        index = min(etat_compteur["i"], len(scores) - 1)
        etat_compteur["i"] += 1
        score = scores[index]

        rapport = _un_rapport([] if score is None else [_un_passage(score)])
        if contexte is not None:
            contexte.dernier_rapport_recherche = rapport

        if score is None:
            return ResultatOutil(outil="search", succes=True, message="Rien trouvé.")
        return ResultatOutil(
            outil="search",
            succes=True,
            message="1 passage trouvé.",
            sources=[_une_source(score)],
        )

    def _fabrique() -> DefinitionOutil:
        return DefinitionOutil(
            nom="search",
            description="Search factice à scores contrôlés.",
            schema_arguments=_ArgsSearchFactice,
            fonction=_fonction,
        )

    return _fabrique


def _outil_search(*, score: float | None) -> DefinitionOutil:
    """Score fixe, répété à chaque appel."""
    return _outil_search_sequence([score])()


def _outil_search_evidence_forte() -> DefinitionOutil:
    return _outil_search(score=_SCORE_EVIDENCE_FORTE)


def _outil_search_evidence_faible() -> DefinitionOutil:
    return _outil_search(score=_SCORE_EVIDENCE_FAIBLE)


def _outil_search_sans_resultats() -> DefinitionOutil:
    return _outil_search(score=None)


class _LLMNonSollicite:
    """Doublure : ces tests ne doivent jamais invoquer de LLM."""

    def invoke(self, messages: Any):  # pragma: no cover - non sollicité ici
        raise AssertionError("Aucun LLM ne devrait être invoqué dans ce test.")


class _LLMScripte:
    """
    Doublure distinguant les deux usages LLM du graphe (reformulation et
    jugement de suffisance) en inspectant le message système — les deux
    passent par `invoquer_llm` avec des messages système différents — et
    répondant selon des scripts fournis séparément.
    """

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
            return AIMessage(content='{"requete_reformulee": "reformulation"}')

        if "juges si des passages" in systeme:
            self.appels_suffisance += 1
            if self._verdicts:
                return AIMessage(content=self._verdicts.pop(0))
            return AIMessage(content='{"suffisant": false, "raison": "défaut de test"}')

        raise AssertionError(f"Message système inattendu : {systeme[:80]!r}")


def _session_factice(*, fabriques, llm: Any = None, max_tentatives: int = 6):
    return construire_session(
        "Quelles sont les conditions de résiliation ?",
        llm=llm if llm is not None else _LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=fabriques,
        max_tentatives=max_tentatives,
    )


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


# ---------------------------------------------------------------------------
# _detecter_intention (Action 03B)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "requete,attendu",
    [
        ("Résume le rapport CNIL 2023.", "summarize"),
        ("Fais-moi un résumé du rapport TotalEnergies sur le climat.", "summarize"),
        ("Summarize the CNIL annual report.", "summarize"),
        ("Give me a summary of the report please.", "summarize"),
        ("Quels sont les points clés du rapport ?", "summarize"),
        ("What are the key points of this document?", "summarize"),
        ("Fais une synthèse des engagements climatiques.", "summarize"),
        ("Peux-tu me faire un sommaire du document ?", "summarize"),
        ("tldr this report please", "summarize"),
        ("Who is the CEO of company X?", "search"),
        ("Quelle est la politique de confidentialité de l'entreprise ?", "search"),
        ("What were the total emissions in 2022?", "search"),
        ("Quel est le montant de la sanction infligée à Criteo ?", "search"),
        ("", "search"),
    ],
)
def test_detecter_intention(requete: str, attendu: str) -> None:
    assert nodes._detecter_intention(requete) == attendu


def test_detecter_intention_insensible_a_la_casse_et_aux_accents() -> None:
    assert nodes._detecter_intention("RÉSUMÉ DU RAPPORT") == "summarize"
    assert nodes._detecter_intention("resume du rapport") == "summarize"


# ---------------------------------------------------------------------------
# noeud_detecter_intention / router_intention
# ---------------------------------------------------------------------------


def test_noeud_detecter_intention_journalise_la_trace() -> None:
    session = _session_factice(fabriques=[_outil_search_evidence_forte])
    session.etat.requete_courante = "Résume le rapport CNIL 2023."
    etat = EtatGraphe(session=session)

    mise_a_jour = nodes.noeud_detecter_intention(etat)

    assert mise_a_jour["intention"] == "summarize"
    assert session.etat.trace[-1].nom == "intention"
    assert session.etat.trace[-1].donnees["intention"] == "summarize"


def test_noeud_detecter_intention_search_par_defaut() -> None:
    session = _session_factice(fabriques=[_outil_search_evidence_forte])
    etat = EtatGraphe(session=session)

    mise_a_jour = nodes.noeud_detecter_intention(etat)

    assert mise_a_jour["intention"] == "search"


def test_router_intention_lit_le_champ_deja_calcule() -> None:
    session = _session_factice(fabriques=[_outil_search_evidence_forte])

    assert nodes.router_intention(EtatGraphe(session=session, intention="summarize")) == "summarize"
    assert nodes.router_intention(EtatGraphe(session=session, intention="search")) == "rechercher"
    # Repli sûr : toute valeur inattendue (y compris None) route vers l'existant.
    assert nodes.router_intention(EtatGraphe(session=session, intention=None)) == "rechercher"


# ---------------------------------------------------------------------------
# noeud_summarize
# ---------------------------------------------------------------------------


def test_noeud_summarize_transmet_les_documents_resolus(monkeypatch) -> None:
    perimetre = PerimetreDocumentaire(
        statut="exact", valeurs_filtre=("doc-cnil",), libelles=("CNIL 2023",)
    )
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    resultat = ResultatOutil(outil="summarize", succes=True, message="Résumé produit.")
    fabrique, appels = _outil_summarize_capture(resultat)

    session = construire_session(
        "Résume le rapport CNIL 2023.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique],
    )
    etat = EtatGraphe(session=session)

    mise_a_jour = nodes.noeud_summarize(etat)

    assert appels == [{"objectif": "Résume le rapport CNIL 2023.", "documents": ["doc-cnil"]}]
    assert mise_a_jour["reponse"] is resultat
    # Passe bien par le registre : le résultat est consigné dans le contexte.
    assert session.contexte.resultats[-1].outil == "summarize"
    assert session.etat.trace[-1].nom == "summarize"
    assert session.etat.trace[-1].donnees["resolution_documentaire"] == "exact"


def test_noeud_summarize_sans_document_resolu_passe_documents_none(monkeypatch) -> None:
    perimetre = PerimetreDocumentaire(statut="aucun", raison="aucune_correspondance")
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    resultat = ResultatOutil(outil="summarize", succes=True, message="Résumé produit.")
    fabrique, appels = _outil_summarize_capture(resultat)

    session = construire_session(
        "Résume les éléments trouvés.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique],
    )
    etat = EtatGraphe(session=session)

    nodes.noeud_summarize(etat)

    assert appels == [{"objectif": "Résume les éléments trouvés.", "documents": None}]


def test_noeud_summarize_exact_appelle_summarize_une_fois_avec_un_document(monkeypatch) -> None:
    perimetre = PerimetreDocumentaire(
        statut="exact", valeurs_filtre=("doc-unique",), libelles=("Rapport unique",)
    )
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    resultat = ResultatOutil(outil="summarize", succes=True, message="Résumé produit.")
    fabrique, appels = _outil_summarize_capture(resultat)

    session = construire_session(
        "Résume le document rapport_unique.pdf.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique],
    )
    nodes.noeud_summarize(EtatGraphe(session=session))

    assert appels == [
        {"objectif": "Résume le document rapport_unique.pdf.", "documents": ["doc-unique"]}
    ]
    assert session.etat.trace[-1].donnees["mode"] == "document_complet"


def test_noeud_summarize_ambiguite_refuse_sans_appeler_summarize(monkeypatch) -> None:
    perimetre = PerimetreDocumentaire(
        statut="ambigu", raison="marge_insuffisante", libelles=("Rapport A", "Rapport B")
    )
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    resultat = ResultatOutil(outil="summarize", succes=False, message="ne doit pas être appelé")
    fabrique, appels = _outil_summarize_capture(resultat)

    session = construire_session(
        "Résume le rapport.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique],
    )
    mise_a_jour = nodes.noeud_summarize(EtatGraphe(session=session))

    # L'outil summarize N'EST PAS appelé : refus déterministe construit dans le nœud.
    assert appels == []
    assert mise_a_jour["reponse"].succes is False
    assert "Rapport A" in mise_a_jour["reponse"].message
    assert session.etat.trace[-1].donnees["mode"] == "document_vise_non_resolu"


def test_noeud_summarize_compatible_multi_documents_refuse_sans_appeler_summarize(monkeypatch) -> None:
    perimetre = PerimetreDocumentaire(
        statut="compatible",
        valeurs_filtre=("doc-a", "doc-b", "doc-c"),
        libelles=("Doc A", "Doc B", "Doc C"),
    )
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    resultat = ResultatOutil(outil="summarize", succes=False, message="ne doit pas être appelé")
    fabrique, appels = _outil_summarize_capture(resultat)

    session = construire_session(
        "Résume le document cquae_doc_219.txt.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique],
    )
    mise_a_jour = nodes.noeud_summarize(EtatGraphe(session=session))

    assert appels == []  # jamais les 3 documents envoyés au map-reduce
    assert mise_a_jour["reponse"].succes is False
    assert session.etat.trace[-1].donnees["mode"] == "document_vise_non_resolu"


def test_noeud_summarize_resolution_en_echec_ne_casse_pas_le_graphe(monkeypatch) -> None:
    def _explose(requete: str):
        raise RuntimeError("collection indisponible")

    monkeypatch.setattr(nodes, "resoudre_document", _explose)

    resultat = ResultatOutil(outil="summarize", succes=True, message="Résumé produit.")
    fabrique, appels = _outil_summarize_capture(resultat)

    session = construire_session(
        "Résume le rapport CNIL 2023.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique],
    )
    etat = EtatGraphe(session=session)

    mise_a_jour = nodes.noeud_summarize(etat)

    assert appels[0]["documents"] is None
    assert mise_a_jour["reponse"] is resultat
    assert session.etat.trace[-1].donnees["resolution_documentaire"] == "erreur"


# ---------------------------------------------------------------------------
# _detecter_intention (extension CLASSIFY) / _desambiguiser_intention_classify
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "requete,attendu",
    [
        ("Classe ce document.", "classify"),
        ("Classe le rapport CNIL 2023.", "classify"),
        ("Classify this document.", "classify"),
        ("Categorize the CNIL report.", "classify"),
        ("Peux-tu classifier ce rapport ?", "classify"),
        # Zone grise : nom seul, résolue par le classifieur LLM borné, pas ici.
        ("Quelle catégorie correspond à ce rapport ?", "ambigu_classify"),
        ("Quelle est la classification de risque mentionnée dans ce document ?", "ambigu_classify"),
        ("What is the classification mentioned in this document?", "ambigu_classify"),
    ],
)
def test_detecter_intention_classify(requete: str, attendu: str) -> None:
    assert nodes._detecter_intention(requete) == attendu


class _LLMDesambiguisation:
    def __init__(self, intention: str) -> None:
        self._intention = intention

    def invoke(self, messages: Any) -> AIMessage:
        return AIMessage(content=f'{{"intention": "{self._intention}"}}')


def test_desambiguiser_intention_classify_retourne_classify() -> None:
    llm = _LLMDesambiguisation("CLASSIFY")
    assert nodes._desambiguiser_intention_classify(llm, "Quelle catégorie correspond à ce rapport ?") == "classify"


def test_desambiguiser_intention_classify_retourne_search_par_defaut() -> None:
    llm = _LLMDesambiguisation("SEARCH")
    assert nodes._desambiguiser_intention_classify(llm, "Quelle classification de risque est mentionnée ?") == "search"


def test_desambiguiser_intention_classify_repli_sur_erreur_llm() -> None:
    class _LLMExplose:
        def invoke(self, messages):
            raise RuntimeError("Ollama injoignable.")

    assert nodes._desambiguiser_intention_classify(_LLMExplose(), "peu importe") == "search"


def test_router_intention_route_vers_classify() -> None:
    session = _session_factice(fabriques=[_outil_search_evidence_forte])
    assert nodes.router_intention(EtatGraphe(session=session, intention="classify")) == "classify"


# ---------------------------------------------------------------------------
# _detecter_intention (extension EXTRACT, Action 04) /
# _desambiguiser_intention_search_extract / _parser_champs_extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "requete,attendu",
    [
        # EXTRACT explicite (verbe sûr) — mono-champ.
        ("Extrais la date de signature.", "extract"),
        ("Extract the contract duration.", "extract"),
        # EXTRACT explicite — multi-champs (le verbe tranche avant même
        # d'atteindre le signal d'énumération).
        ("Extrais le fournisseur, la date et le montant.", "extract"),
        ("Extract the invoice number, supplier and total amount.", "extract"),
        # Motif structurel "champ : ?" — sûr, indépendant du vocabulaire.
        ("Fournisseur : ? Date : ? Montant : ?", "extract"),
        # EXTRACT implicite (énumération sans verbe) — zone grise.
        ("Donne-moi le fournisseur, la date et le montant total.", "ambigu_search_extract"),
        ("Quels sont le numéro de facture, le fournisseur et la date d'échéance ?", "ambigu_search_extract"),
        ("Je veux les parties, la date de signature et la durée du contrat.", "ambigu_search_extract"),
        ("What are the invoice number, supplier and total amount?", "ambigu_search_extract"),
        ("Give me the parties, signing date and contract duration.", "ambigu_search_extract"),
        # QA simple mono-information — reste SEARCH directement, sans LLM.
        ("Quel est le montant total ?", "search"),
        ("Qui a signé le contrat ?", "search"),
        ("What is the invoice date?", "search"),
        # Question analytique avec plusieurs valeurs — passe par la zone
        # grise (le signal "et" seul ne suffit pas à décider ; c'est
        # exactement le rôle du classifieur borné, testé plus bas).
        ("Compare le chiffre d'affaires 2024 et 2025", "ambigu_search_extract"),
        # Mot "extraction" présent mais intention réellement QA (nom, pas
        # verbe) : ne doit jamais forcer EXTRACT tant qu'aucun signal
        # d'énumération n'est présent.
        (
            "Quelle est la méthode d'extraction des données utilisée dans ce rapport ?",
            "search",
        ),
    ],
)
def test_detecter_intention_extract(requete: str, attendu: str) -> None:
    assert nodes._detecter_intention(requete) == attendu


def test_desambiguiser_intention_extract_retourne_extract() -> None:
    llm = _LLMDesambiguisation("EXTRACT")
    assert (
        nodes._desambiguiser_intention_search_extract(
            llm, "Donne-moi le fournisseur, la date et le montant total."
        )
        == "extract"
    )


def test_desambiguiser_intention_extract_retourne_search_pour_question_analytique() -> None:
    llm = _LLMDesambiguisation("SEARCH")
    assert (
        nodes._desambiguiser_intention_search_extract(llm, "Compare le chiffre d'affaires 2024 et 2025")
        == "search"
    )


def test_desambiguiser_intention_extract_repli_sur_erreur_llm() -> None:
    class _LLMExplose:
        def invoke(self, messages):
            raise RuntimeError("Ollama injoignable.")

    assert nodes._desambiguiser_intention_search_extract(_LLMExplose(), "peu importe") == "search"


def test_router_intention_route_vers_extract() -> None:
    session = _session_factice(fabriques=[_outil_search_evidence_forte])
    assert nodes.router_intention(EtatGraphe(session=session, intention="extract")) == "extract"


# ---------------------------------------------------------------------------
# P1.2 — extensions de vocabulaire de routage (bande B du banc de routage)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "requete,attendu",
    [
        # SUMMARIZE — demandes indirectes (RT-028, RT-031..RT-034).
        ("TL;DR de document_b.txt ?", "summarize"),
        ("tl;dr", "summarize"),
        ("Donne-moi les points essentiels du document rapport_alpha.pdf.", "summarize"),
        ("Quelles sont les idées principales de ce rapport ?", "summarize"),
        ("Give me the main takeaways from report_alpha.pdf.", "summarize"),
        ("Quels sont les grands axes de essai_philo.pdf ?", "summarize"),
        # CLASSIFY sûr — type / nature DU document (RT-038, RT-039, RT-041).
        ("Quel est le type de ce document ?", "classify"),
        ("Is this document a contract or a report?", "classify"),
        ("Détermine la nature de contrat_2025.pdf.", "classify"),
        # EXTRACT — verbe « récupère » + énumération (RT-048, RT-049).
        (
            "Récupère les champs fournisseur, date et montant dans facture_2025.pdf.",
            "extract",
        ),
        ("Récupère le numéro de contrat et la date d'effet.", "extract"),
    ],
)
def test_detecter_intention_extensions_p1_2(requete: str, attendu: str) -> None:
    assert nodes._detecter_intention(requete) == attendu


def test_detecter_intention_p1_2_sagit_il_reste_une_zone_grise() -> None:
    # RT-040 : « s'agit-il d'un X ou d'un Y ? » n'est pas forcé lexicalement
    # (risque de faux positif sur une question factuelle) — il passe par le
    # désambiguïsateur borné.
    assert (
        nodes._detecter_intention("S'agit-il d'une facture ou d'un devis ?")
        == nodes._AMBIGU_CLASSIFY
    )


@pytest.mark.parametrize(
    "requete",
    [
        # Anti-faux-positifs explicitement listés pour P1.2 : ces questions
        # factuelles ne doivent JAMAIS basculer vers summarize / classify /
        # extract à cause des nouvelles expressions.
        "Quels sont les points de contrôle mentionnés dans rapport_alpha.pdf ?",  # RT-015
        "Combien de points sont listés dans ce document ?",  # RT-016
        "Compare les deux méthodes décrites dans rapport_alpha.pdf.",  # RT-017
        "Quels sont les points communs entre les deux approches présentées dans ce document ?",  # RT-018
        "Quelle méthode d'extraction des logs est utilisée dans ce rapport ?",  # RT-020
    ],
)
def test_detecter_intention_p1_2_anti_faux_positifs_restent_search(requete: str) -> None:
    assert nodes._detecter_intention(requete) == "search"


def test_detecter_intention_p1_2_classification_de_contenu_reste_zone_grise() -> None:
    # RT-019 : « classification » nu = fait potentiel du contenu -> zone grise
    # (repli déterministe = search). Les expressions P1.2 ne doivent pas la
    # transformer en classify ferme.
    assert (
        nodes._detecter_intention(
            "Quelle classification de risque est indiquée dans rapport_alpha.pdf ?"
        )
        == nodes._AMBIGU_CLASSIFY
    )


def test_detecter_intention_p1_2_recupere_seul_ne_force_pas_extract() -> None:
    # « récupère » sans énumération de champs ne doit pas forcer EXTRACT.
    assert nodes._detecter_intention("Récupère le rapport_alpha.pdf.") == "search"


class _LLMChamps:
    def __init__(self, champs: list[str]) -> None:
        self._champs = champs

    def invoke(self, messages: Any) -> AIMessage:
        return AIMessage(content=json.dumps({"champs": self._champs}))


def test_parser_champs_extraction_retourne_les_champs_demandes() -> None:
    llm = _LLMChamps(["fournisseur", "date", "montant total"])
    assert nodes._parser_champs_extraction(llm, "peu importe") == [
        "fournisseur",
        "date",
        "montant total",
    ]


def test_parser_champs_extraction_deduplique_et_nettoie() -> None:
    llm = _LLMChamps(["  fournisseur ", "fournisseur", "date"])
    assert nodes._parser_champs_extraction(llm, "peu importe") == ["fournisseur", "date"]


def test_parser_champs_extraction_repli_liste_vide_sur_erreur_llm() -> None:
    class _LLMExplose:
        def invoke(self, messages):
            raise RuntimeError("Ollama injoignable.")

    assert nodes._parser_champs_extraction(_LLMExplose(), "peu importe") == []


def test_parser_champs_extraction_repli_liste_vide_sur_json_invalide() -> None:
    class _LLMJSONInvalide:
        def invoke(self, messages):
            return AIMessage(content="pas du JSON {{{")

    assert nodes._parser_champs_extraction(_LLMJSONInvalide(), "peu importe") == []


# ---------------------------------------------------------------------------
# noeud_classify
# ---------------------------------------------------------------------------


class _ArgsClassifyFactice(BaseModel):
    categories: list[str] = Field(default_factory=list)
    document: str | None = Field(default=None)
    critere: str | None = Field(default=None)
    instruction: str | None = Field(default=None)


def _outil_classify_capture(
    resultat: ResultatOutil,
) -> tuple[Callable[[], DefinitionOutil], list[dict[str, Any]]]:
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


def _outil_search_avec_compteur(
    score: float | None,
) -> tuple[Callable[[], DefinitionOutil], list[int]]:
    """Comme `_outil_search_sequence`, mais renvoie aussi un compteur d'appels."""
    compteur: list[int] = [0]

    def _fonction(*, contexte=None, **kw) -> ResultatOutil:
        compteur[0] += 1
        rapport = _un_rapport([] if score is None else [_un_passage(score)])
        if contexte is not None:
            contexte.dernier_rapport_recherche = rapport
        if score is None:
            return ResultatOutil(outil="search", succes=True, message="Rien trouvé.")
        return ResultatOutil(
            outil="search", succes=True, message="1 passage trouvé.", sources=[_une_source(score)]
        )

    def _fabrique() -> DefinitionOutil:
        return DefinitionOutil(
            nom="search",
            description="Search factice à compteur.",
            schema_arguments=_ArgsSearchFactice,
            fonction=_fonction,
        )

    return _fabrique, compteur


def test_noeud_classify_document_resolu_utilise_mode_document_complet_sans_search(monkeypatch) -> None:
    """
    Périmètre résolu de façon non ambiguë (un seul document) : classify passe
    en mode document complet (Option E — voir `src.tools.classify`) et
    n'exécute plus aucun search interne, contrairement à l'ancien
    comportement (top-k) que cette mission remplace pour ce cas précis.
    """
    perimetre = PerimetreDocumentaire(
        statut="exact", valeurs_filtre=("cnil-44e-rapport-annuel-2023.pdf",), libelles=("CNIL 2023",)
    )
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    resultat_classify = ResultatOutil(outil="classify", succes=True, message="ok")
    fabrique_classify, appels_classify = _outil_classify_capture(resultat_classify)
    fabrique_search, compteur_recherche = _outil_search_avec_compteur(_SCORE_EVIDENCE_FORTE)

    session = construire_session(
        "Classe le rapport CNIL 2023.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_classify],
    )
    etat = EtatGraphe(session=session)

    mise_a_jour = nodes.noeud_classify(etat)

    assert appels_classify[0]["documents"] == ["cnil-44e-rapport-annuel-2023.pdf"]
    assert "document" not in appels_classify[0]
    assert appels_classify[0]["categories"]  # non vide, vient du profil technique
    assert mise_a_jour["reponse"] is resultat_classify
    # Document résolu de façon fiable : plus besoin d'un search interne.
    assert compteur_recherche[0] == 0


def test_noeud_classify_ne_relance_pas_search_si_sources_deja_presentes(monkeypatch) -> None:
    monkeypatch.setattr(
        nodes, "resoudre_document", lambda requete: PerimetreDocumentaire(statut="aucun", raison="aucune")
    )

    resultat_classify = ResultatOutil(outil="classify", succes=True, message="ok")
    fabrique_classify, appels_classify = _outil_classify_capture(resultat_classify)
    fabrique_search, compteur_recherche = _outil_search_avec_compteur(_SCORE_EVIDENCE_FORTE)

    session = construire_session(
        "Classe ce document.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_classify],
    )
    session.contexte.sources.append(_une_source(0.9))
    etat = EtatGraphe(session=session)

    nodes.noeud_classify(etat)

    assert compteur_recherche[0] == 0
    assert appels_classify[0]["document"] is None


def test_noeud_classify_document_non_ambigu_refuse_sans_appeler_loutil(monkeypatch) -> None:
    """
    Périmètre 'compatible' à plusieurs valeurs : la requête vise
    explicitement un document, mais la résolution ne peut pas trancher entre
    plusieurs candidats également valables. `classify` n'est PAS appelé (le
    refus est construit directement par le nœud, aucun choix implicite), et
    aucun `search` de repli n'est déclenché.
    """
    perimetre = PerimetreDocumentaire(
        statut="compatible", valeurs_filtre=("doc-a", "doc-b"), libelles=("Doc A", "Doc B")
    )
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    resultat_classify = ResultatOutil(outil="classify", succes=True, message="ne devrait jamais être appelé")
    fabrique_classify, appels_classify = _outil_classify_capture(resultat_classify)
    fabrique_search, compteur_recherche = _outil_search_avec_compteur(_SCORE_EVIDENCE_FORTE)

    session = construire_session(
        "Classe les rapports A et B.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_classify],
    )
    etat = EtatGraphe(session=session)

    mise_a_jour = nodes.noeud_classify(etat)

    assert appels_classify == []
    assert compteur_recherche[0] == 0
    assert mise_a_jour["reponse"].succes is False
    assert "Doc A" in mise_a_jour["reponse"].message and "Doc B" in mise_a_jour["reponse"].message


def test_noeud_classify_document_ambigu_refuse_sans_search(monkeypatch) -> None:
    """Périmètre 'ambigu' (candidats trop proches) : refus propre, aucun search."""
    perimetre = PerimetreDocumentaire(
        statut="ambigu", raison="marge_insuffisante", libelles=("Rapport A", "Rapport B")
    )
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    resultat_classify = ResultatOutil(outil="classify", succes=True, message="ne devrait jamais être appelé")
    fabrique_classify, appels_classify = _outil_classify_capture(resultat_classify)
    fabrique_search, compteur_recherche = _outil_search_avec_compteur(_SCORE_EVIDENCE_FORTE)

    session = construire_session(
        "Classe le rapport.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_classify],
    )
    etat = EtatGraphe(session=session)

    mise_a_jour = nodes.noeud_classify(etat)

    assert appels_classify == []
    assert compteur_recherche[0] == 0
    assert mise_a_jour["reponse"].succes is False


def test_noeud_classify_document_introuvable_refuse_sans_search(monkeypatch) -> None:
    """
    Correspondance détectée mais sous le seuil de résolution
    (`raison="score_insuffisant"`) : la requête semble viser un document
    précis (introuvable de façon fiable), refus propre, aucun search de
    repli — distinct du cas où aucune référence documentaire n'est présente
    du tout (voir `test_noeud_classify_ne_relance_pas_search_si_sources_deja_presentes`).
    """
    perimetre = PerimetreDocumentaire(statut="aucun", raison="score_insuffisant", score=0.05)
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    resultat_classify = ResultatOutil(outil="classify", succes=True, message="ne devrait jamais être appelé")
    fabrique_classify, appels_classify = _outil_classify_capture(resultat_classify)
    fabrique_search, compteur_recherche = _outil_search_avec_compteur(_SCORE_EVIDENCE_FORTE)

    session = construire_session(
        "Classe le rapport Inexistant-XYZ.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_classify],
    )
    etat = EtatGraphe(session=session)

    mise_a_jour = nodes.noeud_classify(etat)

    assert appels_classify == []
    assert compteur_recherche[0] == 0
    assert mise_a_jour["reponse"].succes is False


def test_noeud_classify_passe_par_le_registre_et_journalise(monkeypatch) -> None:
    monkeypatch.setattr(
        nodes, "resoudre_document", lambda requete: PerimetreDocumentaire(statut="aucun", raison="aucune")
    )

    resultat_classify = ResultatOutil(outil="classify", succes=True, message="ok")
    fabrique_classify, _ = _outil_classify_capture(resultat_classify)
    fabrique_search, _ = _outil_search_avec_compteur(_SCORE_EVIDENCE_FORTE)

    session = construire_session(
        "Classe ce document.",
        llm=_LLMNonSollicite(),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_classify],
    )
    etat = EtatGraphe(session=session)

    nodes.noeud_classify(etat)

    assert session.contexte.resultats[-1].outil == "classify"
    assert session.etat.trace[-1].nom == "classify"
    assert session.etat.trace[-1].donnees["succes"] is True


# ---------------------------------------------------------------------------
# noeud_extract (Action 04)
# ---------------------------------------------------------------------------


class _ArgsExtractFactice(BaseModel):
    champs: list[str] = Field(default_factory=list)
    document: str | None = Field(default=None)
    documents: list[str] | None = Field(default=None)
    instruction: str | None = Field(default=None)


def _outil_extract_capture(
    resultat: ResultatOutil,
) -> tuple[Callable[[], DefinitionOutil], list[dict[str, Any]]]:
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


def test_noeud_extract_document_resolu_utilise_mode_document_complet_sans_search(monkeypatch) -> None:
    """
    Périmètre résolu de façon non ambiguë (un seul document) : extract passe
    en mode document complet, sans search interne — même routage que
    `noeud_classify`, réutilisé délibérément (voir docstring de `noeud_extract`).
    """
    perimetre = PerimetreDocumentaire(
        statut="exact", valeurs_filtre=("cnil-44e-rapport-annuel-2023.pdf",), libelles=("CNIL 2023",)
    )
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    resultat_extract = ResultatOutil(outil="extract", succes=True, message="ok")
    fabrique_extract, appels_extract = _outil_extract_capture(resultat_extract)
    fabrique_search, compteur_recherche = _outil_search_avec_compteur(_SCORE_EVIDENCE_FORTE)

    session = construire_session(
        "Extrais la date et l'émetteur du rapport CNIL 2023.",
        llm=_LLMChamps(["date", "émetteur"]),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_extract],
    )
    etat = EtatGraphe(session=session)

    mise_a_jour = nodes.noeud_extract(etat)

    assert appels_extract[0]["documents"] == ["cnil-44e-rapport-annuel-2023.pdf"]
    assert "document" not in appels_extract[0]
    assert appels_extract[0]["champs"] == ["date", "émetteur"]
    assert mise_a_jour["reponse"] is resultat_extract
    assert compteur_recherche[0] == 0


def test_noeud_extract_document_ambigu_refuse_sans_appeler_loutil(monkeypatch) -> None:
    """Périmètre 'compatible' à plusieurs valeurs : refus propre, aucun appel à extract ni search."""
    perimetre = PerimetreDocumentaire(
        statut="compatible", valeurs_filtre=("doc-a", "doc-b"), libelles=("Doc A", "Doc B")
    )
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    resultat_extract = ResultatOutil(outil="extract", succes=True, message="ne devrait jamais être appelé")
    fabrique_extract, appels_extract = _outil_extract_capture(resultat_extract)
    fabrique_search, compteur_recherche = _outil_search_avec_compteur(_SCORE_EVIDENCE_FORTE)

    session = construire_session(
        "Donne-moi le montant des documents A et B.",
        llm=_LLMChamps(["montant"]),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_extract],
    )
    etat = EtatGraphe(session=session)

    mise_a_jour = nodes.noeud_extract(etat)

    assert appels_extract == []
    assert compteur_recherche[0] == 0
    assert mise_a_jour["reponse"].succes is False
    assert "Doc A" in mise_a_jour["reponse"].message and "Doc B" in mise_a_jour["reponse"].message


def test_noeud_extract_document_introuvable_refuse_sans_search(monkeypatch) -> None:
    """Correspondance détectée mais sous le seuil de résolution : refus propre, aucun search."""
    perimetre = PerimetreDocumentaire(statut="aucun", raison="score_insuffisant", score=0.05)
    monkeypatch.setattr(nodes, "resoudre_document", lambda requete: perimetre)

    resultat_extract = ResultatOutil(outil="extract", succes=True, message="ne devrait jamais être appelé")
    fabrique_extract, appels_extract = _outil_extract_capture(resultat_extract)
    fabrique_search, compteur_recherche = _outil_search_avec_compteur(_SCORE_EVIDENCE_FORTE)

    session = construire_session(
        "Donne-moi le montant du rapport Inexistant-XYZ.",
        llm=_LLMChamps(["montant"]),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_extract],
    )
    etat = EtatGraphe(session=session)

    mise_a_jour = nodes.noeud_extract(etat)

    assert appels_extract == []
    assert compteur_recherche[0] == 0
    assert mise_a_jour["reponse"].succes is False


def test_noeud_extract_aucun_document_fiable_refuse_sans_search_ni_extract(monkeypatch) -> None:
    """
    P1.6 — aucune référence documentaire fiable (`statut="aucun"`,
    `raison != "score_insuffisant"`) : refus déterministe. Ni `search` global,
    ni `extract` : EXTRACT ne fabrique pas de périmètre à partir d'un top-k.
    """
    monkeypatch.setattr(
        nodes,
        "resoudre_document",
        lambda requete: PerimetreDocumentaire(statut="aucun", raison="aucune_correspondance"),
    )

    resultat_extract = ResultatOutil(outil="extract", succes=True, message="ne devrait jamais être appelé")
    fabrique_extract, appels_extract = _outil_extract_capture(resultat_extract)
    fabrique_search, compteur_recherche = _outil_search_avec_compteur(_SCORE_EVIDENCE_FORTE)

    session = construire_session(
        "Donne-moi le montant.",
        llm=_LLMChamps(["montant"]),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_extract],
    )
    etat = EtatGraphe(session=session)

    mise_a_jour = nodes.noeud_extract(etat)

    assert appels_extract == []
    assert compteur_recherche[0] == 0
    assert mise_a_jour["reponse"].succes is False
    assert "Précise le document" in mise_a_jour["reponse"].message


def test_noeud_extract_ne_choisit_jamais_un_document_depuis_un_top_k(monkeypatch) -> None:
    """
    Même sans document nommé, des sources déjà présentes dans le contexte (un
    éventuel top-k d'un search antérieur) ne doivent JAMAIS servir de
    périmètre implicite : refus, aucun `search`, aucun `extract`.
    """
    monkeypatch.setattr(
        nodes,
        "resoudre_document",
        lambda requete: PerimetreDocumentaire(statut="aucun", raison="aucune_correspondance"),
    )

    resultat_extract = ResultatOutil(outil="extract", succes=True, message="ne devrait jamais être appelé")
    fabrique_extract, appels_extract = _outil_extract_capture(resultat_extract)
    fabrique_search, compteur_recherche = _outil_search_avec_compteur(_SCORE_EVIDENCE_FORTE)

    session = construire_session(
        "Donne-moi le montant.",
        llm=_LLMChamps(["montant"]),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_extract],
    )
    session.contexte.sources.append(_une_source(0.9))
    etat = EtatGraphe(session=session)

    mise_a_jour = nodes.noeud_extract(etat)

    assert appels_extract == []
    assert compteur_recherche[0] == 0
    assert mise_a_jour["reponse"].succes is False


def test_noeud_extract_journalise_le_refus_aucun_document_fiable(monkeypatch) -> None:
    monkeypatch.setattr(
        nodes,
        "resoudre_document",
        lambda requete: PerimetreDocumentaire(statut="aucun", raison="aucune_correspondance"),
    )

    resultat_extract = ResultatOutil(outil="extract", succes=True, message="ne devrait jamais être appelé")
    fabrique_extract, _ = _outil_extract_capture(resultat_extract)
    fabrique_search, _ = _outil_search_avec_compteur(_SCORE_EVIDENCE_FORTE)

    session = construire_session(
        "Donne-moi le montant.",
        llm=_LLMChamps(["montant"]),
        charger_profil_domaine=False,
        fabriques=[fabrique_search, fabrique_extract],
    )
    etat = EtatGraphe(session=session)

    nodes.noeud_extract(etat)

    assert session.contexte.resultats[-1].outil == "extract"
    assert session.etat.trace[-1].nom == "extract"
    assert session.etat.trace[-1].donnees["succes"] is False
    assert session.etat.trace[-1].donnees["mode"] == "aucun_document_fiable"
    assert session.etat.trace[-1].donnees["champs_demandes"] == ["montant"]


# ---------------------------------------------------------------------------
# noeud_rechercher
# ---------------------------------------------------------------------------


def test_rechercher_consomme_une_tentative_et_accumule_les_sources():
    session = _session_factice(fabriques=[_outil_search_evidence_forte])
    etat = EtatGraphe(session=session)

    mise_a_jour = nodes.noeud_rechercher(etat)

    assert mise_a_jour["session"] is session
    assert session.etat.tentatives == 1
    assert session.nombre_preuves == 1
    assert "outil" in session.etat.noms_etapes()
    assert session.contexte.dernier_rapport_recherche is not None


def test_rechercher_leve_si_le_budget_est_deja_epuise():
    from src.agent.state import BudgetTentativesEpuise

    session = _session_factice(fabriques=[_outil_search_evidence_forte], max_tentatives=1)
    session.etat.incrementer_tentative()
    etat = EtatGraphe(session=session)

    with pytest.raises(BudgetTentativesEpuise):
        nodes.noeud_rechercher(etat)


# ---------------------------------------------------------------------------
# noeud_evaluer_preuves : niveau 1 (pertinence) et niveau 2 (suffisance)
# ---------------------------------------------------------------------------


def test_evaluer_preuves_pertinent_et_suffisant():
    llm = _LLMScripte(verdicts_suffisance=['{"suffisant": true, "raison": "ok", "elements_support": ["S1"]}'])
    session = _session_factice(fabriques=[_outil_search_evidence_forte], llm=llm)
    session.executer_outil("search", requete="peu importe")
    etat = EtatGraphe(session=session)

    mise_a_jour = nodes.noeud_evaluer_preuves(etat)

    assert mise_a_jour["preuves_pertinentes"] is True
    assert mise_a_jour["preuves_suffisantes"] is True
    assert llm.appels_suffisance == 1

    noms = session.etat.noms_etapes()
    assert noms[-2:] == ["evaluation_pertinence", "evaluation_suffisance"]
    trace_pertinence = session.etat.trace[-2]
    assert trace_pertinence.donnees["pertinent"] is True
    assert trace_pertinence.donnees["score_pertinence_maximal"] == _SCORE_EVIDENCE_FORTE
    assert trace_pertinence.donnees["seuil_pertinence"] == nodes.SEUIL_PERTINENCE_MINIMALE
    trace_suffisance = session.etat.trace[-1]
    assert trace_suffisance.donnees["suffisant"] is True
    assert trace_suffisance.donnees["elements_support"] == ["S1"]


def test_evaluer_preuves_pertinent_mais_insuffisant():
    """
    Cas central de cette tâche : un passage thématiquement pertinent (score
    au-dessus du seuil) mais qui ne contient pas la valeur précise demandée.
    Le niveau 1 passe, le niveau 2 doit pouvoir le rattraper.
    """
    llm = _LLMScripte(
        verdicts_suffisance=[
            '{"suffisant": false, "raison": "le passage parle du bon sujet mais '
            'ne donne pas la valeur demandée", "elements_support": []}'
        ]
    )
    session = _session_factice(fabriques=[_outil_search_evidence_forte], llm=llm)
    session.executer_outil("search", requete="peu importe")
    etat = EtatGraphe(session=session)

    mise_a_jour = nodes.noeud_evaluer_preuves(etat)

    assert mise_a_jour["preuves_pertinentes"] is True
    assert mise_a_jour["preuves_suffisantes"] is False
    assert "ne donne pas la valeur demandée" in mise_a_jour["raison_insuffisance"]


def test_evaluer_preuves_non_pertinent_ne_declenche_jamais_le_juge_de_suffisance():
    """
    Le niveau 2 ne doit jamais être payé sur un cas déjà tranché par le
    niveau 1 : `_LLMNonSollicite` lève si `.invoke()` est appelé.
    """
    session = _session_factice(fabriques=[_outil_search_evidence_faible])
    session.executer_outil("search", requete="peu importe")
    etat = EtatGraphe(session=session)

    mise_a_jour = nodes.noeud_evaluer_preuves(etat)

    assert mise_a_jour["preuves_pertinentes"] is False
    assert mise_a_jour["preuves_suffisantes"] is None
    derniere_trace = session.etat.trace[-1]
    assert derniere_trace.nom == "evaluation_pertinence"
    assert derniere_trace.donnees["score_pertinence_maximal"] == _SCORE_EVIDENCE_FAIBLE


def test_evaluer_preuves_resultat_vide():
    session = _session_factice(fabriques=[_outil_search_sans_resultats])
    session.executer_outil("search", requete="peu importe")
    etat = EtatGraphe(session=session)

    mise_a_jour = nodes.noeud_evaluer_preuves(etat)

    assert mise_a_jour["preuves_pertinentes"] is False
    assert mise_a_jour["preuves_suffisantes"] is None
    derniere_trace = session.etat.trace[-1]
    assert derniere_trace.donnees["nombre_preuves"] == 0
    assert derniere_trace.donnees["score_pertinence_maximal"] == 0.0


def test_evaluer_preuves_sans_rapport_du_tout_est_traite_comme_non_pertinent():
    """`dernier_rapport_recherche` peut rester `None` (search jamais réussi)."""
    session = _session_factice(fabriques=[_outil_search_evidence_forte])
    etat = EtatGraphe(session=session)  # aucune recherche exécutée

    mise_a_jour = nodes.noeud_evaluer_preuves(etat)

    assert mise_a_jour["preuves_pertinentes"] is False
    assert session.contexte.dernier_rapport_recherche is None


@pytest.mark.parametrize(
    "reponse_llm",
    [
        "ceci n'est pas du JSON",
        '{"raison": "champ suffisant manquant"}',
        '{"suffisant": "oui"}',
    ],
)
def test_juger_suffisance_echec_ou_structure_invalide_retombe_sur_non_suffisant(reponse_llm):
    """
    Le jugement de suffisance ne doit jamais conclure « suffisant » par
    défaut : JSON invalide, champ manquant ou mal typé retombent tous sur un
    verdict conservateur.
    """
    llm = _LLMScripte(verdicts_suffisance=[reponse_llm])
    session = _session_factice(fabriques=[_outil_search_evidence_forte], llm=llm)
    session.executer_outil("search", requete="peu importe")
    etat = EtatGraphe(session=session)

    mise_a_jour = nodes.noeud_evaluer_preuves(etat)

    assert mise_a_jour["preuves_pertinentes"] is True
    assert mise_a_jour["preuves_suffisantes"] is False
    trace_suffisance = session.etat.trace[-1]
    assert trace_suffisance.donnees["origine"] == "repli_technique"


def test_juger_suffisance_llm_indisponible_retombe_sur_non_suffisant():
    class _LLMExplose:
        def invoke(self, messages: Any):
            raise RuntimeError("Ollama injoignable.")

    session = _session_factice(fabriques=[_outil_search_evidence_forte], llm=_LLMExplose())
    session.executer_outil("search", requete="peu importe")
    etat = EtatGraphe(session=session)

    mise_a_jour = nodes.noeud_evaluer_preuves(etat)

    assert mise_a_jour["preuves_suffisantes"] is False


def test_stagnation_detectee_quand_le_score_ne_bouge_pas():
    session = _session_factice(fabriques=[_outil_search_evidence_faible])

    session.executer_outil("search", requete="v1")
    nodes.noeud_evaluer_preuves(EtatGraphe(session=session))  # 1re évaluation : pas de stagnation possible

    session.executer_outil("search", requete="v2")  # même score factice
    mise_a_jour = nodes.noeud_evaluer_preuves(EtatGraphe(session=session))

    assert mise_a_jour["stagnation"] is True


def test_pas_de_stagnation_quand_le_score_progresse():
    session = _session_factice(
        fabriques=[_outil_search_sequence([0.03, 0.12])]
    )

    session.executer_outil("search", requete="v1")
    nodes.noeud_evaluer_preuves(EtatGraphe(session=session))

    session.executer_outil("search", requete="v2")
    mise_a_jour = nodes.noeud_evaluer_preuves(EtatGraphe(session=session))

    assert mise_a_jour["stagnation"] is False


# ---------------------------------------------------------------------------
# router_apres_evaluation
# ---------------------------------------------------------------------------


def test_router_vers_generer_reponse_si_pertinent_et_suffisant():
    session = _session_factice(fabriques=[_outil_search_evidence_forte])
    etat = EtatGraphe(session=session, preuves_pertinentes=True, preuves_suffisantes=True)

    assert nodes.router_apres_evaluation(etat) == "generer_reponse"


def test_router_vers_reformuler_si_pertinent_mais_insuffisant_et_budget_restant():
    session = _session_factice(fabriques=[_outil_search_evidence_forte], max_tentatives=3)
    session.etat.incrementer_tentative()
    etat = EtatGraphe(session=session, preuves_pertinentes=True, preuves_suffisantes=False)

    assert nodes.router_apres_evaluation(etat) == "reformuler"


def test_router_vers_reformuler_si_non_pertinent_et_budget_restant():
    session = _session_factice(fabriques=[_outil_search_evidence_faible], max_tentatives=3)
    session.etat.incrementer_tentative()
    etat = EtatGraphe(session=session, preuves_pertinentes=False, preuves_suffisantes=None)

    assert nodes.router_apres_evaluation(etat) == "reformuler"


def test_router_vers_generer_reponse_si_budget_epuise_meme_sans_preuves():
    session = _session_factice(fabriques=[_outil_search_sans_resultats], max_tentatives=1)
    session.etat.incrementer_tentative()
    etat = EtatGraphe(session=session, preuves_pertinentes=False, preuves_suffisantes=None)

    assert not session.etat.peut_reessayer
    assert nodes.router_apres_evaluation(etat) == "generer_reponse"


def test_router_vers_generer_reponse_si_stagnation_meme_avec_budget_restant():
    session = _session_factice(fabriques=[_outil_search_evidence_faible], max_tentatives=6)
    session.etat.incrementer_tentative()
    etat = EtatGraphe(
        session=session,
        preuves_pertinentes=False,
        preuves_suffisantes=None,
        stagnation=True,
    )

    assert session.etat.peut_reessayer
    assert nodes.router_apres_evaluation(etat) == "generer_reponse"


def test_router_lit_la_decision_deja_calculee_sans_la_recalculer():
    """
    Le router ne doit jamais recalculer un jugement à partir de `session` :
    il lit uniquement les champs de `etat`, pour ne jamais diverger de ce
    que `evaluer_preuves` a journalisé.
    """
    session = _session_factice(fabriques=[_outil_search_evidence_forte], max_tentatives=3)
    session.executer_outil("search", requete="peu importe")  # score fort réel

    etat_incoherent = EtatGraphe(session=session, preuves_pertinentes=False, preuves_suffisantes=None)

    assert nodes.router_apres_evaluation(etat_incoherent) == "reformuler"


# ---------------------------------------------------------------------------
# noeud_reformuler
# ---------------------------------------------------------------------------


class _LLMReformulationValide:
    def invoke(self, messages: Any) -> AIMessage:
        return AIMessage(content='{"requete_reformulee": "conditions résiliation contrat"}')


class _LLMReformulationJSONInvalide:
    def invoke(self, messages: Any) -> AIMessage:
        return AIMessage(content="ceci n'est pas du JSON")


class _LLMIndisponible:
    def invoke(self, messages: Any):
        raise RuntimeError("Ollama injoignable.")


def test_reformuler_met_a_jour_la_requete_courante():
    session = construire_session(
        "Question initiale introuvable",
        llm=_LLMReformulationValide(),
        charger_profil_domaine=False,
        fabriques=[_outil_search_sans_resultats],
    )
    etat = EtatGraphe(session=session)

    nodes.noeud_reformuler(etat)

    assert session.etat.requete_courante == "conditions résiliation contrat"
    assert session.etat.a_ete_reformulee


def test_reformuler_transmet_la_raison_d_insuffisance_au_prompt():
    session = construire_session(
        "Quel est le niveau exact de la métrique X ?",
        llm=_LLMReformulationValide(),
        charger_profil_domaine=False,
        fabriques=[_outil_search_sans_resultats],
    )
    capture: dict[str, Any] = {}
    llm_original = session.llm

    class _LLMCapture:
        def invoke(self, messages: Any):
            capture["utilisateur"] = messages[1].content
            return llm_original.invoke(messages)

    session.llm = _LLMCapture()
    etat = EtatGraphe(
        session=session,
        preuves_pertinentes=True,
        raison_insuffisance="le passage parle du sujet mais pas de la valeur exacte",
    )

    nodes.noeud_reformuler(etat)

    assert "valeur exacte" in capture["utilisateur"]


@pytest.mark.parametrize("faux_llm", [_LLMReformulationJSONInvalide(), _LLMIndisponible()])
def test_reformuler_echec_ne_casse_pas_et_journalise(faux_llm):
    session = construire_session(
        "Question initiale introuvable",
        llm=faux_llm,
        charger_profil_domaine=False,
        fabriques=[_outil_search_sans_resultats],
    )
    requete_avant = session.etat.requete_courante
    etat = EtatGraphe(session=session)

    nodes.noeud_reformuler(etat)

    assert session.etat.requete_courante == requete_avant
    assert session.etat.noms_etapes()[-1] == "reformulation_echec"


# ---------------------------------------------------------------------------
# noeud_generer_reponse
# ---------------------------------------------------------------------------


def _reponse_factice(question: str = "") -> ReponseRAG:
    return ReponseRAG(
        question=question,
        reponse="Réponse factice.",
        profil="generic",
        contexte_suffisant=True,
        citations_valides=True,
        citations_reparees=False,
        sources=[],
    )


def test_generer_reponse_delegue_a_generer_depuis_recherche_avec_le_meme_rapport(monkeypatch):
    """
    Preuve du correctif « double retrieval » : le nœud doit transmettre
    EXACTEMENT l'objet `RapportRecherche` déjà accumulé par `noeud_rechercher`
    (identité d'objet, pas une reconstruction), et ne jamais réexécuter
    l'outil `search`.
    """
    appels: list[dict[str, Any]] = []

    def _fausse_generation(**kwargs: Any) -> ReponseRAG:
        appels.append(kwargs)
        return _reponse_factice(kwargs["question"])

    monkeypatch.setattr(nodes, "generer_depuis_recherche", _fausse_generation)

    session = _session_factice(fabriques=[_outil_search_evidence_forte])
    session.executer_outil("search", requete="peu importe")
    rapport_attendu = session.contexte.dernier_rapport_recherche
    tentatives_avant = session.etat.tentatives

    etat = EtatGraphe(session=session, preuves_pertinentes=True, preuves_suffisantes=True)
    mise_a_jour = nodes.noeud_generer_reponse(etat)

    assert len(appels) == 1
    assert appels[0]["question"] == session.etat.requete_courante
    assert appels[0]["recherche"] is rapport_attendu
    assert mise_a_jour["reponse"] is not None
    assert session.etat.noms_etapes()[-1] == "reponse"
    # Aucune recherche supplémentaire n'a été déclenchée par la génération.
    assert session.etat.tentatives == tentatives_avant


def test_generer_reponse_refuse_sans_llm_si_non_pertinent_et_budget_epuise(monkeypatch):
    """
    Cœur de cette tâche : preuves non pertinentes, budget épuisé -> refus
    déterministe, ZÉRO appel au LLM de génération.
    """

    def _generation_interdite(**kwargs: Any):
        raise AssertionError("generer_depuis_recherche ne doit pas être appelé ici.")

    monkeypatch.setattr(nodes, "generer_depuis_recherche", _generation_interdite)

    session = _session_factice(fabriques=[_outil_search_evidence_faible], max_tentatives=1)
    session.executer_outil("search", requete="peu importe")
    session.etat.incrementer_tentative()  # simule le budget épuisé après cette tentative
    assert not session.etat.peut_reessayer

    etat = EtatGraphe(session=session, preuves_pertinentes=False, preuves_suffisantes=None)
    mise_a_jour = nodes.noeud_generer_reponse(etat)

    resultat = mise_a_jour["reponse"]
    assert resultat.contexte_suffisant is False
    assert resultat.sources == []
    assert "Recherche interrompue" in resultat.avertissements[0]
    assert "budget de tentatives épuisé" in resultat.avertissements[0]


def test_generer_reponse_refuse_sans_llm_si_pertinent_mais_insuffisant_et_budget_epuise(monkeypatch):
    """
    Autre branche du même correctif : le score de pertinence était bon, mais
    le niveau 2 a jugé les preuves insuffisantes, et le budget est épuisé.
    Même exigence : aucun appel au LLM de génération.
    """

    def _generation_interdite(**kwargs: Any):
        raise AssertionError("generer_depuis_recherche ne doit pas être appelé ici.")

    monkeypatch.setattr(nodes, "generer_depuis_recherche", _generation_interdite)

    session = _session_factice(fabriques=[_outil_search_evidence_forte], max_tentatives=1)
    session.executer_outil("search", requete="peu importe")
    session.etat.incrementer_tentative()
    assert not session.etat.peut_reessayer

    etat = EtatGraphe(
        session=session,
        preuves_pertinentes=True,
        preuves_suffisantes=False,
        raison_insuffisance="la valeur exacte n'apparaît pas dans les passages",
    )
    mise_a_jour = nodes.noeud_generer_reponse(etat)

    resultat = mise_a_jour["reponse"]
    assert resultat.contexte_suffisant is False
    assert "valeur exacte" in resultat.avertissements[0]


def test_generer_reponse_hors_sujet_refuse_proprement_sans_rapport(monkeypatch):
    """Comportement hors-sujet conservé : aucun rapport n'a jamais été construit."""

    def _generation_interdite(**kwargs: Any):
        raise AssertionError("generer_depuis_recherche ne doit pas être appelé ici.")

    monkeypatch.setattr(nodes, "generer_depuis_recherche", _generation_interdite)

    session = _session_factice(fabriques=[_outil_search_evidence_forte])
    etat = EtatGraphe(session=session, preuves_pertinentes=False, preuves_suffisantes=None)

    mise_a_jour = nodes.noeud_generer_reponse(etat)

    resultat = mise_a_jour["reponse"]
    assert resultat.contexte_suffisant is False
    assert "Corpus indisponible" in resultat.avertissements[0]


def test_generer_reponse_ne_journalise_pas_contexte_suffisant_comme_tel(monkeypatch):
    """
    `resultat.contexte_suffisant` ne mesure qu'une validité structurelle
    (voir src/rag/validation.py::valider_contexte) : la trace agent ne doit
    pas exposer ce champ sous un nom qui laisserait croire à un jugement de
    pertinence ou de suffisance, ce jugement étant déjà celui, distinct, de
    `evaluation_pertinence` / `evaluation_suffisance`.
    """
    monkeypatch.setattr(
        nodes, "generer_depuis_recherche", lambda **kw: _reponse_factice(kw["question"])
    )

    session = _session_factice(fabriques=[_outil_search_evidence_forte])
    session.executer_outil("search", requete="peu importe")
    etat = EtatGraphe(session=session, preuves_pertinentes=True, preuves_suffisantes=True)

    nodes.noeud_generer_reponse(etat)

    donnees = session.etat.trace[-1].donnees
    assert "contexte_suffisant" not in donnees
    assert donnees["rag_contexte_structurellement_valide"] is True
    assert donnees["genere_par_llm"] is True
