"""
Tests du détecteur multi-document (`src/agent/multidoc.py`) — étape P1.4.

Fonction pure et déterministe : aucun appel LLM / retrieval / génération.
Le détecteur PRODUIT UN SIGNAL, il ne route pas et n'exécute aucune
comparaison.
"""

from __future__ import annotations

import pytest

from src.agent.multidoc import (
    HINT_AUCUN,
    HINT_COMPARE,
    HINT_SYNTHESIZE,
    SignalMultiDoc,
    detecter_multidoc,
)


# --------------------------------------------------------------------------
# Multi-document — références explicites
# --------------------------------------------------------------------------


def test_deux_docs_explicites_plus_compare() -> None:
    s = detecter_multidoc("Compare rapport_alpha.pdf et rapport_beta.pdf.")
    assert s.is_multidoc is True
    assert s.operation_hint == HINT_COMPARE
    assert s.nombre_documents == 2
    assert s.references_detectees == ("rapport_alpha.pdf", "rapport_beta.pdf")
    assert s.confiance == "haute"


def test_deux_docs_explicites_plus_synthese() -> None:
    s = detecter_multidoc("Fais une synthèse de rapport_alpha.pdf et rapport_beta.pdf.")
    assert s.is_multidoc is True
    assert s.operation_hint == HINT_SYNTHESIZE
    assert s.nombre_documents == 2


def test_trois_docs_plus_consolidation() -> None:
    s = detecter_multidoc(
        "Consolide les recommandations de rapport_alpha.pdf, rapport_beta.pdf "
        "et rapport_gamma.pdf."
    )
    assert s.is_multidoc is True
    assert s.operation_hint == HINT_SYNTHESIZE
    assert s.nombre_documents == 3
    assert len(s.references_detectees) == 3


def test_docs_separes_par_slash() -> None:
    s = detecter_multidoc("En quoi contrat_2025.pdf diffère-t-il de contrat_2024.pdf ?")
    assert s.is_multidoc is True
    assert s.operation_hint == HINT_COMPARE


def test_reference_de_fichier_dupliquee_compte_une_fois() -> None:
    s = detecter_multidoc("Que dit rapport_alpha.pdf ? Et encore, rapport_alpha.pdf ?")
    assert s.is_multidoc is False
    assert s.references_detectees == ("rapport_alpha.pdf",)
    assert s.nombre_documents == 1


# --------------------------------------------------------------------------
# Multi-document — marqueurs pluriels / pronominaux
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "Quelles différences entre ces deux documents ?",
        "Compare these two reports.",
        "What do both documents have in common?",
        "Résume ces rapports.",
        "Combine the key findings from these two reports.",
    ],
)
def test_marqueurs_pluriels_declenchent_multidoc(query: str) -> None:
    s = detecter_multidoc(query)
    assert s.is_multidoc is True
    assert s.marqueur_pluriel is not None


def test_bilingue_fr_en() -> None:
    fr = detecter_multidoc("Compare rapport_alpha.pdf et rapport_beta.pdf.")
    en = detecter_multidoc("Compare report_a.pdf and report_b.pdf.")
    assert fr.is_multidoc and en.is_multidoc
    assert fr.operation_hint == en.operation_hint == HINT_COMPARE


# --------------------------------------------------------------------------
# MONO-document — pièges à éviter (le point délicat)
# --------------------------------------------------------------------------


def test_un_doc_plus_compare_les_deux_methodes_reste_mono() -> None:
    s = detecter_multidoc("Compare les deux méthodes décrites dans rapport_alpha.pdf.")
    assert s.is_multidoc is False
    assert s.operation_hint == HINT_AUCUN
    assert s.nombre_documents == 1


def test_points_communs_entre_deux_approches_dans_ce_document_reste_mono() -> None:
    s = detecter_multidoc(
        "Quels sont les points communs entre les deux approches présentées "
        "dans ce document ?"
    )
    assert s.is_multidoc is False
    assert s.operation_hint == HINT_AUCUN


def test_synthetiser_un_seul_doc_nest_pas_synthesize() -> None:
    s = detecter_multidoc("Peux-tu synthétiser rapport_alpha.pdf ?")
    assert s.is_multidoc is False
    assert s.operation_hint == HINT_AUCUN


def test_deixis_singuliere_neutralise_un_marqueur_pluriel_faible() -> None:
    # « les deux sections » n'est pas un nom de document -> aucun marqueur ;
    # « ce document » impose le mono.
    s = detecter_multidoc("Compare les deux sections de ce document.")
    assert s.is_multidoc is False


# --------------------------------------------------------------------------
# Multi-document SANS opération (lecture factuelle)
# --------------------------------------------------------------------------


def test_query_factuelle_sur_plusieurs_docs_est_multidoc_sans_operation() -> None:
    s = detecter_multidoc("Dans ces documents, quelle est la date limite de dépôt ?")
    assert s.is_multidoc is True
    assert s.operation_hint == HINT_AUCUN
    assert s.confiance == "moyenne"


# --------------------------------------------------------------------------
# Requête vague — ne rien inventer
# --------------------------------------------------------------------------


def test_requete_vague_compare_ninvente_pas_deux_documents() -> None:
    s = detecter_multidoc("Compare.")
    assert s.is_multidoc is False
    assert s.operation_hint == HINT_AUCUN
    assert s.references_detectees == ()
    assert s.nombre_documents == 0


def test_requete_vide() -> None:
    s = detecter_multidoc("")
    assert s.is_multidoc is False
    assert s.operation_hint == HINT_AUCUN


# --------------------------------------------------------------------------
# Anti-faux-positifs SEARCH du banc P1.1
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        # RT-017
        "Compare les deux méthodes décrites dans rapport_alpha.pdf.",
        # RT-018
        "Quels sont les points communs entre les deux approches présentées "
        "dans ce document ?",
        # variantes proches
        "Résume le document rapport_alpha.pdf.",
        "Quel est le montant total indiqué dans facture_2025.pdf ?",
    ],
)
def test_pas_de_faux_positif_multidoc(query: str) -> None:
    assert detecter_multidoc(query).is_multidoc is False


def test_rt023_est_le_seul_multidoc_search_attendu() -> None:
    # RT-023 : signalé multi-doc, mais operation_hint doit rester none.
    s = detecter_multidoc("Dans ces documents, quelle est la date limite de dépôt des dossiers ?")
    assert s.is_multidoc is True
    assert s.operation_hint == HINT_AUCUN


# --------------------------------------------------------------------------
# Déterminisme & résolveur optionnel
# --------------------------------------------------------------------------


def test_deterministe() -> None:
    q = "Compare rapport_alpha.pdf et rapport_beta.pdf."
    sorties = {detecter_multidoc(q).vers_dict()["raison"] for _ in range(5)}
    assert len(sorties) == 1


def test_resolveur_optionnel_jamais_appele_par_defaut() -> None:
    appels: list[str] = []

    def _resolveur(_q: str):  # pragma: no cover - ne doit pas être atteint
        appels.append(_q)
        return ["a.pdf", "b.pdf"]

    # Références explicites présentes -> le résolveur n'est pas sollicité.
    detecter_multidoc("Compare rapport_alpha.pdf et rapport_beta.pdf.", resolveur=_resolveur)
    assert appels == []


def test_resolveur_renforce_le_signal_quand_aucune_reference_explicite() -> None:
    def _resolveur(_q: str):
        return ["doc-uuid-1", "doc-uuid-2", "doc-uuid-2"]  # 2 distincts

    s = detecter_multidoc("Compare les rapports trimestriels.", resolveur=_resolveur)
    assert s.is_multidoc is True
    assert s.nombre_documents == 2
    assert s.confiance == "moyenne"


def test_resolveur_defaillant_ne_casse_pas_la_detection() -> None:
    def _resolveur(_q: str):
        raise RuntimeError("catalogue indisponible")

    s = detecter_multidoc("Compare rapport_alpha.pdf et rapport_beta.pdf.", resolveur=_resolveur)
    assert s.is_multidoc is True


# --------------------------------------------------------------------------
# Contrat de type
# --------------------------------------------------------------------------


def test_signal_est_immuable() -> None:
    s = detecter_multidoc("Compare rapport_alpha.pdf et rapport_beta.pdf.")
    assert isinstance(s, SignalMultiDoc)
    with pytest.raises((AttributeError, TypeError)):
        s.is_multidoc = False  # type: ignore[misc]
