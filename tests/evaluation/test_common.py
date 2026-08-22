"""
Tests du socle partagé du harnais d'évaluation.

Couvre deux régressions constatées sur le run d'ablation C0_reference :
une contention transitoire du verrou Qdrant local comptée comme 15 erreurs
de retrieval, et un import cassé (`evaluation._runner_config`) qui faisait
échouer silencieusement les configurations d'ablation nécessitant une
surcharge de configuration (C5, C6).
"""

from __future__ import annotations

import runpy

import pytest

import evaluation.common as common


def test_attendre_client_qdrant_reessaie_puis_reussit(monkeypatch):
    """Un verrou transitoire ne doit pas être remonté après quelques essais."""
    appels = {"n": 0}

    def get_client_factice():
        appels["n"] += 1
        if appels["n"] < 3:
            raise RuntimeError(
                "Storage folder ... is already accessed by another instance "
                "of Qdrant client."
            )
        return object()

    import src.rag.vectorstore as vectorstore

    monkeypatch.setattr(vectorstore, "get_client", get_client_factice)
    monkeypatch.setattr(common.time, "sleep", lambda _: None)

    common.attendre_client_qdrant(tentatives=5, delai_secondes=0)

    assert appels["n"] == 3


def test_attendre_client_qdrant_leve_apres_epuisement(monkeypatch):
    """Une contention qui ne se résorbe jamais doit rester une vraie erreur."""

    def get_client_toujours_verrouille():
        raise RuntimeError(
            "Storage folder ... is already accessed by another instance "
            "of Qdrant client."
        )

    import src.rag.vectorstore as vectorstore

    monkeypatch.setattr(vectorstore, "get_client", get_client_toujours_verrouille)
    monkeypatch.setattr(common.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="toujours occupé"):
        common.attendre_client_qdrant(tentatives=3, delai_secondes=0)


def test_attendre_client_qdrant_ne_masque_pas_une_autre_erreur(monkeypatch):
    """Une RuntimeError sans rapport avec le verrou ne doit jamais être avalée."""

    def get_client_en_panne():
        raise RuntimeError("collection introuvable")

    import src.rag.vectorstore as vectorstore

    monkeypatch.setattr(vectorstore, "get_client", get_client_en_panne)

    with pytest.raises(RuntimeError, match="collection introuvable"):
        common.attendre_client_qdrant(tentatives=5, delai_secondes=0)


def test_module_runner_config_importable_sous_le_nom_attendu():
    """
    Régression : le fichier s'appelait `runner_config.py` (sans tiret bas)
    alors que `run_ablation.py` invoque `python -m evaluation._runner_config`
    pour les configurations qui surchargent `default.yaml` (C5, C6). Le nom
    de fichier doit correspondre exactement au module invoqué.
    """
    module = runpy.run_module("evaluation._runner_config", run_name="_test_import")
    assert "installer_surcharges" in module
    assert "appliquer_surcharge" in module
