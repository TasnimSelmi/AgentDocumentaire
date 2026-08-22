"""
Tests de la traduction YAML -> modèle Pydantic dynamique (`src/config.py`).

Couvre une régression d'ingestion : 19 des 21 échecs du dernier rapport
(`data/logs/rapport_ingestion.json`) venaient tous d'une date partielle
("2016-09-00", jour recopié tel quel depuis une citation qui ne précise que
le mois) rejetée strictement par Pydantic sur un champ optionnel, ce qui
faisait échouer l'ingestion du document entier plutôt que de simplement
laisser le champ vide.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import Champ, _modele_depuis_champs


def _modele_date(obligatoire: bool):
    champ = Champ(
        nom="date_document",
        type="date",
        description="date du document",
        obligatoire=obligatoire,
    )
    return _modele_depuis_champs("TestMetaDate", "test", [champ])


def test_date_invalide_sur_champ_optionnel_devient_none():
    """Un jour hors calendrier ('00') ne doit pas faire échouer le document."""
    modele = _modele_date(obligatoire=False)

    instance = modele(date_document="2016-09-00")

    assert instance.date_document is None


def test_date_valide_reste_inchangee():
    modele = _modele_date(obligatoire=False)

    instance = modele(date_document="2020-01-15")

    assert str(instance.date_document) == "2020-01-15"


def test_date_vide_reste_none():
    modele = _modele_date(obligatoire=False)

    instance = modele(date_document="")

    assert instance.date_document is None


def test_date_invalide_sur_champ_obligatoire_echoue_toujours():
    """Un champ de date obligatoire doit continuer à échouer si invalide."""
    modele = _modele_date(obligatoire=True)

    with pytest.raises(ValidationError):
        modele(date_document="2016-09-00")
