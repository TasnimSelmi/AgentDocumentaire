"""
Tests de la traduction YAML -> modèle Pydantic dynamique (`src/config.py`).

Couvre une régression d'ingestion : 19 des 21 échecs du dernier rapport
(`data/logs/rapport_ingestion.json`) venaient tous d'une date partielle
("2016-09-00", jour recopié tel quel depuis une citation qui ne précise que
le mois) rejetée strictement par Pydantic sur un champ optionnel, ce qui
faisait échouer l'ingestion du document entier plutôt que de simplement
laisser le champ vide.

Couvre aussi une seconde régression, même symptôme, cause différente :
sur le corpus CQuAE (contenu historique), le LLM renvoie parfois une année
nue en JSON (1993) au lieu d'une date ISO. Reçue comme int/float, Pydantic
l'interprète comme un timestamp et lève `date_from_datetime_inexact` — 11
documents sur 205 échouaient intégralement pour cette seule raison.
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


def test_date_annee_nue_entiere_sur_champ_optionnel_devient_none():
    """Une année nue en int (1993) ne doit pas faire échouer le document."""
    modele = _modele_date(obligatoire=False)

    instance = modele(date_document=1993)

    assert instance.date_document is None


def test_date_annee_nue_flottante_sur_champ_optionnel_devient_none():
    modele = _modele_date(obligatoire=False)

    instance = modele(date_document=1993.0)

    assert instance.date_document is None


def test_date_annee_nue_sur_champ_obligatoire_echoue_toujours():
    """Même règle que pour une date-chaîne invalide : un champ obligatoire échoue."""
    modele = _modele_date(obligatoire=True)

    with pytest.raises(ValidationError):
        modele(date_document=1993)


def test_date_booleenne_sur_champ_optionnel_echoue_toujours():
    """
    Un booléen n'est pas neutralisé comme une année nue : `isinstance(True, int)`
    vaut True en Python, mais un booléen n'est jamais une année valide — le
    laisser remonter tel quel à Pydantic (qui le rejette) est plus sûr qu'une
    neutralisation silencieuse qui masquerait un bug amont.
    """
    modele = _modele_date(obligatoire=False)

    with pytest.raises(ValidationError):
        modele(date_document=True)
