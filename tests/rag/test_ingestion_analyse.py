"""
Tests du modèle Pydantic assemblé pour l'analyse LLM d'un document
(`src.rag.ingestion._modele_analyse`).

Régression : 6 des 21 échecs d'ingestion observés venaient d'un appel LLM qui
omettait le champ `confiance`, obligatoire dans le schéma Pydantic mais déjà
lu de façon défensive par `analyser_document` (`donnees.get("confiance",
1.0)`). Le document entier — catégorie et métadonnées comprises — était
rejeté pour l'absence d'un seul champ que le code appelant savait déjà
gérer comme optionnel.
"""

from __future__ import annotations

from src.config import Categorie, ConfigClassification, ConfigSchemaExtraction, Profil
from src.rag.ingestion import _modele_analyse


def _profil_minimal() -> Profil:
    return Profil(
        profile_name="test",
        classification=ConfigClassification(
            categories=[Categorie(nom="autre", description="par défaut")],
            categorie_defaut="autre",
        ),
        schema_extraction=ConfigSchemaExtraction(champs=[]),
    )


def test_confiance_absente_ne_fait_pas_echouer_le_document():
    Modele = _modele_analyse(_profil_minimal())

    instance = Modele(categorie="autre")

    assert instance.confiance == 1.0


def test_confiance_fournie_reste_prise_en_compte():
    Modele = _modele_analyse(_profil_minimal())

    instance = Modele(categorie="autre", confiance=0.4)

    assert instance.confiance == 0.4
