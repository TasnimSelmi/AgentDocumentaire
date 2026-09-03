"""
`InstrumentedIngestionService` — délégation unique, rapport intact,
started/completed, partiel, `ErreurSource`, exception inattendue, compteurs
exacts, aucun chemin local exposé.
"""

from __future__ import annotations

import json

import pytest

from src.observability.events import (
    INGESTION_COMPLETED,
    INGESTION_FAILED,
    INGESTION_STARTED,
)
from src.observability.instrumentation import InstrumentedIngestionService
from src.observability.redaction import evenement_vers_log
from src.sources.base import ErreurSource
from tests.observability.conftest import (
    EspionIngestionInner,
    FauxRapportIngestion,
    SinkQuiExplose,
)


def test_inner_appele_une_fois_rapport_intact(sink, source_factice):
    rapport = FauxRapportIngestion(fichiers_trouves=3, fichiers_traites=3, chunks_indexes=9)
    inner = EspionIngestionInner(rapport)
    service = InstrumentedIngestionService(inner, sink)

    resultat = service.sync(source_factice, reinitialiser=True, limite=5)

    assert len(inner.appels) == 1
    assert inner.appels[0] == {
        "source": source_factice,
        "reinitialiser": True,
        "limite": 5,
        "inferer": True,
        "nom_profil": None,
    }
    assert resultat is rapport


def test_started_puis_completed_succes(sink, source_factice):
    rapport = FauxRapportIngestion(
        profil="generic",
        fichiers_trouves=10,
        fichiers_ignores_inchanges=2,
        fichiers_traites=8,
        fichiers_vides=1,
        fichiers_ocr=3,
        fichiers_supprimes=1,
        chunks_indexes=120,
    )
    InstrumentedIngestionService(EspionIngestionInner(rapport), sink).sync(source_factice)

    assert sink.noms() == [INGESTION_STARTED, INGESTION_COMPLETED]
    completed = sink.unique(INGESTION_COMPLETED)
    assert completed.outcome == "success"
    a = completed.attributes
    assert (a.files_found, a.files_processed, a.files_skipped) == (10, 8, 2)
    assert (a.files_empty, a.files_ocr, a.files_deleted) == (1, 3, 1)
    assert a.chunks_indexed == 120
    assert a.profile == "generic"


def test_rapport_avec_echecs_est_partial(sink, source_factice):
    rapport = FauxRapportIngestion(
        fichiers_trouves=10, fichiers_traites=7, fichiers_en_echec=3, chunks_indexes=40
    )
    InstrumentedIngestionService(EspionIngestionInner(rapport), sink).sync(source_factice)
    completed = sink.unique(INGESTION_COMPLETED)
    assert completed.outcome == "partial"
    assert completed.attributes.files_failed == 3


def test_erreur_source_emet_failed_source_unavailable_puis_releve(sink, source_factice):
    inner = EspionIngestionInner(exception=ErreurSource("montage //nas/share indispo"))
    service = InstrumentedIngestionService(inner, sink)

    with pytest.raises(ErreurSource):
        service.sync(source_factice)

    failed = sink.unique(INGESTION_FAILED)
    assert failed.outcome == "error"
    assert failed.attributes.error_category == "source_unavailable"
    assert failed.attributes.error_code == "ErreurSource"


def test_exception_inattendue_emet_failed_unexpected_puis_releve(sink, source_factice):
    inner = EspionIngestionInner(exception=RuntimeError("qdrant timeout 10.0.0.5"))
    service = InstrumentedIngestionService(inner, sink)

    with pytest.raises(RuntimeError):
        service.sync(source_factice)

    failed = sink.unique(INGESTION_FAILED)
    assert failed.attributes.error_category == "unexpected"
    assert failed.attributes.error_code == "RuntimeError"
    assert failed.attributes.error_stack is not None


def test_aucun_chemin_local_dans_la_trace(sink, source_factice):
    """La source porte un chemin sensible (`/home/secret/...`). Il ne doit
    apparaître ni dans l'événement, ni dans sa sérialisation."""
    inner = EspionIngestionInner(
        exception=RuntimeError(f"échec sur {source_factice.racine}/doc.pdf")
    )
    with pytest.raises(RuntimeError):
        InstrumentedIngestionService(inner, sink).sync(source_factice)

    failed = sink.unique(INGESTION_FAILED)
    assert failed.attributes.source == "SourceLocaleFactice"  # nom de type, pas un chemin
    charge = json.dumps(evenement_vers_log(failed))
    assert "/home/secret" not in charge


def test_sink_defaillant_n_impacte_pas_l_ingestion(source_factice):
    rapport = FauxRapportIngestion(chunks_indexes=5)
    inner = EspionIngestionInner(rapport)
    service = InstrumentedIngestionService(inner, SinkQuiExplose())
    assert service.sync(source_factice) is rapport


def test_emit_start_desactive(sink, source_factice):
    InstrumentedIngestionService(
        EspionIngestionInner(FauxRapportIngestion()), sink, emit_start=False
    ).sync(source_factice)
    assert sink.noms() == [INGESTION_COMPLETED]


def test_inner_property_expose_le_service_enveloppe(sink):
    inner = EspionIngestionInner(FauxRapportIngestion())
    assert InstrumentedIngestionService(inner, sink).inner is inner
