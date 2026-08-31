"""
`IngestionService` — orchestration mince source → pipeline gelé.

Le pipeline (`ingerer`) est remplacé par un faux : ces tests n'exécutent
jamais le vrai socle, donc aucun accès Ollama / Qdrant. Ils vérifient que la
façade se contente de matérialiser la source et de déléguer, sans rien
réimplémenter.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from src.rag.ingestion import RapportIngestion
from src.sources import ErreurSource, IngestionService, LocalDocumentSource


class _PipelineEspion:
    """Faux `ingerer` : mémorise ses arguments, renvoie un rapport marqueur."""

    def __init__(self) -> None:
        self.appels: list[dict] = []

    def __call__(self, **kwargs) -> RapportIngestion:
        self.appels.append(kwargs)
        return RapportIngestion(profil="espion")


class FakeDocumentSource:
    """Source minimale : matérialise un répertoire fixe déjà rempli."""

    def __init__(self, repertoire: Path) -> None:
        self._repertoire = repertoire
        self.entrees = 0
        self.sorties = 0

    @contextmanager
    def materialiser(self):
        self.entrees += 1
        try:
            yield self._repertoire
        finally:
            self.sorties += 1


def test_sync_delegue_au_pipeline_avec_le_repertoire_materialise(tmp_path):
    pipeline = _PipelineEspion()
    source = FakeDocumentSource(tmp_path)

    rapport = IngestionService(pipeline=pipeline).sync(source)

    assert rapport.profil == "espion"
    assert len(pipeline.appels) == 1
    assert pipeline.appels[0]["dossier"] == tmp_path
    assert source.entrees == 1 and source.sorties == 1


def test_sync_transmet_les_options_telles_quelles(tmp_path):
    pipeline = _PipelineEspion()

    IngestionService(pipeline=pipeline).sync(
        FakeDocumentSource(tmp_path),
        reinitialiser=True,
        limite=3,
        inferer=False,
        nom_profil="generic",
    )

    (appel,) = pipeline.appels
    assert appel == {
        "dossier": tmp_path,
        "reinitialiser": True,
        "limite": 3,
        "inferer": False,
        "nom_profil": "generic",
    }


def test_sync_valeurs_par_defaut_identiques_a_ingerer(tmp_path):
    pipeline = _PipelineEspion()

    IngestionService(pipeline=pipeline).sync(FakeDocumentSource(tmp_path))

    (appel,) = pipeline.appels
    assert appel == {
        "dossier": tmp_path,
        "reinitialiser": False,
        "limite": None,
        "inferer": True,
        "nom_profil": None,
    }


def test_local_source_equivaut_a_ingerer_sur_le_dossier(tmp_path):
    """Comportement local historique : sync(LocalDocumentSource(d)) == ingerer(dossier=d)."""
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    pipeline = _PipelineEspion()

    IngestionService(pipeline=pipeline).sync(LocalDocumentSource(tmp_path))

    (appel,) = pipeline.appels
    assert appel["dossier"] == tmp_path.resolve()
    assert appel["reinitialiser"] is False
    assert appel["inferer"] is True


def test_erreur_source_se_propage_sans_appeler_le_pipeline(tmp_path):
    pipeline = _PipelineEspion()
    source = LocalDocumentSource(tmp_path / "absent")

    with pytest.raises(ErreurSource):
        IngestionService(pipeline=pipeline).sync(source)

    assert pipeline.appels == []


def test_pipeline_par_defaut_est_ingerer():
    from src.rag.ingestion import ingerer

    assert IngestionService()._pipeline is ingerer
