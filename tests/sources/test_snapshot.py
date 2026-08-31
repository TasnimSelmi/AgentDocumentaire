"""
`SnapshotDocumentSource` — invariant de sûreté du contrat `DocumentSource`.

Une erreur de récupération ne doit JAMAIS pouvoir être exposée au pipeline
comme un snapshot, donc jamais être interprétée comme une suppression
documentaire. Ces tests exercent le schéma staging → publication atomique
avec un `_recuperer` scripté (succès, panne, snapshot partiel, snapshot
vide). Aucun accès Ollama / Qdrant : le pipeline est un espion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.rag.ingestion import RapportIngestion
from src.sources import ErreurSource, IngestionService, SnapshotDocumentSource
from src.sources.base import DocumentSource


class _PipelineEspion:
    def __init__(self) -> None:
        self.appels: list[dict] = []

    def __call__(self, **kwargs) -> RapportIngestion:
        self.appels.append(kwargs)
        return RapportIngestion(profil="espion")


class _FakeSnapshot(SnapshotDocumentSource):
    """Source snapshot dont la récupération est fournie par un callable."""

    def __init__(self, *, racine_miroir, remplir, autoriser_snapshot_vide=False):
        super().__init__(
            racine_miroir=racine_miroir,
            autoriser_snapshot_vide=autoriser_snapshot_vide,
        )
        self._remplir = remplir
        self.appels = 0

    def _recuperer(self, staging: Path) -> None:
        self.appels += 1
        self._remplir(staging)


def _ecrire_docs(staging: Path, noms) -> None:
    for nom in noms:
        (staging / nom).write_text(f"contenu {nom}", encoding="utf-8")


def _fichiers(repertoire: Path) -> list[str]:
    return sorted(p.name for p in repertoire.iterdir() if p.is_file())


def _residus(miroir: Path) -> list[Path]:
    """Répertoires de staging ou `.precedent` laissés autour du miroir."""
    motif_staging = list(miroir.parent.glob(f".{miroir.name}.staging-*"))
    precedent = miroir.with_name(miroir.name + ".precedent")
    return motif_staging + ([precedent] if precedent.exists() else [])


# ---------------------------------------------------------------------------
# Snapshot réussi
# ---------------------------------------------------------------------------

def test_snapshot_reussi_appelle_le_pipeline_avec_un_miroir_complet(tmp_path):
    miroir = tmp_path / "mir"
    pipeline = _PipelineEspion()
    source = _FakeSnapshot(
        racine_miroir=miroir,
        remplir=lambda s: _ecrire_docs(s, ["a.txt", "b.txt"]),
    )

    rapport = IngestionService(pipeline=pipeline).sync(source)

    assert rapport.profil == "espion"
    (appel,) = pipeline.appels
    assert appel["dossier"] == miroir
    assert _fichiers(miroir) == ["a.txt", "b.txt"]
    assert _residus(miroir) == []


def test_premier_snapshot_sans_miroir_prealable(tmp_path):
    miroir = tmp_path / "pas_encore" / "mir"  # le parent n'existe pas
    source = _FakeSnapshot(
        racine_miroir=miroir,
        remplir=lambda s: _ecrire_docs(s, ["a.txt"]),
    )

    with source.materialiser() as repertoire:
        assert repertoire == miroir
        assert _fichiers(repertoire) == ["a.txt"]


def test_yield_seulement_apres_recuperation_complete(tmp_path):
    ordre: list[str] = []

    def remplir(staging: Path) -> None:
        ordre.append("recuperer")
        _ecrire_docs(staging, ["a.txt"])

    source = _FakeSnapshot(racine_miroir=tmp_path / "mir", remplir=remplir)
    with source.materialiser():
        ordre.append("yield")

    assert ordre == ["recuperer", "yield"]


def test_conforme_au_protocole(tmp_path):
    source = _FakeSnapshot(
        racine_miroir=tmp_path / "mir",
        remplir=lambda s: _ecrire_docs(s, ["a.txt"]),
    )
    assert isinstance(source, DocumentSource)


# ---------------------------------------------------------------------------
# Récupération en échec → aucun snapshot exposé
# ---------------------------------------------------------------------------

def test_recuperation_echoue_pipeline_jamais_appele(tmp_path):
    pipeline = _PipelineEspion()

    def boom(staging: Path) -> None:
        _ecrire_docs(staging, ["a.txt"])          # écriture partielle...
        raise RuntimeError("API GED indisponible")  # ...puis panne

    source = _FakeSnapshot(racine_miroir=tmp_path / "mir", remplir=boom)

    with pytest.raises(ErreurSource):
        IngestionService(pipeline=pipeline).sync(source)

    assert pipeline.appels == []


def test_snapshot_partiel_ne_remplace_pas_le_miroir_precedent(tmp_path):
    miroir = tmp_path / "mir"

    with _FakeSnapshot(
        racine_miroir=miroir,
        remplir=lambda s: _ecrire_docs(s, ["a.txt", "b.txt"]),
    ).materialiser() as repertoire:
        assert _fichiers(repertoire) == ["a.txt", "b.txt"]

    def partiel(staging: Path) -> None:
        _ecrire_docs(staging, ["a.txt"])
        raise ConnectionError("coupure réseau à mi-parcours")

    with pytest.raises(ErreurSource):
        with _FakeSnapshot(racine_miroir=miroir, remplir=partiel).materialiser():
            pass

    # miroir v1 strictement intact, aucun résidu de staging / .precedent
    assert _fichiers(miroir) == ["a.txt", "b.txt"]
    assert (miroir / "b.txt").read_text(encoding="utf-8") == "contenu b.txt"
    assert _residus(miroir) == []


def test_suppression_reelle_visible_seulement_apres_snapshot_complet(tmp_path):
    miroir = tmp_path / "mir"

    # v1 : a, b, c
    with _FakeSnapshot(
        racine_miroir=miroir,
        remplir=lambda s: _ecrire_docs(s, ["a.txt", "b.txt", "c.txt"]),
    ).materialiser():
        pass

    # Une récupération ratée ne fait PAS disparaître b et c.
    def ratee(staging: Path) -> None:
        _ecrire_docs(staging, ["a.txt"])
        raise TimeoutError("GED muette")

    with pytest.raises(ErreurSource):
        with _FakeSnapshot(racine_miroir=miroir, remplir=ratee).materialiser():
            pass
    assert _fichiers(miroir) == ["a.txt", "b.txt", "c.txt"]

    # v2 réussie : c a réellement disparu de la source → suppression légitime,
    # visible uniquement parce que le snapshot est garanti complet.
    with _FakeSnapshot(
        racine_miroir=miroir,
        remplir=lambda s: _ecrire_docs(s, ["a.txt", "b.txt"]),
    ).materialiser() as repertoire:
        assert _fichiers(repertoire) == ["a.txt", "b.txt"]


# ---------------------------------------------------------------------------
# Snapshot vide : suspect par défaut, publiable si explicitement autorisé
# ---------------------------------------------------------------------------

def test_snapshot_vide_traite_comme_echec_par_defaut(tmp_path):
    miroir = tmp_path / "mir"
    with _FakeSnapshot(
        racine_miroir=miroir,
        remplir=lambda s: _ecrire_docs(s, ["a.txt"]),
    ).materialiser():
        pass

    vide = _FakeSnapshot(racine_miroir=miroir, remplir=lambda s: None)
    with pytest.raises(ErreurSource):
        with vide.materialiser():
            pass

    assert _fichiers(miroir) == ["a.txt"]  # rien supprimé
    assert _residus(miroir) == []


def test_snapshot_vide_publie_si_explicitement_autorise(tmp_path):
    miroir = tmp_path / "mir"
    with _FakeSnapshot(
        racine_miroir=miroir,
        remplir=lambda s: _ecrire_docs(s, ["a.txt", "b.txt"]),
    ).materialiser():
        pass

    vide_ok = _FakeSnapshot(
        racine_miroir=miroir,
        remplir=lambda s: None,
        autoriser_snapshot_vide=True,
    )
    with vide_ok.materialiser() as repertoire:
        assert _fichiers(repertoire) == []   # suppression totale volontaire

    assert _fichiers(miroir) == []


# ---------------------------------------------------------------------------
# Reprise d'interruption
# ---------------------------------------------------------------------------

def test_precedent_orphelin_resorbe_au_prochain_publie(tmp_path):
    miroir = tmp_path / "mir"
    with _FakeSnapshot(
        racine_miroir=miroir,
        remplir=lambda s: _ecrire_docs(s, ["a.txt"]),
    ).materialiser():
        pass

    # Simule un processus tué au milieu de la publication : un `.precedent` traîne.
    orphelin = miroir.with_name(miroir.name + ".precedent")
    orphelin.mkdir()
    (orphelin / "vieux.txt").write_text("x", encoding="utf-8")

    with _FakeSnapshot(
        racine_miroir=miroir,
        remplir=lambda s: _ecrire_docs(s, ["b.txt"]),
    ).materialiser() as repertoire:
        assert _fichiers(repertoire) == ["b.txt"]

    assert not orphelin.exists()
    assert _fichiers(miroir) == ["b.txt"]
