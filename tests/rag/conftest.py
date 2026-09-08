"""
Fixtures partagées pour les tests multi-corpus de la couche `src/rag`.

Aucun Qdrant réel, aucun embedding réel : un faux client Qdrant en mémoire
(`FauxQdrantClient`) fait tourner le VRAI code de `src.rag.vectorstore`
(sélection de collection, construction de filtre, payload) contre un état
en mémoire organisé par collection — c'est précisément le point à prouver
pour l'isolation multi-corpus : que le bon nom de collection est utilisé
pour le bon corpus, pas de re-tester les garanties internes de Qdrant
lui-même (mature, déjà éprouvé).

`QDRANT_PATH` est redirigé vers un répertoire temporaire pour que les
registres de fichiers (`registry.json` / `registries/<corpus>.json`),
réellement écrits sur disque, n'interfèrent jamais avec un vrai corpus du
dépôt.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Any

import pytest

from src.config import ConfigCorpus, get_settings, recharger_config
from src.rag import retrieval, vectorstore


def _valeur_correspond(valeur: Any, match: Any) -> bool:
    if match is None:
        return True
    if hasattr(match, "value"):
        return valeur == match.value
    if hasattr(match, "any"):
        cible = valeur if isinstance(valeur, list) else [valeur]
        return any(v in match.any for v in cible)
    return True


def _point_correspond(payload: dict[str, Any], filtre: Any) -> bool:
    if filtre is None:
        return True
    for condition in getattr(filtre, "must", None) or []:
        cle = getattr(condition, "key", None)
        match = getattr(condition, "match", None)
        if cle is None:
            continue
        if not _valeur_correspond(payload.get(cle), match):
            return False
    return True


@dataclass
class _FausseCollection:
    points: dict[str, dict[str, Any]] = field(default_factory=dict)
    payload_schema: dict[str, Any] = field(default_factory=dict)


class FauxQdrantClient:
    """
    Double minimal du `QdrantClient` réel : un dict `{nom_collection:
    _FausseCollection}` en mémoire. Implémente exactement les méthodes que
    `src.rag.vectorstore` appelle. Aucune vraie recherche vectorielle : les
    scores sont fictifs, seul le filtrage par payload (utilisé par TOUTES
    les primitives de lecture documentaire testées ici) est réellement
    évalué.
    """

    def __init__(self) -> None:
        self._collections: dict[str, _FausseCollection] = {}

    # -- Collections ---------------------------------------------------

    def collection_exists(self, nom: str) -> bool:
        return nom in self._collections

    def create_collection(self, collection_name: str, **_kwargs: Any) -> None:
        self._collections[collection_name] = _FausseCollection()

    def delete_collection(self, nom: str) -> None:
        self._collections.pop(nom, None)

    def get_collection(self, nom: str) -> Any:
        col = self._collections[nom]
        return types.SimpleNamespace(
            payload_schema=dict(col.payload_schema),
            points_count=len(col.points),
            status="green",
        )

    def create_payload_index(
        self, *, collection_name: str, field_name: str, field_schema: Any, wait: bool = True
    ) -> None:
        self._collections[collection_name].payload_schema[field_name] = field_schema

    # -- Écriture --------------------------------------------------------

    def upsert(self, *, collection_name: str, points: list[Any], wait: bool = True) -> None:
        col = self._collections.setdefault(collection_name, _FausseCollection())
        for p in points:
            col.points[str(p.id)] = {"id": p.id, "payload": dict(p.payload)}

    def delete(self, *, collection_name: str, points_selector: Any, wait: bool = True) -> None:
        col = self._collections.get(collection_name)
        if col is None:
            return
        filtre = points_selector.filter
        a_retirer = [
            pid for pid, p in col.points.items() if _point_correspond(p["payload"], filtre)
        ]
        for pid in a_retirer:
            col.points.pop(pid, None)

    # -- Lecture -----------------------------------------------------------

    def scroll(
        self,
        *,
        collection_name: str,
        scroll_filter: Any = None,
        limit: int = 100,
        offset: Any = None,
        with_payload: Any = True,
        with_vectors: bool = False,
    ) -> tuple[list[Any], Any]:
        col = self._collections.get(collection_name)
        if col is None:
            return [], None
        items = [p for p in col.points.values() if _point_correspond(p["payload"], scroll_filter)]
        debut = int(offset) if offset else 0
        page = items[debut : debut + limit]
        suivant = debut + limit if debut + limit < len(items) else None
        resultats = [
            types.SimpleNamespace(id=p["id"], payload=dict(p["payload"]) if with_payload else None)
            for p in page
        ]
        return resultats, suivant

    def query_points(
        self,
        *,
        collection_name: str,
        query: Any = None,
        using: str | None = None,
        query_filter: Any = None,
        limit: int = 10,
        with_payload: bool = True,
        prefetch: Any = None,
    ) -> Any:
        col = self._collections.get(collection_name)
        # Recherche hybride (RRF) : le filtre voyage dans chaque `Prefetch`,
        # jamais en `query_filter` de premier niveau — voir
        # `vectorstore.rechercher`. Les deux prefetch (dense/sparse) portent
        # le même filtre dans ce projet ; le premier suffit.
        filtre_effectif = query_filter
        if filtre_effectif is None and prefetch:
            filtre_effectif = getattr(prefetch[0], "filter", None)
        items = (
            [p for p in col.points.values() if _point_correspond(p["payload"], filtre_effectif)]
            if col
            else []
        )
        # Score fictif : ordre d'insertion, décroissant — suffisant, aucun
        # test multi-corpus ne porte sur la qualité du classement sémantique.
        points = [
            types.SimpleNamespace(id=p["id"], score=1.0 - i * 0.001, payload=dict(p["payload"]))
            for i, p in enumerate(items[:limit])
        ]
        return types.SimpleNamespace(points=points)

    def count(self, *, collection_name: str, count_filter: Any = None, exact: bool = True) -> Any:
        col = self._collections.get(collection_name)
        items = (
            [p for p in col.points.values() if _point_correspond(p["payload"], count_filter)]
            if col
            else []
        )
        return types.SimpleNamespace(count=len(items))

    def close(self) -> None:
        pass


@pytest.fixture()
def faux_qdrant(monkeypatch, tmp_path):
    """
    Redirige `QDRANT_PATH` vers un répertoire temporaire (les registres de
    fichiers sont réellement écrits sur disque) et remplace le client Qdrant
    par `FauxQdrantClient` — le vrai code de `vectorstore.py`/`retrieval.py`
    s'exécute contre cet état en mémoire.

    `CORPORA_DIR` est redirigé de même (indépendant de `QDRANT_PATH`) : sans
    cela, un test d'upload/import-URL écrirait de vrais fichiers sous
    `data/corpora/` dans le dépôt (voir `Settings.corpora_dir`).
    """
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("CORPORA_DIR", str(tmp_path / "corpora"))
    recharger_config()
    get_settings().creer_dossiers()

    client = FauxQdrantClient()
    monkeypatch.setattr(vectorstore, "get_client", lambda: client)
    retrieval.reinitialiser_catalogue()

    yield client

    retrieval.reinitialiser_catalogue()
    recharger_config()


def cabler_registre_corpus(monkeypatch, entrees: dict[str, ConfigCorpus]) -> None:
    """Remplace le registre `config/corpus.yaml` par `entrees` pour la durée
    du test, sans jamais toucher au vrai fichier.

    Patche aussi `ecrire_registre_corpus` (absorbée en mémoire, jamais sur
    disque) : un appelant qui persiste un effet de bord — p. ex. la route
    `/ingestion` qui trace `derniere_ingestion` via `mettre_a_jour_corpus`
    après une synchronisation — ne doit JAMAIS pouvoir atteindre le vrai
    `config/corpus.yaml`, même quand le test n'attend explicitement aucune
    écriture. Pour un test qui veut inspecter cet état écrit, préférer
    `cabler_registre_corpus_inscriptible`."""
    cabler_registre_corpus_inscriptible(monkeypatch, entrees)


def cabler_registre_corpus_inscriptible(
    monkeypatch, entrees: dict[str, ConfigCorpus] | None = None
) -> dict[str, ConfigCorpus]:
    """Comme `cabler_registre_corpus`, mais `creer_corpus`/`mettre_a_jour_corpus`
    (donc `ecrire_registre_corpus`) restent utilisables : les deux opèrent sur
    le même dict en mémoire, jamais sur le vrai `config/corpus.yaml`. Renvoie
    ce dict, pour que le test inspecte l'état écrit."""
    import src.config as config_module

    registre: dict[str, ConfigCorpus] = dict(entrees or {"default": ConfigCorpus()})

    def _lire() -> dict[str, ConfigCorpus]:
        return dict(registre)

    _lire.cache_clear = lambda: None  # type: ignore[attr-defined]

    def _ecrire(nouveau: dict[str, ConfigCorpus]) -> None:
        registre.clear()
        registre.update(nouveau)

    monkeypatch.setattr(config_module, "get_registre_corpus", _lire)
    monkeypatch.setattr(config_module, "ecrire_registre_corpus", _ecrire)
    return registre
