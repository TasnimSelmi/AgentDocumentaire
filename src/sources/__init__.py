"""
Abstraction générique des sources documentaires (P2.2).

Découple l'origine des documents du pipeline d'ingestion. Le pipeline
(`src/rag/ingestion.py`) reste **inchangé** : une source matérialise ses
documents dans un répertoire local — un **snapshot complet et cohérent** —
et `ingerer(dossier=…)` prend le relais.

    from src.sources import IngestionService, LocalDocumentSource

    service = IngestionService()
    rapport = service.sync(LocalDocumentSource("data/documents"))

Un connecteur entreprise futur héritera de `SnapshotDocumentSource` (même
contrat `DocumentSource`) sans rien changer au RAG ni à l'agent. Voir
`docs/P2.2_SOURCES.md`.
"""

from src.sources.base import DocumentSource, ErreurSource
from src.sources.local import LocalDocumentSource
from src.sources.service import IngestionService, Pipeline
from src.sources.snapshot import SnapshotDocumentSource

__all__ = [
    "DocumentSource",
    "ErreurSource",
    "LocalDocumentSource",
    "SnapshotDocumentSource",
    "IngestionService",
    "Pipeline",
]
