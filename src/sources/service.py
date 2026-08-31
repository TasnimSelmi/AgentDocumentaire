"""
`IngestionService` — orchestration mince entre une `DocumentSource` et le
pipeline d'ingestion gelé.

Rôle, et rien de plus :
    1. demander à la source son répertoire matérialisé ;
    2. faire **un seul** appel au pipeline : `ingerer(dossier=…)` ;
    3. renvoyer le `RapportIngestion` produit, intact.

La façade ne reproduit **aucune** logique du socle : ni découverte, ni hash,
ni détection « inchangé » / suppression, ni parsing, ni chunking, ni
embeddings, ni écriture Qdrant. Tout cela reste dans `src/rag/ingestion.py`,
inchangé.

Aucune dépendance FastAPI / UI / scheduler / auth / connecteur entreprise.
Le pipeline est injectable (`IngestionService(pipeline=…)`) pour des tests
hors ligne, sans Ollama ni Qdrant.
"""

from __future__ import annotations

from typing import Any, Callable

from src.rag.ingestion import RapportIngestion, ingerer
from src.sources.base import DocumentSource

#: Signature du pipeline délégué. `ingerer` par défaut ; un fake suffit en test.
Pipeline = Callable[..., RapportIngestion]


class IngestionService:
    """
    Point d'entrée applicatif de l'ingestion multi-source.

    Usage :

        service = IngestionService()
        rapport = service.sync(LocalDocumentSource("data/documents"))

    et, demain, sans rien changer ici :

        rapport = service.sync(EnterpriseDocumentSource(...))
    """

    def __init__(self, *, pipeline: Pipeline = ingerer) -> None:
        self._pipeline = pipeline

    def sync(
        self,
        source: DocumentSource,
        *,
        reinitialiser: bool = False,
        limite: int | None = None,
        inferer: bool = True,
        nom_profil: str | None = None,
    ) -> RapportIngestion:
        """
        Synchronise le contenu de `source` vers l'index.

        La source matérialise ses documents dans un répertoire, puis le
        pipeline gelé est exécuté sur ce répertoire. Les options
        (`reinitialiser`, `limite`, `inferer`, `nom_profil`) sont transmises
        telles quelles à `ingerer` — mêmes valeurs par défaut, même sémantique.

        Sûreté : le pipeline n'est appelé qu'à l'**intérieur** du bloc
        `with source.materialiser()`. Si la source ne peut pas produire un
        snapshot complet, `materialiser()` lève `ErreurSource` **avant** le
        `yield` ; l'exception se propage ici sans que `_pipeline` ait été
        appelé, donc sans aucune écriture Qdrant et sans qu'une récupération
        ratée puisse être lue comme une suppression documentaire.
        """
        options: dict[str, Any] = {
            "reinitialiser": reinitialiser,
            "limite": limite,
            "inferer": inferer,
            "nom_profil": nom_profil,
        }
        with source.materialiser() as repertoire:
            return self._pipeline(dossier=repertoire, **options)


__all__ = ["IngestionService", "Pipeline"]
