"""
Construction **paresseuse** des collaborateurs par défaut de l'API.

Rien n'est instancié à l'import : `create_app()` appelle ces fabriques, et le
registre de sources ne matérialise `LocalDocumentSource` qu'au moment d'un
`POST /ingestion`. Aucun accès Ollama / Qdrant / système de fichiers ici.

Registre de sources — contrainte de sécurité P2.3
------------------------------------------------
Le client ne fournit jamais de chemin. Il désigne au plus une source par un
**nom logique** (`"local"`, …) résolu contre ce registre `nom -> fabrique`.
Ajouter une source (connecteur entreprise implémentant `DocumentSource` via
`SnapshotDocumentSource`) = ajouter une entrée ici, jamais un paramètre HTTP.
"""

from __future__ import annotations

from typing import Callable

from src.agent.service import AgentService
from src.config import get_settings
from src.rag.corpus import dossier_managed_pour_corpus
from src.sources import DocumentSource, IngestionService, LocalDocumentSource

#: `nom logique -> fabrique de DocumentSource`, appelée à chaque ingestion
#: avec le `corpus_id` cible — une fabrique qui n'en a pas besoin (`"local"`)
#: l'ignore simplement. Permet à une source de dériver un emplacement propre
#: à CE corpus (`"managed"`) sans jamais accepter de chemin du client.
FabriqueSource = Callable[[str], DocumentSource]
RegistreSources = dict[str, FabriqueSource]


def registre_sources_par_defaut() -> RegistreSources:
    """
    Sources logiques réellement enregistrées côté backend.

    - `"local"` : dossier documentaire configuré globalement
      (`Settings.documents_dir`), historique, partagé — comportement inchangé.
    - `"managed"` : stockage propre à CE corpus (`corpus_id`), alimenté par
      upload de dossier ou import URL (`POST /corpora/{corpus_id}/upload`,
      `POST /corpora/{corpus_id}/import-url`) — jamais partagé entre corpus.

    `get_settings()` est relu à chaque appel de fabrique — pas de chemin figé
    à la création de l'app.
    """
    return {
        "local": lambda corpus_id: LocalDocumentSource(get_settings().documents_dir),
        "managed": lambda corpus_id: LocalDocumentSource(dossier_managed_pour_corpus(corpus_id)),
    }


def agent_service_par_defaut() -> AgentService:
    """Façade P1 avec ses réglages par défaut (`point_entree=executer_agent`)."""
    return AgentService()


def ingestion_service_par_defaut() -> IngestionService:
    """Façade d'ingestion P2 avec le pipeline gelé (`pipeline=ingerer`)."""
    return IngestionService()


__all__ = [
    "FabriqueSource",
    "RegistreSources",
    "registre_sources_par_defaut",
    "agent_service_par_defaut",
    "ingestion_service_par_defaut",
]
