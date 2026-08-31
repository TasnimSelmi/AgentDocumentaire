"""
Contrat générique de source documentaire — frontière entre l'origine des
documents et le pipeline d'ingestion gelé (`src/rag/ingestion.py`).

Idée directrice
---------------
Le pipeline sait déjà ingérer **un répertoire local** : `ingerer(dossier=…)`
accepte n'importe quel dossier, et toute la logique de déduplication, de
détection « inchangé » et de suppression y est portée par `RegistreFichiers`,
à partir du **contenu du répertoire**.

Une `DocumentSource` ne fournit donc pas des octets *au pipeline* : elle
**matérialise** ses documents dans un répertoire local, puis le pipeline gelé
prend le relais tel quel. Cette frontière ne demande aucune modification de
`src/rag/**`.

    origine (dossier local aujourd'hui ; API / GED / autre demain)
            │
            ▼
    DocumentSource.materialiser()  ──▶  répertoire = SNAPSHOT COMPLET  (Path)
            │
            ▼
    IngestionService.sync(source)  ──▶  ingerer(dossier=…)   [socle gelé, inchangé]
            │
            ▼
    découverte → hash → doc_id → parsing → chunks → embeddings → Qdrant

Invariant de sûreté — `materialiser()`
--------------------------------------
Le socle interprète *« présent au registre, absent du répertoire »* comme une
**suppression documentaire** (`RegistreFichiers.entrees_absentes` →
`supprimer_document`). Un répertoire incomplet exposé au pipeline
effacerait donc de l'index des documents qui n'ont pas disparu — ils ont
seulement échoué à être récupérés.

D'où l'invariant, **contraignant pour toute implémentation** :

    Le Path exposé par `materialiser()` représente TOUJOURS un snapshot
    complet et cohérent de la source à un instant donné — jamais un état
    partiel, jamais un état en cours de construction.

Corollaires :

  - une récupération incomplète (réseau, API, GED indisponible…) doit lever
    `ErreurSource` **sans rien exposer** : `materialiser()` ne `yield` pas,
    donc `IngestionService.sync` n'appelle jamais le pipeline, et l'index
    reste tel quel ;
  - une nouvelle matérialisation se construit **à l'écart** ; elle ne
    remplace le snapshot précédent qu'**après succès complet**, de façon
    atomique du point de vue d'un lecteur ;
  - une absence de document dans le snapshot ne peut être lue comme une vraie
    suppression **que parce que** le snapshot est garanti complet.

`LocalDocumentSource` satisfait l'invariant trivialement : le répertoire de
l'utilisateur *est* le snapshot, toujours cohérent, sans étape de
récupération faillible. Pour une source distante, hériter de
`SnapshotDocumentSource` (`src/sources/snapshot.py`), qui implémente le
schéma staging → publication atomique et rend l'invariant difficile à violer.

Ce que le contrat **n'inclut pas**, volontairement :
    - aucune découverte : elle reste dans le socle (`decouvrir_fichiers`) ;
    - aucune métadonnée métier : l'inférence LLM du socle en reste la seule
      source ;
    - aucune détection de changement / suppression propre : `RegistreFichiers`
      du socle s'en charge, à partir du snapshot ;
    - aucune hypothèse sur une API entreprise : pas d'URL, pas d'auth, pas de
      dépendance SharePoint / S3 / etc.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol, runtime_checkable


class ErreurSource(Exception):
    """
    Erreur d'accès à une source documentaire, levée **avant** toute écriture
    dans Qdrant : répertoire introuvable, récupération distante incomplète,
    snapshot invalide…

    Tant que cette exception peut être levée, aucun snapshot partiel n'est
    exposé au pipeline (voir l'invariant de `materialiser()`).

    Distincte de `src.rag.loaders.ErreurChargement`, qui porte sur l'échec
    d'un document individuel *pendant* l'ingestion, pas sur la source entière.
    """


@runtime_checkable
class DocumentSource(Protocol):
    """
    Contrat minimal d'une origine de documents.

    Une seule opération : `materialiser()` renvoie un gestionnaire de
    contexte qui expose un répertoire local — un **snapshot complet et
    cohérent** de la source — que le pipeline gelé sait ingérer.

        with source.materialiser() as repertoire:
            ingerer(dossier=repertoire, ...)

    Garanties exigées de toute implémentation :

      - le répertoire exposé n'est jamais partiel ni en cours de
        construction (voir l'invariant détaillé dans le module) ;
      - en cas d'impossibilité de produire un snapshot complet, l'appel lève
        `ErreurSource` **sans** entrer dans le bloc `with` (rien n'est
        `yield`) — le pipeline n'est donc pas appelé et l'index reste
        inchangé ;
      - les documents portent des noms **stables** d'un run à l'autre : le
        socle compare les chemins pour reconnaître un document déjà indexé.

    Pour une source locale, le répertoire *est* le dossier de l'utilisateur
    (aucune copie). Pour une source distante, c'est un miroir local persistant
    remplacé atomiquement après récupération complète.
    """

    def materialiser(self) -> AbstractContextManager[Path]:
        ...


__all__ = [
    "DocumentSource",
    "ErreurSource",
]
