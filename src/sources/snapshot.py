"""
`SnapshotDocumentSource` — base pour toute source dont la matérialisation
implique une **récupération faillible** (réseau, API, GED, montage distant).

Elle implémente `materialiser()` une fois pour toutes, de façon à garantir
l'invariant de sûreté du contrat `DocumentSource` :

    Une erreur de récupération ne peut JAMAIS être exposée au pipeline comme
    un snapshot, donc ne peut jamais être interprétée comme une suppression
    documentaire.

Schéma :

    1. un répertoire de *staging* neuf est créé à côté du miroir publié ;
    2. la sous-classe le remplit via `_recuperer(staging)` — intégralement,
       ou en levant ;
    3a. `_recuperer` lève  → le staging est détruit, le miroir publié reste
        **strictement intact**, `ErreurSource` est relevée, `materialiser()`
        ne `yield` rien → `IngestionService.sync` n'appelle pas le pipeline ;
    3b. `_recuperer` réussit → le staging **remplace atomiquement** le miroir
        publié (du point de vue d'un lecteur, jamais d'état mixte) ;
    4. `materialiser()` `yield` le miroir publié — un snapshot complet.

Le miroir publié est **persistant** : on ne le supprime pas en sortie. C'est
nécessaire pour que la détection « inchangé » / « supprimé » du socle, qui
compare des chemins de fichiers d'un run à l'autre, reste stable.

`EnterpriseDocumentSource` n'est pas fourni : cette base et ses tests rendent
seulement l'invariant explicite et difficile à violer.
"""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from src.sources.base import ErreurSource

_SUFFIXE_STAGING = ".staging-"
_SUFFIXE_PRECEDENT = ".precedent"


def _publier_atomiquement(staging: Path, publie: Path) -> None:
    """
    Remplace `publie` par `staging`. Du point de vue d'un lecteur (le
    pipeline), `publie` ne contient jamais un mélange ancien / nouveau : il
    pointe soit sur l'intégralité de l'ancien snapshot, soit sur
    l'intégralité du nouveau.

    `publie` peut ne pas exister (premier snapshot) ou être un répertoire
    peuplé (snapshots suivants). `staging` et `publie` doivent être sur le
    même système de fichiers — c'est le cas par construction (`staging` est
    créé comme voisin de `publie`).

    Reprise : si le processus est tué entre les deux renommages, un
    répertoire `<publie>.precedent` subsiste ; l'appel suivant le résorbe.
    """
    publie.parent.mkdir(parents=True, exist_ok=True)
    precedent = publie.with_name(publie.name + _SUFFIXE_PRECEDENT)

    # Résorption d'une éventuelle interruption antérieure.
    if precedent.exists() and not publie.exists():
        os.replace(precedent, publie)
    if precedent.exists():
        shutil.rmtree(precedent)

    if publie.exists():
        os.replace(publie, precedent)   # renommage atomique du répertoire publié
    os.replace(staging, publie)         # le staging complet devient le miroir publié
    if precedent.exists():
        shutil.rmtree(precedent)


class SnapshotDocumentSource(ABC):
    """
    Base pour une source à récupération faillible.

    Une sous-classe implémente uniquement `_recuperer(staging)`. Elle **ne
    doit jamais** avaler une erreur de récupération : à la moindre récupération
    incomplète, elle lève (n'importe quelle exception convient — elle est
    requalifiée en `ErreurSource`).

    Paramètres
    ----------
    racine_miroir :
        Emplacement **persistant** du snapshot publié, sur un système de
        fichiers accessible en écriture. C'est ce chemin qui est passé à
        `ingerer(dossier=…)` ; il doit rester le même d'un run à l'autre.
    autoriser_snapshot_vide :
        Si `False` (défaut), un `_recuperer` qui laisse le staging **vide**
        est considéré comme suspect (récupération silencieusement ratée) et
        lève `ErreurSource` sans rien publier. Ne passer `True` que si
        « la source ne contient réellement aucun document » est un état
        légitime et distinct d'un échec — auquel cas publier un snapshot
        vide supprimera tout le corpus de l'index, volontairement.
    """

    def __init__(
        self,
        *,
        racine_miroir: str | Path,
        autoriser_snapshot_vide: bool = False,
    ) -> None:
        self._racine_miroir = Path(racine_miroir)
        self._autoriser_snapshot_vide = autoriser_snapshot_vide

    @property
    def racine_miroir(self) -> Path:
        return self._racine_miroir

    @abstractmethod
    def _recuperer(self, staging: Path) -> None:
        """
        Remplit `staging` avec **l'intégralité** des documents actuellement
        exposés par la source (noms de fichiers stables d'un run à l'autre).

        `staging` est un répertoire neuf et vide, voisin du miroir publié.

        Contrat : retourner normalement **signifie** « tous les documents de
        la source sont présents dans `staging` ». Toute récupération
        incomplète — même partielle, même un seul document manquant — doit
        lever. Ne jamais retourner sur un état douteux.
        """

    @contextmanager
    def materialiser(self) -> Iterator[Path]:
        publie = self._racine_miroir
        publie.parent.mkdir(parents=True, exist_ok=True)
        staging = publie.with_name(f".{publie.name}{_SUFFIXE_STAGING}{uuid4().hex}")

        if staging.exists():  # pragma: no cover - collision d'uuid quasi impossible
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        try:
            self._recuperer(staging)
            vide = not any(staging.rglob("*"))
            if vide and not self._autoriser_snapshot_vide:
                raise ErreurSource(
                    f"{type(self).__name__}._recuperer a produit un snapshot vide. "
                    "Interprété comme une récupération ratée : rien n'est publié, "
                    "le miroir précédent reste inchangé. Passer "
                    "autoriser_snapshot_vide=True si un corpus vide est un état "
                    "légitime de la source."
                )
        except ErreurSource:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        except Exception as exc:  # noqa: BLE001 — toute panne de récupération
            shutil.rmtree(staging, ignore_errors=True)
            raise ErreurSource(
                f"Récupération incomplète de {type(self).__name__} : {exc!r}. "
                "Snapshot non publié ; le miroir précédent reste inchangé."
            ) from exc

        # Succès complet uniquement : publication atomique, puis exposition.
        _publier_atomiquement(staging, publie)
        yield publie


__all__ = ["SnapshotDocumentSource"]
