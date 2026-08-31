"""
`LocalDocumentSource` — adaptateur MVP : la source *est* un dossier local
déjà présent sur la machine.

C'est le comportement historique du pipeline, exprimé derrière le contrat
`DocumentSource`. `materialiser()` renvoie le dossier tel quel, sans copie ni
répertoire temporaire : appeler

    IngestionService().sync(LocalDocumentSource(chemin))

est strictement équivalent à l'appel direct `ingerer(dossier=chemin)`.

Invariant de snapshot — satisfait trivialement
----------------------------------------------
Il n'y a **aucune étape de récupération faillible** : le dossier de
l'utilisateur *est* le snapshot, toujours cohérent. `materialiser()` ne fait
que vérifier que le chemin est bien un répertoire (sinon `ErreurSource`, sans
rien exposer). La cohérence d'écriture du dossier lui-même relève de
l'utilisateur, exactement comme pour l'appel direct `ingerer(dossier=…)` —
P2.2 ne change pas ce point.

`inventaire()` n'appartient pas au contrat `DocumentSource` : c'est une
commodité concrète (observabilité, tests) qui réutilise la découverte du
socle (`decouvrir_fichiers`) — jamais une seconde implémentation.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src.config import get_config_technique
from src.rag.ingestion import decouvrir_fichiers
from src.sources.base import ErreurSource


class LocalDocumentSource:
    """
    Source documentaire adossée à un répertoire local.

    Paramètres
    ----------
    racine :
        Dossier contenant les documents (parcouru récursivement par le socle).
    extensions :
        Restriction éventuelle des extensions prises en compte par
        `inventaire()`. `None` → `ingestion.extensions_supportees` de la
        configuration technique, exactement comme le pipeline.

    `materialiser()` ne filtre rien : le filtrage (extension, taille) reste
    entièrement dans le socle au moment de l'ingestion. `extensions` n'agit
    que sur `inventaire()`.
    """

    def __init__(
        self,
        racine: str | Path,
        *,
        extensions: list[str] | None = None,
    ) -> None:
        self._racine = Path(racine)
        self._extensions = extensions

    @property
    def racine(self) -> Path:
        return self._racine

    def _resolue(self) -> Path:
        chemin = self._racine.resolve()
        if not chemin.exists():
            raise ErreurSource(f"Répertoire source introuvable : {self._racine}")
        if not chemin.is_dir():
            raise ErreurSource(
                f"La source locale doit être un répertoire : {self._racine}"
            )
        return chemin

    def _extensions_effectives(self) -> list[str]:
        if self._extensions is not None:
            return [e.lower() for e in self._extensions]
        return list(get_config_technique().ingestion.extensions_supportees)

    def inventaire(self) -> list[str]:
        """
        Identifiants stables des documents que le pipeline ingérerait depuis
        ce dossier : chemins relatifs POSIX depuis la racine, triés.

        Réutilise `decouvrir_fichiers` du socle (mêmes filtres d'extension et
        de taille). Ne lit pas le contenu des fichiers. Lève `ErreurSource`
        si le dossier est absent ou n'est pas un répertoire.
        """
        racine = self._resolue()
        fichiers = decouvrir_fichiers(racine, self._extensions_effectives())
        return sorted(f.relative_to(racine).as_posix() for f in fichiers)

    @contextmanager
    def materialiser(self) -> Iterator[Path]:
        """
        Rend le dossier local prêt à ingérer : ici, le dossier lui-même.

        Aucune copie, aucun répertoire temporaire, aucun nettoyage en sortie.
        Lève `ErreurSource` si le dossier est absent ou n'est pas un
        répertoire — avant que le pipeline ne touche à Qdrant.
        """
        yield self._resolue()


__all__ = ["LocalDocumentSource"]
