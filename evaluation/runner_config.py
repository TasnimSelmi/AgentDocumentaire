"""
Lanceur d'évaluation avec surcharge de configuration en mémoire.

Deux des six leviers d'ablation — hybride/dense seul et expansion par
voisins — ne sont pas des paramètres de `rechercher_passages()` mais des
clés de `config/default.yaml`.

Trois solutions étaient possibles, deux sont mauvaises :

  1. Modifier config/default.yaml puis le restaurer.
     Rejeté : un plantage en cours d'ablation laisserait le dépôt dans un
     état modifié, avec un pipeline silencieusement dégradé.

  2. Passer par une variable d'environnement.
     Rejeté : src/config.py code le chemin en dur
     (`DOSSIER_CONFIG / "default.yaml"`) et ne lit aucune variable.

  3. Rediriger `_lire_yaml` en mémoire, avant tout appel à
     `get_config_technique()`.  ← retenu

La redirection est posée AVANT l'import des modules du pipeline, donc avant
que le cache `lru_cache(maxsize=1)` ne soit rempli. Le fichier du dépôt est
lu, jamais écrit.

Ce module n'est pas destiné à un usage direct : `run_ablation.py` l'invoque
en sous-processus.

Usage interne :
    python -m evaluation._runner_config \\
        --surcharges '{"qdrant.sparse_active": false}' \\
        -- <module> <arguments...>
"""

from __future__ import annotations

import argparse
import copy
import json
import runpy
import sys
from pathlib import Path
from typing import Any

RACINE_PROJET = Path(__file__).resolve().parent.parent
if str(RACINE_PROJET) not in sys.path:
    sys.path.insert(0, str(RACINE_PROJET))


def appliquer_surcharge(donnees: dict[str, Any], chemin: str, valeur: Any) -> None:
    """Applique une surcharge 'a.b.c' sur un dict imbriqué, en échouant fort."""
    cles = chemin.split(".")
    courant = donnees
    for cle in cles[:-1]:
        if not isinstance(courant.get(cle), dict):
            raise KeyError(
                f"Chemin de configuration introuvable : {chemin!r} "
                f"(bloqué sur {cle!r})."
            )
        courant = courant[cle]
    if cles[-1] not in courant:
        raise KeyError(
            f"Clé de configuration introuvable : {chemin!r}. "
            "L'ablation serait silencieusement identique à la référence."
        )
    courant[cles[-1]] = valeur


def installer_surcharges(surcharges: dict[str, Any]) -> None:
    """
    Remplace `src.config._lire_yaml` par une version qui applique les surcharges.

    Seul `default.yaml` est concerné : les profils techniques et les profils
    de domaine restent lus normalement.
    """
    import src.config as configuration

    lecture_originale = configuration._lire_yaml
    cible = (configuration.DOSSIER_CONFIG / "default.yaml").resolve()

    def _lire_yaml_surcharge(chemin: Path) -> dict[str, Any]:
        donnees = lecture_originale(chemin)
        try:
            concerne = Path(chemin).resolve() == cible
        except OSError:  # pragma: no cover
            concerne = False
        if not concerne:
            return donnees

        donnees = copy.deepcopy(donnees)
        for cle, valeur in surcharges.items():
            appliquer_surcharge(donnees, cle, valeur)
        return donnees

    configuration._lire_yaml = _lire_yaml_surcharge

    # Le cache pourrait déjà être chaud si un import a eu lieu plus tôt.
    configuration.recharger_config()

    # Contrôle : la surcharge est-elle réellement visible ?
    technique = configuration.get_config_technique()
    for cle, attendu in surcharges.items():
        courant: Any = technique
        for morceau in cle.split("."):
            courant = getattr(courant, morceau)
        if courant != attendu:
            raise SystemExit(
                f"Surcharge non appliquée : {cle} vaut {courant!r} "
                f"au lieu de {attendu!r}. Ablation interrompue."
            )
    print(
        f"[runner] surcharges actives : {surcharges}",
        file=sys.stderr,
    )


def main() -> int:
    parseur = argparse.ArgumentParser(add_help=False)
    parseur.add_argument("--surcharges", default="{}")
    arguments, reste = parseur.parse_known_args()

    if reste and reste[0] == "--":
        reste = reste[1:]
    if not reste:
        print("Aucun module à exécuter.", file=sys.stderr)
        return 2

    surcharges = json.loads(arguments.surcharges)
    if surcharges:
        installer_surcharges(surcharges)

    module, *arguments_module = reste
    sys.argv = [module, *arguments_module]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())