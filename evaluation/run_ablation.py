"""
Ablation study du retrieval.

Objectif : déterminer quelle composante apporte réellement de la valeur.
Pas d'améliorer les scores — aucune configuration n'est écrite dans le
dépôt, et le pipeline n'est jamais modifié.

Deux familles de leviers
------------------------
Leviers passés en PARAMÈTRE à `rechercher_passages()` — exécutés dans le
processus courant :

    utiliser_reranker      reranker ON/OFF
    appliquer_seuil        seuil de pertinence ON/OFF
    resolution_document    cloisonnement documentaire ON/OFF
    max_par_document       diversification

Leviers pilotés par la CONFIGURATION — exécutés en sous-processus :

    qdrant.sparse_active        hybride dense+sparse vs dense seul
    decoupage.voisins.actif     expansion par voisins ON/OFF

Pourquoi un sous-processus pour les seconds : `get_config_technique()` est
en `lru_cache(maxsize=1)`. Modifier la configuration dans un processus déjà
chaud n'aurait aucun effet, ou pire, un effet partiel selon l'ordre des
appels. Chaque configuration part donc d'un processus neuf, avec un fichier
de configuration temporaire — le `config/default.yaml` du dépôt n'est jamais
touché.

`src/config.py` code le chemin de default.yaml en dur et ne lit aucune
variable d'environnement : la surcharge passe donc par `_runner_config.py`,
qui redirige la lecture YAML en mémoire. Le fichier du dépôt n'est jamais
écrit, même si une ablation plante.

Exemple
-------
    python -m evaluation.run_ablation \\
        --questions evaluation/data/finance_esg.jsonl --nom ablation_v1

    # Ne lancer que les configurations rapides
    python -m evaluation.run_ablation \\
        --questions evaluation/data/finance_esg.jsonl --sans-sous-processus
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.common import (
    DOSSIER_RAPPORTS,
    RACINE_PROJET,
    charger_enregistrements,
    configurer_logs,
    ecrire_rapport,
    horodatage,
)

logger = logging.getLogger("evaluation.ablation")

CHEMIN_CONFIG_DEPOT = RACINE_PROJET / "config" / "default.yaml"


@dataclass
class Configuration:
    """
    Une configuration d'ablation.

    Attributes:
        nom: identifiant court, utilisé dans le tableau comparatif.
        description: ce que la configuration teste, en une ligne.
        parametres: passés à evaluate_retrieval_document (via CLI).
        surcharges_config: chemins pointés dans default.yaml à surcharger.
            Leur présence impose l'exécution en sous-processus.
    """

    nom: str
    description: str
    parametres: list[str] = field(default_factory=list)
    surcharges_config: dict[str, Any] = field(default_factory=dict)

    @property
    def necessite_sous_processus(self) -> bool:
        return bool(self.surcharges_config)


CONFIGURATIONS = [
    Configuration(
        "C0_reference",
        "Configuration actuelle du projet, telle quelle.",
    ),
    Configuration(
        "C1_sans_reranker",
        "Le reranker BGE apporte-t-il un gain mesurable ?",
        parametres=["--sans-reranker"],
    ),
    Configuration(
        "C2_avec_seuil",
        "Le seuil de pertinence filtre-t-il utilement ?",
        parametres=["--avec-seuil"],
    ),
    Configuration(
        "C3_sans_resolution_document",
        "Part du score due au cloisonnement documentaire plutôt qu'à la "
        "recherche sémantique.",
        parametres=["--sans-resolution-document"],
    ),
    Configuration(
        "C4_sans_diversification",
        "La limite de passages par document aide-t-elle ou nuit-elle ?",
        parametres=["--max-par-document", "999"],
    ),
    Configuration(
        "C5_dense_seul",
        "Le vecteur sparse de BGE-M3 apporte-t-il quelque chose au dense ?",
        surcharges_config={"qdrant.sparse_active": False},
    ),
    Configuration(
        "C6_sans_voisins",
        "L'expansion par chunks voisins améliore-t-elle la couverture ?",
        surcharges_config={"decoupage.voisins.actif": False},
    ),
]


def lancer_configuration(
    configuration: Configuration,
    *,
    questions: Path,
    dossier: Path,
    limite: int | None,
    arguments_communs: list[str],
) -> dict[str, Any] | None:
    """
    Exécute une configuration et retourne son résumé.

    Toujours via subprocess : cela garantit un état identique pour toutes les
    configurations, y compris celles qui n'ont pas besoin de surcharge. Sans
    cela, C0 partirait avec un cache de modèle chaud et les autres non, et les
    temps ne seraient plus comparables.
    """
    nom_rapport = f"ablation_{configuration.nom}"

    commande_module = [
        "evaluation.evaluate_retrieval_document",
        "--questions",
        str(questions),
        "--nom",
        nom_rapport,
        "--sortie",
        str(dossier),
        *arguments_communs,
        *configuration.parametres,
    ]
    if limite is not None:
        commande_module += ["--limite", str(limite)]

    if configuration.surcharges_config:
        # Surcharge de configuration : passage par le lanceur dédié.
        commande = [
            sys.executable,
            "-m",
            "evaluation._runner_config",
            "--surcharges",
            json.dumps(configuration.surcharges_config),
            "--",
            *commande_module,
        ]
        logger.info("  surcharges : %s", configuration.surcharges_config)
    else:
        commande = [sys.executable, "-m", *commande_module]

    logger.info("\u25b6 %s \u2014 %s", configuration.nom, configuration.description)

    processus = subprocess.run(
        commande,
        cwd=str(RACINE_PROJET),
        capture_output=True,
        text=True,
    )
    if processus.returncode != 0:
        logger.error(
            "%s a \u00e9chou\u00e9 (code %d).\n%s",
            configuration.nom,
            processus.returncode,
            processus.stderr[-2000:],
        )
        return None

    chemin_resultat = dossier / f"{nom_rapport}.json"
    if not chemin_resultat.exists():
        logger.error("Rapport absent : %s", chemin_resultat)
        return None

    donnees = json.loads(chemin_resultat.read_text(encoding="utf-8"))
    return donnees.get("resume", {})


def construire_tableau(
    resultats: dict[str, dict[str, Any]],
    reference: str = "C0_reference",
) -> list[dict[str, Any]]:
    """
    Tableau comparatif avec écart à la configuration de référence.

    L'écart, et non la valeur absolue, est ce qui répond à la question
    « ce composant sert-il à quelque chose ? ».
    """
    base = resultats.get(reference, {})
    colonnes = ("hit@1", "hit@3", "hit@5", "hit@10", "mrr@10", "recall@10")

    lignes: list[dict[str, Any]] = []
    for configuration in CONFIGURATIONS:
        resume = resultats.get(configuration.nom)
        if resume is None:
            lignes.append(
                {
                    "configuration": configuration.nom,
                    "description": configuration.description,
                    "statut": "ECHEC",
                }
            )
            continue

        ligne: dict[str, Any] = {
            "configuration": configuration.nom,
            "description": configuration.description,
            "statut": "ok",
            "questions": resume.get("evaluees"),
            "duree_moyenne_s": resume.get("duree_moyenne_secondes"),
        }
        for colonne in colonnes:
            valeur = resume.get(colonne)
            ligne[colonne] = valeur
            if configuration.nom != reference and valeur is not None:
                valeur_base = base.get(colonne)
                ligne[f"delta_{colonne}"] = (
                    round(valeur - valeur_base, 4)
                    if valeur_base is not None
                    else None
                )
        lignes.append(ligne)

    return lignes


def construire_parseur() -> argparse.ArgumentParser:
    parseur = argparse.ArgumentParser(
        description="Ablation study du retrieval, sans modifier le pipeline."
    )
    parseur.add_argument("--questions", type=Path, required=True)
    parseur.add_argument("--limite", type=int, default=None)
    parseur.add_argument("--nom", default=None)
    parseur.add_argument(
        "--sortie", type=Path, default=DOSSIER_RAPPORTS / "ablation"
    )
    parseur.add_argument("--top-k", type=int, default=10)
    parseur.add_argument("--candidats", type=int, default=None)
    parseur.add_argument(
        "--configurations",
        nargs="+",
        default=None,
        help="sous-ensemble de configurations à lancer",
    )
    parseur.add_argument(
        "--sans-sous-processus",
        action="store_true",
        help="ignore C5 et C6, qui exigent une surcharge de configuration",
    )
    parseur.add_argument("--verbose", action="store_true")
    return parseur


def main(argv: list[str] | None = None) -> int:
    arguments = construire_parseur().parse_args(argv)
    configurer_logs(arguments.verbose)

    if not arguments.questions.exists():
        logger.error("Jeu introuvable : %s", arguments.questions)
        return 1

    nombre = len(charger_enregistrements(arguments.questions, uniquement_answerable=True))
    logger.info("Jeu : %s (%d questions répondables)", arguments.questions, nombre)

    a_lancer = [
        configuration
        for configuration in CONFIGURATIONS
        if (
            arguments.configurations is None
            or configuration.nom in arguments.configurations
        )
        and not (
            arguments.sans_sous_processus
            and configuration.necessite_sous_processus
        )
    ]

    logger.info(
        "%d configuration(s) à évaluer, sur le même jeu et la même seed.",
        len(a_lancer),
    )

    arguments_communs = ["--top-k", str(arguments.top_k)]
    if arguments.candidats is not None:
        arguments_communs += ["--candidats", str(arguments.candidats)]

    resultats: dict[str, dict[str, Any]] = {}
    for configuration in a_lancer:
        resume = lancer_configuration(
            configuration,
            questions=arguments.questions,
            dossier=arguments.sortie,
            limite=arguments.limite,
            arguments_communs=arguments_communs,
        )
        if resume is not None:
            resultats[configuration.nom] = resume

    if not resultats:
        logger.error("Aucune configuration n'a abouti.")
        return 1

    tableau = construire_tableau(resultats)

    resume_global = {
        "jeu": str(arguments.questions),
        "configurations_lancees": list(resultats),
        "reference": "C0_reference",
        "top_k": arguments.top_k,
        "limite": arguments.limite,
    }

    nom = arguments.nom or f"ablation_{horodatage()}"
    ecrire_rapport(
        arguments.sortie, nom, resume=resume_global, details=tableau
    )

    # --- Tableau lisible ----------------------------------------------------
    print()
    entete = f"{'configuration':<30} {'Hit@1':>8} {'Hit@5':>8} {'MRR@10':>8} {'Δ Hit@1':>9} {'temps':>7}"
    print(entete)
    print("-" * len(entete))
    for ligne in tableau:
        if ligne["statut"] != "ok":
            print(f"{ligne['configuration']:<30} {'ÉCHEC':>8}")
            continue
        delta = ligne.get("delta_hit@1")
        print(
            f"{ligne['configuration']:<30} "
            f"{ligne.get('hit@1', 0):>8.4f} "
            f"{ligne.get('hit@5', 0):>8.4f} "
            f"{ligne.get('mrr@10', 0):>8.4f} "
            f"{('—' if delta is None else f'{delta:+.4f}'):>9} "
            f"{ligne.get('duree_moyenne_s', 0):>6.2f}s"
        )
    print()
    print(
        "  Un Δ proche de 0 signifie que le composant retiré n'apportait rien\n"
        "  de mesurable sur ce jeu. Un Δ fortement négatif le justifie.\n"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())