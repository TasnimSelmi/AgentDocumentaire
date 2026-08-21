"""
Interface en ligne de commande des profils de domaine.

    python -m src.profiling.cli suggest --domain "Finance et comptabilité"
    python -m src.profiling.cli suggest --domain "Droit du travail" --save --yes
    python -m src.profiling.cli suggest --domain "الأمن السيبراني" \
        --name cybersecurite_ar --language ar --save --yes
    python -m src.profiling.cli list
    python -m src.profiling.cli show finance
    python -m src.profiling.cli delete finance

Rien n'est enregistré sans confirmation explicite : `--save` en mode
interactif demande toujours une validation, `--yes` la fournit d'avance pour
les scripts et les tests.

La langue vient de `--language` si l'option est fournie, sinon de
`DOMAIN_PROFILE_OUTPUT_LANGUAGE`. `--name` impose l'identifiant technique,
ce qui permet de profiler un domaine écrit en arabe ou en chinois sans
relâcher la contrainte ASCII qui protège les noms de fichiers.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

from src.profiling.exceptions import DomainProfileError
from src.profiling.loader import langue_sortie_par_defaut
from src.profiling.models import (
    DomainProfile,
    normaliser_mots_cles,
    normaliser_nom_profil,
)
from src.profiling.service import suggest_domain_profile
from src.profiling.storage import (
    delete_domain_profile,
    domain_profile_exists,
    list_domain_profiles,
    load_domain_profile,
    save_domain_profile,
)

logger = logging.getLogger(__name__)


def _configurer_sortie() -> None:
    """Force l'UTF-8 en sortie, notamment sur les consoles Windows."""
    for flux in (sys.stdout, sys.stderr):
        reconfigurer = getattr(flux, "reconfigure", None)
        if callable(reconfigurer):
            try:
                reconfigurer(encoding="utf-8")
            except (ValueError, OSError):  # flux non reconfigurable
                pass


def _afficher_profil(profil: DomainProfile) -> None:
    print("-" * 62)
    print(f"profile_name    : {profil.profile_name}")
    print(f"domain          : {profil.domain}")
    print(f"description     : {profil.description}")
    print(f"output_language : {profil.output_language}")
    print("keywords        :")
    for mot in profil.keywords:
        print(f"    - {mot}")
    print("-" * 62)


def _demander(question: str, defaut: str) -> str:
    """Saisie avec valeur par défaut ; une entrée vide conserve la valeur."""
    reponse = input(f"{question} [{defaut}] : ").strip()
    return reponse or defaut


def _confirmer(question: str) -> bool:
    reponse = input(f"{question} [o/N] : ").strip().lower()
    return reponse in {"o", "oui", "y", "yes"}


def _editer_profil(profil: DomainProfile) -> DomainProfile:
    """Permet à l'administrateur de corriger chaque champ du profil proposé."""
    print("\nModification du profil (Entrée = conserver la valeur proposée).")
    nom = _demander("profile_name", profil.profile_name)
    domaine = _demander("domain", profil.domain)
    description = _demander("description", profil.description)
    langue = _demander("output_language", profil.output_language)
    mots = _demander("keywords (séparés par des virgules)", ", ".join(profil.keywords))

    return DomainProfile(
        profile_name=nom,
        domain=domaine,
        description=description,
        keywords=normaliser_mots_cles(mots.split(",")),
        output_language=langue,
    )


def _commande_suggest(args: argparse.Namespace) -> int:
    domaine = args.domain
    if not domaine:
        domaine = input("Domaine métier : ").strip()
    if not domaine:
        print("Aucun domaine saisi, abandon.", file=sys.stderr)
        return 2

    # --language l'emporte sur la configuration ; sans option, la langue vient
    # de DOMAIN_PROFILE_OUTPUT_LANGUAGE.
    langue = args.language or langue_sortie_par_defaut()

    # Le nom explicite est validé avant l'appel au LLM : une saisie invalide
    # ne doit pas coûter une génération.
    nom_impose: str | None = None
    if args.name:
        nom_impose = normaliser_nom_profil(args.name)

    profil = suggest_domain_profile(domaine, output_language=langue)

    if nom_impose is not None and nom_impose != profil.profile_name:
        # Reconstruction plutôt que model_copy : le modèle revalide l'ensemble.
        profil = DomainProfile(**{**profil.model_dump(), "profile_name": nom_impose})

    print("\nProfil proposé par le LLM :")
    _afficher_profil(profil)

    interactif = not args.yes
    if interactif and _confirmer("Modifier ce profil ?"):
        profil = _editer_profil(profil)
        print("\nProfil après modification :")
        _afficher_profil(profil)

    if not args.save:
        print("Profil non enregistré (ajoute --save pour l'enregistrer).")
        return 0

    if interactif and not _confirmer(f"Enregistrer le profil '{profil.profile_name}' ?"):
        print("Annulé : aucun fichier créé.")
        return 0

    if domain_profile_exists(profil.profile_name) and not args.overwrite:
        print(
            f"Un profil '{profil.profile_name}' existe déjà. "
            "Relance avec --overwrite pour le remplacer.",
            file=sys.stderr,
        )
        return 1

    chemin = save_domain_profile(profil, overwrite=args.overwrite)
    print(f"Profil enregistré : {chemin}")
    return 0


def _commande_list(_: argparse.Namespace) -> int:
    profils = list_domain_profiles()
    if not profils:
        print("Aucun profil de domaine enregistré.")
        return 0
    for profil in profils:
        print(f"{profil.profile_name:<24} {profil.domain}")
    return 0


def _commande_show(args: argparse.Namespace) -> int:
    _afficher_profil(load_domain_profile(args.profile_name))
    return 0


def _commande_delete(args: argparse.Namespace) -> int:
    if not args.yes and not _confirmer(f"Supprimer le profil '{args.profile_name}' ?"):
        print("Annulé.")
        return 0
    if delete_domain_profile(args.profile_name):
        print(f"Profil supprimé : {args.profile_name}")
        return 0
    print(f"Aucun profil nommé '{args.profile_name}'.", file=sys.stderr)
    return 1


def construire_parseur() -> argparse.ArgumentParser:
    parseur = argparse.ArgumentParser(
        prog="python -m src.profiling.cli",
        description="Gestion des profils de domaine.",
    )
    sous = parseur.add_subparsers(dest="commande", required=True)

    p_suggest = sous.add_parser("suggest", help="Proposer un profil via le LLM.")
    p_suggest.add_argument("--domain", help="Domaine métier saisi par l'administrateur.")
    p_suggest.add_argument(
        "--language",
        default=None,
        help=(
            "Langue de sortie (fr, en, ar...). "
            "Par défaut : DOMAIN_PROFILE_OUTPUT_LANGUAGE."
        ),
    )
    p_suggest.add_argument(
        "--name",
        default=None,
        help=(
            "Identifiant technique imposé, en ASCII minuscule. Utile pour un "
            "domaine écrit dans un alphabet non latin."
        ),
    )
    p_suggest.add_argument("--save", action="store_true", help="Enregistrer le profil.")
    p_suggest.add_argument(
        "--yes",
        action="store_true",
        help="Mode non interactif : aucune question posée.",
    )
    p_suggest.add_argument(
        "--overwrite",
        action="store_true",
        help="Autoriser le remplacement d'un profil existant.",
    )
    p_suggest.set_defaults(fonction=_commande_suggest)

    p_list = sous.add_parser("list", help="Lister les profils enregistrés.")
    p_list.set_defaults(fonction=_commande_list)

    p_show = sous.add_parser("show", help="Afficher un profil.")
    p_show.add_argument("profile_name")
    p_show.set_defaults(fonction=_commande_show)

    p_delete = sous.add_parser("delete", help="Supprimer un profil.")
    p_delete.add_argument("profile_name")
    p_delete.add_argument("--yes", action="store_true", help="Ne pas demander confirmation.")
    p_delete.set_defaults(fonction=_commande_delete)

    return parseur


def main(argv: Sequence[str] | None = None) -> int:
    _configurer_sortie()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    args = construire_parseur().parse_args(argv)
    try:
        return int(args.fonction(args))
    except DomainProfileError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Entrée invalide : {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interruption utilisateur
        print("\nInterrompu.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
