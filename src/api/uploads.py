"""
Validation d'un lot de fichiers uploadés (`POST /corpora/{corpus_id}/upload`)
— frontière de sécurité pure, aucun accès disque ici.

Toute validation porte sur l'INTÉGRALITÉ du lot AVANT la moindre écriture :
un lot invalide (format, taille, chemin) n'écrit RIEN — jamais un état
partiel dans le stockage géré du corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

#: Nombre maximal de fichiers par upload — raisonnable pour un dossier
#: utilisateur, pas pour un import massif (hors périmètre MVP).
MAX_FICHIERS_UPLOAD = 200

#: Taille totale maximale d'un upload, en Mo.
TAILLE_TOTALE_MAX_MO = 500


class UploadInvalide(ValueError):
    """Un fichier ou un chemin du lot uploadé ne respecte pas les règles de
    sécurité/format — la route qui lève ceci n'a encore rien écrit."""


@dataclass(frozen=True)
class FichierValide:
    """Un fichier du lot, déjà validé : chemin POSIX relatif garanti sans
    traversée de répertoire, format et taille déjà vérifiés."""

    chemin_relatif: Path
    contenu: bytes


def valider_chemin_relatif(chemin_relatif: str) -> Path:
    """
    Valide un chemin relatif déclaré par le client (position d'un fichier au
    sein du dossier uploadé). Refuse : chemin vide, chemin absolu (POSIX ou
    lettre de lecteur Windows), tout segment `..` ou vide. Les antislashs
    sont acceptés en entrée (client Windows) puis normalisés en `/`.

    Lève `UploadInvalide` sinon ; ne touche jamais au disque.
    """
    brut = (chemin_relatif or "").strip().replace("\\", "/")
    if not brut:
        raise UploadInvalide("Chemin de fichier vide.")

    p = PurePosixPath(brut)
    parts = p.parts
    if not parts:
        raise UploadInvalide(f"Chemin invalide : {chemin_relatif!r}.")
    if p.is_absolute():
        raise UploadInvalide(f"Chemin absolu refusé : {chemin_relatif!r}.")
    if any(part in ("..", "") for part in parts):
        raise UploadInvalide(
            f"Chemin invalide (traversée de répertoire) : {chemin_relatif!r}."
        )
    if ":" in parts[0]:
        raise UploadInvalide(f"Chemin invalide : {chemin_relatif!r}.")

    return Path(*parts)


def valider_lot(
    fichiers: list[tuple[str, bytes]],
    *,
    extensions_autorisees: set[str],
    taille_max_fichier_octets: int,
    max_fichiers: int = MAX_FICHIERS_UPLOAD,
    taille_totale_max_octets: int = TAILLE_TOTALE_MAX_MO * 1024 * 1024,
) -> list[FichierValide]:
    """
    Valide intégralement un lot `(chemin_relatif, contenu)` avant tout accès
    disque. Le premier problème rencontré lève `UploadInvalide` — le lot
    entier est alors rejeté, sans qu'aucun fichier n'ait été écrit.
    """
    if not fichiers:
        raise UploadInvalide("Aucun fichier reçu.")
    if len(fichiers) > max_fichiers:
        raise UploadInvalide(f"Trop de fichiers (maximum {max_fichiers}).")

    valides: list[FichierValide] = []
    taille_totale = 0
    for chemin_relatif, contenu in fichiers:
        chemin = valider_chemin_relatif(chemin_relatif)
        if not contenu:
            raise UploadInvalide(f"Fichier vide : {chemin_relatif!r}.")
        if len(contenu) > taille_max_fichier_octets:
            raise UploadInvalide(f"Fichier trop volumineux : {chemin_relatif!r}.")
        if chemin.suffix.lower() not in extensions_autorisees:
            raise UploadInvalide(
                f"Format non supporté : {chemin.suffix!r} ({chemin_relatif!r})."
            )
        taille_totale += len(contenu)
        if taille_totale > taille_totale_max_octets:
            raise UploadInvalide("Taille totale de l'upload trop importante.")
        valides.append(FichierValide(chemin_relatif=chemin, contenu=contenu))

    return valides


def ecrire_lot(valides: list[FichierValide], *, racine: Path) -> None:
    """
    Écrit un lot déjà validé sous `racine` — toujours dérivée côté serveur
    (`src.rag.corpus.dossier_managed_pour_corpus`), jamais un chemin fourni
    par le client. Additif : n'efface jamais le contenu déjà présent.
    """
    racine.mkdir(parents=True, exist_ok=True)
    for fichier in valides:
        cible = racine / fichier.chemin_relatif
        cible.parent.mkdir(parents=True, exist_ok=True)
        cible.write_bytes(fichier.contenu)


__all__ = [
    "UploadInvalide",
    "FichierValide",
    "MAX_FICHIERS_UPLOAD",
    "TAILLE_TOTALE_MAX_MO",
    "valider_chemin_relatif",
    "valider_lot",
    "ecrire_lot",
]
