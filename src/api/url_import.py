"""
Import d'un document depuis une URL publique
(`POST /corpora/{corpus_id}/import-url`).

Protections SSRF minimales mais réelles :

- schéma `http`/`https` uniquement (jamais `file://`, `ftp://`, ni un chemin
  filesystem) ;
- résolution DNS + rejet des plages privées / loopback / link-local /
  multicast / réservées / non spécifiées, revalidées à CHAQUE saut de
  redirection (jamais seulement sur l'URL initiale) ;
- nombre de redirections borné ;
- taille de téléchargement bornée EN FLUX (jamais une confiance dans
  l'en-tête `Content-Length` seul, qui peut mentir ou être absent) ;
- extension de destination limitée aux formats réellement supportés par le
  pipeline d'ingestion (`ConfigIngestion.extensions_supportees`) ;
- timeouts explicites (connexion et lecture) ;
- aucune exception réseau brute n'atteint jamais l'appelant : tout est
  requalifié en `ImportUrlInvalide`, message générique sans détail interne.

Limite assumée, documentée à la livraison : la validation DNS a lieu avant
la connexion effective ; un « DNS rebinding » entre cette vérification et la
connexion réelle n'est pas neutralisé ici (nécessiterait un contrôle réseau
egress dédié — hors périmètre applicatif d'un MVP, à traiter par les règles
réseau internes d'INSY2S si ce risque est jugé significatif pour l'usage
retenu).
"""

from __future__ import annotations

import ipaddress
import socket
from pathlib import PurePosixPath
from urllib.parse import urlsplit

import httpx

TIMEOUT_CONNEXION_S = 10.0
TIMEOUT_LECTURE_S = 30.0
MAX_REDIRECTIONS = 5
#: Défaut si l'appelant n'en fournit pas (alignée sur `ingestion.taille_max_mo`).
TAILLE_TELECHARGEMENT_MAX_OCTETS_PAR_DEFAUT = 50 * 1024 * 1024

_CODES_REDIRECTION = frozenset({301, 302, 303, 307, 308})


class ImportUrlInvalide(ValueError):
    """URL refusée avant toute requête réseau, ou téléchargement refusé /
    interrompu / trop volumineux — jamais une exception réseau brute."""


def _hote_autorise(hote: str) -> None:
    """Résout `hote` et refuse toute adresse privée/loopback/link-local/
    multicast/réservée/non spécifiée. Lève `ImportUrlInvalide` sinon."""
    try:
        infos = socket.getaddrinfo(hote, None)
    except socket.gaierror as exc:
        raise ImportUrlInvalide(f"Nom d'hôte introuvable : {hote!r}.") from exc

    if not infos:
        raise ImportUrlInvalide(f"Nom d'hôte introuvable : {hote!r}.")

    for *_rest, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ImportUrlInvalide(f"Destination réseau interdite : {hote!r}.")


def _valider_url(url: str) -> None:
    morceaux = urlsplit(url)
    if morceaux.scheme not in ("http", "https"):
        raise ImportUrlInvalide("Seuls les schémas http/https sont autorisés.")
    if not morceaux.hostname:
        raise ImportUrlInvalide("URL invalide : hôte manquant.")
    _hote_autorise(morceaux.hostname)


def nom_fichier_depuis_url(url: str, *, extensions_autorisees: set[str]) -> str:
    """
    Dérive un nom de fichier sûr depuis le CHEMIN de l'URL demandée (jamais
    depuis une redirection suivie). Lève `ImportUrlInvalide` si aucun nom
    exploitable n'en ressort, ou si son extension n'est pas dans
    `extensions_autorisees`.
    """
    chemin = urlsplit(url).path
    nom = PurePosixPath(chemin).name
    if not nom:
        raise ImportUrlInvalide("Impossible de déterminer un nom de fichier depuis l'URL.")

    ext = PurePosixPath(nom).suffix.lower()
    if ext not in extensions_autorisees:
        raise ImportUrlInvalide(f"Format non supporté : {ext!r}.")

    nom_nettoye = nom.replace("/", "_").replace("\\", "_").strip()
    if not nom_nettoye or nom_nettoye in (".", ".."):
        raise ImportUrlInvalide("Nom de fichier invalide.")
    return nom_nettoye


def telecharger(
    url: str,
    *,
    extensions_autorisees: set[str],
    taille_max_octets: int = TAILLE_TELECHARGEMENT_MAX_OCTETS_PAR_DEFAUT,
    client: httpx.Client | None = None,
) -> tuple[str, bytes]:
    """
    Télécharge `url` après validation stricte, en bornant la taille lue en
    flux. `client` est injectable (tests contre un serveur HTTP local
    contrôlé, jamais Internet). Renvoie `(nom_fichier, contenu)`.

    Lève `ImportUrlInvalide` sur tout refus — schéma, hôte, redirection vers
    une destination interdite, trop de redirections, réponse non-200, taille
    dépassée, contenu vide, ou toute erreur réseau (requalifiée, jamais
    propagée telle quelle)."""
    _valider_url(url)  # schéma + hôte, avant même de dériver un nom de fichier
    nom_fichier = nom_fichier_depuis_url(url, extensions_autorisees=extensions_autorisees)

    ferme_a_la_fin = client is None
    http_client = client or httpx.Client(
        follow_redirects=False,
        timeout=httpx.Timeout(TIMEOUT_LECTURE_S, connect=TIMEOUT_CONNEXION_S),
    )
    try:
        url_courante = url
        for _ in range(MAX_REDIRECTIONS + 1):
            _valider_url(url_courante)
            try:
                with http_client.stream("GET", url_courante) as reponse:
                    if reponse.status_code in _CODES_REDIRECTION:
                        emplacement = reponse.headers.get("location")
                        if not emplacement:
                            raise ImportUrlInvalide("Redirection sans destination.")
                        url_courante = str(httpx.URL(url_courante).join(emplacement))
                        continue

                    if reponse.status_code != 200:
                        raise ImportUrlInvalide(
                            f"Le serveur distant a répondu {reponse.status_code}."
                        )

                    contenu = bytearray()
                    for morceau in reponse.iter_bytes(chunk_size=65536):
                        contenu.extend(morceau)
                        if len(contenu) > taille_max_octets:
                            raise ImportUrlInvalide(
                                "Le document dépasse la taille maximale autorisée."
                            )

                    if not contenu:
                        raise ImportUrlInvalide("Le document téléchargé est vide.")
                    return nom_fichier, bytes(contenu)
            except httpx.HTTPError as exc:
                raise ImportUrlInvalide(
                    f"Téléchargement impossible : {type(exc).__name__}."
                ) from exc

        raise ImportUrlInvalide("Trop de redirections.")
    finally:
        if ferme_a_la_fin:
            http_client.close()


__all__ = [
    "ImportUrlInvalide",
    "telecharger",
    "nom_fichier_depuis_url",
    "MAX_REDIRECTIONS",
    "TAILLE_TELECHARGEMENT_MAX_OCTETS_PAR_DEFAUT",
]
