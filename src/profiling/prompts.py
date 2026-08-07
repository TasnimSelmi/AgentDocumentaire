"""
Prompts du module de profils de domaine.

Le prompt est centralisé ici pour rester lisible, testable et réutilisable
avec n'importe quel fournisseur de LLM. Il ne contient aucune valeur propre
à un secteur : les exemples sont donnés sous forme d'emplacements neutres
afin de ne pas orienter le modèle vers un domaine particulier.

Le gabarit est envoyé en un seul message : la consigne « uniquement du JSON »
figure dans les règles, ce qui évite de dépendre du format de message système
propre à chaque fournisseur.
"""

from __future__ import annotations

from src.profiling.models import NB_MOTS_CLES_MAX, NB_MOTS_CLES_MIN

_GABARIT = """\
Un administrateur déclare le domaine métier suivant :

DOMAINE
{domaine}

TÂCHE
Produis un profil de domaine décrivant ce champ métier et son vocabulaire.

FORMAT DE SORTIE
Un unique objet JSON, exactement avec ces clés :

{{
  "profile_name": "<identifiant technique court>",
  "domain": "<intitulé lisible du domaine>",
  "description": "<description générique du champ métier>",
  "keywords": ["<concept 1>", "<concept 2>", "<concept 3>"],
  "output_language": "{langue}"
}}

RÈGLES
1. Le profil décrit un domaine de connaissance, jamais un corpus, jamais un
   ensemble de documents précis.
2. N'invente aucun document, aucun fichier, aucune source.
3. Ne propose aucune catégorie documentaire, aucune métadonnée, aucun champ
   d'extraction, aucune instruction de recherche.
4. "profile_name" : identifiant technique court, en minuscules non accentuées,
   composé uniquement de lettres, chiffres, tirets ou underscores, sans espace
   et sans caractère de chemin.
5. "domain" : intitulé lisible, une ligne, dans la langue demandée.
6. "description" : deux à quatre phrases décrivant le champ métier couvert et
   son vocabulaire. Ne prétends pas connaître les documents qui seront indexés.
7. "keywords" : entre {mini} et {maxi} concepts importants du domaine. Ce sont
   des termes ou expressions courtes du vocabulaire métier, pas des phrases.
   Aucun doublon.
8. "output_language" : le code de langue demandé, tel quel.
9. Rédige "domain", "description" et "keywords" dans la langue « {langue} ».
10. Réponds uniquement par l'objet JSON.
"""


def build_domain_profile_prompt(
    domain: str,
    output_language: str = "fr",
) -> str:
    """
    Construit le prompt de suggestion d'un profil de domaine.

    Args:
        domain: domaine métier saisi par l'administrateur.
        output_language: code de langue attendu pour le profil produit.

    Returns:
        Le prompt complet, prêt à être envoyé au LLM.

    Raises:
        ValueError: si le domaine est vide.
    """
    domaine = (domain or "").strip()
    if not domaine:
        raise ValueError("Le domaine est obligatoire pour construire le prompt.")

    return _GABARIT.format(
        domaine=domaine,
        langue=(output_language or "fr").strip().lower(),
        mini=NB_MOTS_CLES_MIN,
        maxi=NB_MOTS_CLES_MAX,
    )


__all__ = ["build_domain_profile_prompt"]
