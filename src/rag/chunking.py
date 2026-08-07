"""
Découpage structure-aware.

Le découpage récursif par taille fixe convient au texte narratif mais coupe
mal les tableaux : l'en-tête part dans un chunk, les lignes dans le suivant,
et l'association entre une colonne et une valeur est perdue. Le retrieval
trouve alors le bon document et la bonne page, mais pas la bonne ligne.

Ce module segmente d'abord une page en blocs (titre, paragraphe, liste,
tableau, texte brut), puis applique à chaque type le traitement qui lui
convient : découpage par lignes avec en-tête répété pour les tableaux,
découpage récursif pour le reste.

Aucune règle n'est propre à un secteur, à une entreprise ou à un document :
la détection repose uniquement sur la forme du texte produit par les loaders.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence

logger = logging.getLogger(__name__)

# Types de blocs reconnus. « texte » est le fallback quand aucune structure
# n'est détectable (page sans ligne vide, OCR brut, etc.).
TYPE_TITRE = "titre"
TYPE_PARAGRAPHE = "paragraphe"
TYPE_LISTE = "liste"
TYPE_TABLEAU = "tableau"
TYPE_TEXTE = "texte"

# Marqueurs de liste les plus courants, toutes langues confondues.
_MOTIF_PUCE = re.compile(r"^\s*(?:[-*•·–—o]|\(?\d{1,3}[.)]|[a-zA-Z][.)])\s+\S")

# Ligne de séparation Markdown : |---|:---:|
_MOTIF_SEPARATEUR_MD = re.compile(r"^:?-{2,}:?$")

# Ponctuation qui disqualifie une ligne comme titre.
_FINS_DE_PHRASE = (".", ";", ",", "!", "?")

LONGUEUR_TITRE_MAX = 120


class SplitterTexte(Protocol):
    """Découpeur de texte narratif (l'implémentation réelle est LangChain)."""

    def split_text(self, text: str) -> list[str]:  # pragma: no cover - protocole
        ...


@dataclass
class Chunk:
    """
    Un fragment de texte, avant vectorisation.

    Les trois premiers champs existaient déjà ; les suivants sont optionnels
    et restent à ``None`` quand l'information n'est pas disponible, ce qui
    préserve la compatibilité avec les payloads Qdrant existants.
    """

    index: int
    texte: str
    page: int | None = None
    type_bloc: str | None = None
    section_title: str | None = None
    parent_id: str | None = None
    table_id: str | None = None
    ordre_dans_parent: int | None = None
    is_table: bool = False
    header_repeated: bool = False


@dataclass
class Bloc:
    """Une unité structurelle repérée dans une page."""

    type_bloc: str
    lignes: list[str]
    page: int
    section_title: str | None = None
    table_id: str | None = None
    cellules: list[list[str]] = field(default_factory=list)

    @property
    def texte(self) -> str:
        return "\n".join(self.lignes).strip()

    @property
    def est_tableau(self) -> bool:
        return self.type_bloc == TYPE_TABLEAU


# ===========================================================================
# Détection de structure
# ===========================================================================


def decouper_cellules(ligne: str, separateur: str = "|") -> list[str]:
    """
    Découpe une ligne tabulaire en cellules.

    Tolère les deux conventions produites par les loaders : « a | b | c »
    (docx, xlsx, csv) et « |a|b|c| » (Markdown).
    """
    brut = ligne.strip()
    if brut.startswith(separateur):
        brut = brut[len(separateur) :]
    if brut.endswith(separateur):
        brut = brut[: -len(separateur)]
    return [cellule.strip() for cellule in brut.split(separateur)]


def est_ligne_tabulaire(ligne: str, colonnes_min: int = 2) -> bool:
    """Une ligne est tabulaire si elle expose au moins deux cellules."""
    if "|" not in ligne:
        return False
    cellules = decouper_cellules(ligne)
    if len(cellules) < colonnes_min:
        return False
    # Une ligne dont toutes les cellules sont vides n'apporte rien.
    return any(cellule for cellule in cellules)


def _est_separateur_markdown(ligne: str) -> bool:
    cellules = decouper_cellules(ligne)
    return bool(cellules) and all(
        _MOTIF_SEPARATEUR_MD.match(cellule) for cellule in cellules if cellule
    )


def _est_titre(lignes: Sequence[str]) -> bool:
    """
    Un titre est un groupe d'une seule ligne, courte et sans ponctuation
    finale de phrase. La détection reste volontairement conservatrice : un
    faux négatif fait juste perdre un `section_title`, sans rien casser.
    """
    if len(lignes) != 1:
        return False
    ligne = lignes[0].strip()
    if not ligne or len(ligne) > LONGUEUR_TITRE_MAX:
        return False
    if ligne.startswith("#"):
        return True
    if est_ligne_tabulaire(ligne) or _MOTIF_PUCE.match(ligne):
        return False
    return not ligne.endswith(_FINS_DE_PHRASE)


def _est_liste(lignes: Sequence[str]) -> bool:
    """Un groupe est une liste si la majorité de ses lignes sont des puces."""
    utiles = [ligne for ligne in lignes if ligne.strip()]
    if len(utiles) < 2:
        return False
    puces = sum(1 for ligne in utiles if _MOTIF_PUCE.match(ligne))
    return puces >= max(2, len(utiles) // 2)


def _nettoyer_titre(ligne: str) -> str:
    return ligne.lstrip("#").strip()


def segmenter_page(
    texte: str,
    page: int,
    *,
    lignes_table_min: int = 2,
    prefixe_table: str = "t",
) -> list[Bloc]:
    """
    Segmente le texte d'une page en blocs structurels.

    Les groupes sont d'abord délimités par les lignes vides, puis chaque
    groupe est scanné pour en extraire les suites de lignes tabulaires.
    Un tableau collé à son titre reste donc rattaché à ce titre.
    """
    blocs: list[Bloc] = []
    section_courante: str | None = None
    compteur_table = 0

    for groupe in _groupes_de_lignes(texte):
        for segment, tabulaire in _separer_segments_tabulaires(groupe, lignes_table_min):
            if tabulaire:
                compteur_table += 1
                table_id = f"p{page}{prefixe_table}{compteur_table}"
                cellules = [
                    decouper_cellules(ligne)
                    for ligne in segment
                    if not _est_separateur_markdown(ligne)
                ]
                lignes_utiles = [
                    ligne for ligne in segment if not _est_separateur_markdown(ligne)
                ]
                blocs.append(
                    Bloc(
                        type_bloc=TYPE_TABLEAU,
                        lignes=lignes_utiles,
                        page=page,
                        section_title=section_courante,
                        table_id=table_id,
                        cellules=cellules,
                    )
                )
                continue

            if _est_titre(segment):
                section_courante = _nettoyer_titre(segment[0])
                blocs.append(
                    Bloc(
                        type_bloc=TYPE_TITRE,
                        lignes=[section_courante],
                        page=page,
                        section_title=section_courante,
                    )
                )
                continue

            type_bloc = TYPE_LISTE if _est_liste(segment) else TYPE_PARAGRAPHE
            blocs.append(
                Bloc(
                    type_bloc=type_bloc,
                    lignes=segment,
                    page=page,
                    section_title=section_courante,
                )
            )

    if not blocs and texte.strip():
        # Aucune structure détectable : fallback texte brut.
        blocs.append(
            Bloc(type_bloc=TYPE_TEXTE, lignes=texte.strip().splitlines(), page=page)
        )
    return blocs


def _groupes_de_lignes(texte: str) -> Iterable[list[str]]:
    """Découpe le texte en groupes séparés par une ou plusieurs lignes vides."""
    courant: list[str] = []
    for ligne in texte.splitlines():
        if ligne.strip():
            courant.append(ligne.rstrip())
        elif courant:
            yield courant
            courant = []
    if courant:
        yield courant


def _separer_segments_tabulaires(
    lignes: list[str],
    lignes_table_min: int,
) -> Iterable[tuple[list[str], bool]]:
    """
    Sépare un groupe en segments homogènes : suites tabulaires d'un côté,
    texte de l'autre. Une suite trop courte reste du texte, ce qui évite de
    transformer une phrase contenant un « | » en tableau.
    """
    segment: list[str] = []
    tabulaire_courant = False

    for ligne in lignes:
        tabulaire = est_ligne_tabulaire(ligne)
        if segment and tabulaire != tabulaire_courant:
            yield from _emettre_segment(segment, tabulaire_courant, lignes_table_min)
            segment = []
        tabulaire_courant = tabulaire
        segment.append(ligne)

    if segment:
        yield from _emettre_segment(segment, tabulaire_courant, lignes_table_min)


def _emettre_segment(
    segment: list[str],
    tabulaire: bool,
    lignes_table_min: int,
) -> Iterable[tuple[list[str], bool]]:
    if tabulaire and len(segment) < lignes_table_min:
        yield segment, False
    else:
        yield segment, tabulaire


# ===========================================================================
# Découpage par type de bloc
# ===========================================================================


def decouper_tableau(
    bloc: Bloc,
    *,
    lignes_par_chunk: int,
    recouvrement_lignes: int,
    conserver_entete: bool,
) -> list[str]:
    """
    Découpe un tableau par lignes entières, jamais au milieu d'une ligne.

    L'en-tête est répété en tête de chaque sous-chunk : sans cela, un chunk
    contenant « Level1 | Level2 » ne permet plus de savoir à quelle année
    chaque valeur correspond. Le recouvrement est exprimé en nombre de
    lignes, pas en caractères, pour la même raison.
    """
    lignes = [ligne for ligne in bloc.lignes if ligne.strip()]
    if not lignes:
        return []

    contexte = [bloc.section_title] if bloc.section_title else []

    entete: list[str] = []
    corps = lignes
    if conserver_entete and len(lignes) > 1:
        entete = [lignes[0]]
        corps = lignes[1:]

    if not corps:
        return ["\n".join([*contexte, *lignes]).strip()]

    pas = max(1, lignes_par_chunk - max(0, recouvrement_lignes))
    morceaux: list[str] = []
    debut = 0
    while debut < len(corps):
        tranche = corps[debut : debut + lignes_par_chunk]
        morceaux.append("\n".join([*contexte, *entete, *tranche]).strip())
        if debut + lignes_par_chunk >= len(corps):
            break
        debut += pas
    return morceaux


def decouper_narratif(
    bloc: Bloc,
    *,
    taille_chunk: int,
    splitter: SplitterTexte | None,
) -> list[str]:
    """
    Découpe un bloc narratif.

    Un bloc qui tient dans la taille cible n'est pas découpé : couper un
    paragraphe court n'apporte rien et dilue son embedding.
    """
    texte = bloc.texte
    if not texte:
        return []

    prefixe = ""
    if bloc.section_title and bloc.type_bloc != TYPE_TITRE:
        # Le titre de section est répété pour que le chunk reste
        # compréhensible seul, comme l'en-tête d'un tableau.
        if not texte.startswith(bloc.section_title):
            prefixe = f"{bloc.section_title}\n"

    if len(prefixe) + len(texte) <= taille_chunk or splitter is None:
        return [f"{prefixe}{texte}".strip()]

    return [
        f"{prefixe}{morceau.strip()}".strip()
        for morceau in splitter.split_text(texte)
        if morceau.strip()
    ]


# ===========================================================================
# Assemblage parent-child
# ===========================================================================


@dataclass
class _Parent:
    """Groupe logique auquel appartiennent plusieurs chunks enfants."""

    identifiant: str
    taille: int = 0
    nb_enfants: int = 0


def decouper_blocs(
    blocs: Sequence[Bloc],
    *,
    taille_chunk: int,
    taille_parent: int,
    parent_child_actif: bool,
    lignes_par_chunk: int,
    recouvrement_lignes: int,
    conserver_entete: bool,
    tables_actives: bool,
    splitter: SplitterTexte | None,
    compteur_depart: int = 0,
) -> list[Chunk]:
    """
    Transforme une suite de blocs en chunks, en attribuant les parents.

    Un tableau constitue toujours un parent à lui seul : ses sous-chunks
    partagent le même `parent_id`, ce qui permet de reconstituer le tableau
    entier au moment de la génération. Le texte narratif est regroupé par
    section tant que la taille du parent le permet.
    """
    chunks: list[Chunk] = []
    compteur = compteur_depart
    parent: _Parent | None = None
    section_parent: str | None = None
    page_parent: int | None = None
    nb_parents = 0

    for bloc in blocs:
        if bloc.type_bloc == TYPE_TITRE:
            # Un titre n'est pas indexé seul : il serait trop court pour
            # produire un embedding utile. Il est répété en tête des blocs
            # de sa section, comme l'en-tête d'un tableau.
            continue

        if bloc.est_tableau and tables_actives:
            morceaux = decouper_tableau(
                bloc,
                lignes_par_chunk=lignes_par_chunk,
                recouvrement_lignes=recouvrement_lignes,
                conserver_entete=conserver_entete,
            )
            if not morceaux:
                continue
            nb_parents += 1
            identifiant = f"{bloc.table_id}" if bloc.table_id else f"p{bloc.page}b{nb_parents}"
            entete_repetee = conserver_entete and len(morceaux) > 1
            for ordre, morceau in enumerate(morceaux):
                chunks.append(
                    Chunk(
                        index=compteur,
                        texte=morceau,
                        page=bloc.page,
                        type_bloc=TYPE_TABLEAU,
                        section_title=bloc.section_title,
                        parent_id=identifiant if parent_child_actif else None,
                        table_id=bloc.table_id,
                        ordre_dans_parent=ordre if parent_child_actif else None,
                        is_table=True,
                        header_repeated=entete_repetee,
                    )
                )
                compteur += 1
            # Un tableau ferme le parent narratif en cours.
            parent = None
            continue

        morceaux = decouper_narratif(bloc, taille_chunk=taille_chunk, splitter=splitter)
        if not morceaux:
            continue

        for morceau in morceaux:
            nouvelle_section = bloc.section_title != section_parent
            trop_grand = parent is not None and parent.taille + len(morceau) > taille_parent
            changement_page = page_parent is not None and bloc.page != page_parent

            if parent is None or nouvelle_section or trop_grand or changement_page:
                nb_parents += 1
                parent = _Parent(identifiant=f"p{bloc.page}s{nb_parents}")
                section_parent = bloc.section_title
                page_parent = bloc.page

            chunks.append(
                Chunk(
                    index=compteur,
                    texte=morceau,
                    page=bloc.page,
                    type_bloc=bloc.type_bloc,
                    section_title=bloc.section_title,
                    parent_id=parent.identifiant if parent_child_actif else None,
                    ordre_dans_parent=parent.nb_enfants if parent_child_actif else None,
                    is_table=False,
                    header_repeated=False,
                )
            )
            parent.taille += len(morceau)
            parent.nb_enfants += 1
            compteur += 1

    return chunks


# ===========================================================================
# Point d'entrée
# ===========================================================================


def decouper_pages_structure(
    pages: Sequence[object],
    *,
    taille_chunk: int,
    taille_parent: int,
    parent_child_actif: bool,
    tables_actives: bool,
    lignes_par_chunk: int,
    recouvrement_lignes: int,
    conserver_entete: bool,
    lignes_table_min: int,
    splitter: SplitterTexte | None,
) -> list[Chunk]:
    """
    Découpe une liste de pages en chunks structure-aware.

    Les pages sont typées `object` volontairement : seul le contrat
    (`numero`, `texte`) est utilisé, ce qui évite une dépendance d'import
    circulaire avec `loaders`.
    """
    chunks: list[Chunk] = []
    for page in pages:
        texte = getattr(page, "texte", "") or ""
        numero = getattr(page, "numero", None)
        if not texte.strip():
            continue

        blocs = segmenter_page(
            texte,
            numero if isinstance(numero, int) else 0,
            lignes_table_min=lignes_table_min,
        )
        chunks_page = decouper_blocs(
            blocs,
            taille_chunk=taille_chunk,
            taille_parent=taille_parent,
            parent_child_actif=parent_child_actif,
            lignes_par_chunk=lignes_par_chunk,
            recouvrement_lignes=recouvrement_lignes,
            conserver_entete=conserver_entete,
            tables_actives=tables_actives,
            splitter=splitter,
            compteur_depart=len(chunks),
        )

        if not chunks_page:
            # Page réduite à un titre : on l'indexe telle quelle plutôt que
            # de perdre son contenu.
            chunks_page = [
                Chunk(
                    index=len(chunks),
                    texte=texte.strip(),
                    page=numero if isinstance(numero, int) else None,
                    type_bloc=TYPE_TEXTE,
                )
            ]

        chunks.extend(chunks_page)
    return chunks


def decouper_pages_recursif(
    pages: Sequence[object],
    splitter: SplitterTexte,
) -> list[Chunk]:
    """
    Découpage historique, conservé à l'identique.

    Utilisé quand `decoupage.strategie` vaut `recursive` : le comportement
    antérieur doit rester disponible sans réindexation.
    """
    chunks: list[Chunk] = []
    compteur = 0
    for page in pages:
        texte = getattr(page, "texte", "") or ""
        if not texte.strip():
            continue
        for morceau in splitter.split_text(texte):
            if morceau.strip():
                chunks.append(
                    Chunk(
                        index=compteur,
                        texte=morceau.strip(),
                        page=getattr(page, "numero", None),
                    )
                )
                compteur += 1
    return chunks


__all__ = [
    "Chunk",
    "Bloc",
    "segmenter_page",
    "decouper_tableau",
    "decouper_narratif",
    "decouper_blocs",
    "decouper_pages_structure",
    "decouper_pages_recursif",
    "est_ligne_tabulaire",
    "decouper_cellules",
    "TYPE_TITRE",
    "TYPE_PARAGRAPHE",
    "TYPE_LISTE",
    "TYPE_TABLEAU",
    "TYPE_TEXTE",
]