"""
Tests du découpage structure-aware.

Aucun accès à Qdrant, aucun modèle chargé : le splitter narratif est injecté,
les pages sont de simples objets porteurs de `numero` et `texte`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.rag.chunking import (
    TYPE_LISTE,
    TYPE_PARAGRAPHE,
    TYPE_TABLEAU,
    Chunk,
    decouper_blocs,
    decouper_pages_recursif,
    decouper_pages_structure,
    est_ligne_tabulaire,
    segmenter_page,
)


@dataclass
class PageFactice:
    numero: int
    texte: str


class SplitterFactice:
    """Découpe par tranches fixes, sans dépendre de LangChain."""

    def __init__(self, taille: int = 200):
        self.taille = taille

    def split_text(self, text: str) -> list[str]:
        return [text[i : i + self.taille] for i in range(0, len(text), self.taille)]


def _decouper(blocs, **surcharges):
    parametres = {
        "taille_chunk": 1800,
        "taille_parent": 5000,
        "parent_child_actif": True,
        "lignes_par_chunk": 12,
        "recouvrement_lignes": 2,
        "conserver_entete": True,
        "tables_actives": True,
        "splitter": SplitterFactice(),
    }
    parametres.update(surcharges)
    return decouper_blocs(blocs, **parametres)


def _table(nb_lignes: int, entete: str = "Indicator | 2022 | 2021 | 2020") -> str:
    lignes = [entete]
    lignes += [f"Ligne{i} | v{i}a | v{i}b | v{i}c" for i in range(1, nb_lignes + 1)]
    return "\n".join(lignes)


# ---------------------------------------------------------------------------
# 1 & 2 — texte narratif
# ---------------------------------------------------------------------------


def test_petit_paragraphe_non_decoupe():
    page = PageFactice(1, "Un paragraphe court qui tient largement dans un chunk.")
    chunks = decouper_pages_structure(
        [page],
        taille_chunk=1800,
        taille_parent=5000,
        parent_child_actif=True,
        tables_actives=True,
        lignes_par_chunk=12,
        recouvrement_lignes=2,
        conserver_entete=True,
        lignes_table_min=2,
        splitter=SplitterFactice(),
    )
    assert len(chunks) == 1
    assert chunks[0].type_bloc == TYPE_PARAGRAPHE
    assert chunks[0].texte.startswith("Un paragraphe court")


def test_long_paragraphe_decoupe():
    long_texte = "Phrase de remplissage. " * 60  # ~1380 caractères
    blocs = segmenter_page(long_texte, 1)
    chunks = _decouper(blocs, taille_chunk=300, splitter=SplitterFactice(200))

    assert len(chunks) > 1
    assert all(chunk.type_bloc == TYPE_PARAGRAPHE for chunk in chunks)
    assert all(not chunk.is_table for chunk in chunks)


def test_liste_reconnue():
    texte = "- premier élément\n- deuxième élément\n- troisième élément"
    blocs = segmenter_page(texte, 1)
    assert blocs[0].type_bloc == TYPE_LISTE


# ---------------------------------------------------------------------------
# 3 à 6 — tableaux
# ---------------------------------------------------------------------------


def test_petite_table_dans_un_seul_chunk():
    blocs = segmenter_page(_table(4), 2)
    chunks = _decouper(blocs, lignes_par_chunk=12)

    assert len(chunks) == 1
    assert chunks[0].is_table is True
    assert chunks[0].header_repeated is False  # un seul chunk : rien à répéter
    assert chunks[0].texte.count("Ligne") == 4


def test_grande_table_decoupee_par_lignes():
    blocs = segmenter_page(_table(10), 2)
    chunks = _decouper(blocs, lignes_par_chunk=4, recouvrement_lignes=1)

    assert len(chunks) > 1
    assert all(chunk.is_table for chunk in chunks)


def test_entete_repetee_dans_chaque_chunk_de_table():
    entete = "Indicator | 2022 | 2021 | 2020"
    blocs = segmenter_page(_table(10, entete), 2)
    chunks = _decouper(blocs, lignes_par_chunk=4, recouvrement_lignes=1)

    assert len(chunks) > 1
    for chunk in chunks:
        assert entete in chunk.texte
        assert chunk.header_repeated is True


def test_ligne_de_table_jamais_coupee():
    blocs = segmenter_page(_table(9), 2)
    chunks = _decouper(blocs, lignes_par_chunk=3, recouvrement_lignes=1)

    for chunk in chunks:
        for ligne in chunk.texte.splitlines():
            if ligne.startswith("Ligne"):
                # Chaque ligne conserve ses quatre cellules d'origine.
                assert ligne.count("|") == 3


def test_annee_et_valeur_dans_le_meme_chunk():
    """Le cas qui motivait le chantier : en-tête et valeur jamais séparées."""
    blocs = segmenter_page(_table(12), 2)
    chunks = _decouper(blocs, lignes_par_chunk=3, recouvrement_lignes=1)

    chunks_avec_ligne7 = [c for c in chunks if "Ligne7 |" in c.texte]
    assert chunks_avec_ligne7
    for chunk in chunks_avec_ligne7:
        assert "2022" in chunk.texte and "2021" in chunk.texte


def test_recouvrement_en_lignes():
    blocs = segmenter_page(_table(6), 2)
    chunks = _decouper(blocs, lignes_par_chunk=3, recouvrement_lignes=1)

    lignes_premier = {l for l in chunks[0].texte.splitlines() if l.startswith("Ligne")}
    lignes_second = {l for l in chunks[1].texte.splitlines() if l.startswith("Ligne")}
    assert lignes_premier & lignes_second  # au moins une ligne partagée


def test_une_seule_ligne_avec_barre_reste_du_texte():
    """Une phrase contenant « | » ne doit pas devenir un tableau."""
    blocs = segmenter_page("Le champ A | B est décrit ci-dessous.", 1)
    assert blocs[0].type_bloc != TYPE_TABLEAU


def test_separateur_markdown_ignore():
    texte = "Col A | Col B\n--- | ---\nv1 | v2\nv3 | v4"
    blocs = segmenter_page(texte, 1)
    chunks = _decouper(blocs)
    assert "---" not in chunks[0].texte


# ---------------------------------------------------------------------------
# 7 — métadonnées
# ---------------------------------------------------------------------------


def test_metadonnees_parent_et_table():
    texte = "Indicateurs de performance\n\n" + _table(10)
    blocs = segmenter_page(texte, 3)
    chunks = _decouper(blocs, lignes_par_chunk=4, recouvrement_lignes=1)

    assert all(chunk.parent_id for chunk in chunks)
    assert len({chunk.parent_id for chunk in chunks}) == 1  # un seul parent
    assert all(chunk.table_id for chunk in chunks)
    assert [chunk.ordre_dans_parent for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.section_title == "Indicateurs de performance" for chunk in chunks)


def test_titre_repete_dans_le_texte_narratif():
    texte = "Conditions générales\n\nLe présent document décrit les conditions."
    blocs = segmenter_page(texte, 1)
    chunks = _decouper(blocs)

    assert len(chunks) == 1  # le titre seul ne produit pas de chunk orphelin
    assert chunks[0].texte.startswith("Conditions générales")


def test_parent_child_desactive():
    blocs = segmenter_page(_table(10), 1)
    chunks = _decouper(blocs, parent_child_actif=False)

    assert all(chunk.parent_id is None for chunk in chunks)
    assert all(chunk.ordre_dans_parent is None for chunk in chunks)


def test_index_continus_sur_plusieurs_pages():
    pages = [PageFactice(1, _table(3)), PageFactice(2, "Un paragraphe.")]
    chunks = decouper_pages_structure(
        pages,
        taille_chunk=1800,
        taille_parent=5000,
        parent_child_actif=True,
        tables_actives=True,
        lignes_par_chunk=12,
        recouvrement_lignes=2,
        conserver_entete=True,
        lignes_table_min=2,
        splitter=SplitterFactice(),
    )
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert {c.page for c in chunks} == {1, 2}


# ---------------------------------------------------------------------------
# 10 — robustesse et compatibilité
# ---------------------------------------------------------------------------


def test_document_sans_structure_detectable():
    texte = "mot " * 500  # un seul bloc, aucune ligne vide, aucun tableau
    chunks = decouper_pages_structure(
        [PageFactice(1, texte)],
        taille_chunk=300,
        taille_parent=5000,
        parent_child_actif=True,
        tables_actives=True,
        lignes_par_chunk=12,
        recouvrement_lignes=2,
        conserver_entete=True,
        lignes_table_min=2,
        splitter=SplitterFactice(200),
    )
    assert chunks
    assert all(chunk.texte.strip() for chunk in chunks)


def test_page_vide_ignoree():
    chunks = decouper_pages_structure(
        [PageFactice(1, "   "), PageFactice(2, "Contenu réel.")],
        taille_chunk=1800,
        taille_parent=5000,
        parent_child_actif=True,
        tables_actives=True,
        lignes_par_chunk=12,
        recouvrement_lignes=2,
        conserver_entete=True,
        lignes_table_min=2,
        splitter=SplitterFactice(),
    )
    assert len(chunks) == 1
    assert chunks[0].page == 2


def test_strategie_recursive_inchangee():
    """La stratégie historique ne produit aucun champ de structure."""
    pages = [PageFactice(1, _table(5)), PageFactice(2, "Un paragraphe.")]
    chunks = decouper_pages_recursif(pages, SplitterFactice(100))

    assert chunks
    assert all(chunk.type_bloc is None for chunk in chunks)
    assert all(chunk.parent_id is None for chunk in chunks)
    assert all(chunk.is_table is False for chunk in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunk_reste_compatible_avec_les_anciens_champs():
    chunk = Chunk(index=0, texte="texte", page=1)
    assert chunk.type_bloc is None
    assert chunk.parent_id is None
    assert chunk.is_table is False


@pytest.mark.parametrize(
    ("ligne", "attendu"),
    [
        ("a | b | c", True),
        ("|a|b|", True),
        ("phrase sans separateur", False),
        ("| | |", False),
    ],
)
def test_detection_ligne_tabulaire(ligne, attendu):
    assert est_ligne_tabulaire(ligne) is attendu


def test_aucune_reference_metier_dans_le_module():
    """Généricité : aucun nom d'entreprise ni de référentiel codé en dur."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2] / "src" / "rag" / "chunking.py"
    ).read_text(encoding="utf-8")
    for interdit in ("absa", "clicks", "sasol", "esg", "b-bbee", "electricity"):
        assert interdit not in source.lower()