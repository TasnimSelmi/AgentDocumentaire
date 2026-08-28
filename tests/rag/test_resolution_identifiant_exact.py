"""
Fast-path « identifiant cité verbatim » de `CatalogueDocuments.resoudre`.

Aucun Qdrant, aucun embedding, aucun reranker, aucun LLM : le catalogue est
construit en mémoire à partir de payloads synthétiques, via exactement la
même fabrique que la production (`_fiche_depuis_payload`).

Cas couverts (mission — correction 2) :
  1. nom de fichier exact avec numéro -> résolution unique ;
  2. nom de fichier proche mais différent -> pas de fast-path, fallback ;
  3. identifiant absent du catalogue -> fallback historique intact ;
  4. nom de fichier non numérique déjà fonctionnel -> pas de régression ;
  5. plusieurs documents partageant un préfixe -> l'exact reste unique.
"""

from __future__ import annotations

import pytest

from src.rag.retrieval import CatalogueDocuments, _fiche_depuis_payload


@pytest.fixture(autouse=True)
def _qdrant_path_local(tmp_path, monkeypatch):
    """
    `resoudre` -> `_perimetre` -> `champs_filtrables()` -> `get_settings()` qui
    tente de créer les dossiers du projet. Sur cette machine `.env` pointe
    QDRANT_PATH vers un chemin absolu non inscriptible : on le rabat sur un
    tmp local (aucun Qdrant n'est ouvert par ces tests).
    """
    import src.config as config

    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "vectordb"))
    config.get_settings.cache_clear()
    config.get_config_technique.cache_clear()
    yield
    config.get_settings.cache_clear()
    config.get_config_technique.cache_clear()


def _catalogue(noms_fichiers: list[str]) -> CatalogueDocuments:
    fiches = []
    for i, nom in enumerate(noms_fichiers):
        payload = {
            "doc_id": f"00000000-0000-0000-0000-{i:012d}",
            "nom_fichier": nom,
            "source": f"data/documents/{nom}",
        }
        fiche = _fiche_depuis_payload(payload)
        assert fiche is not None
        fiches.append(fiche)
    return CatalogueDocuments.construire(fiches)


# Corpus type « fichiers séquentiels » : c'est le cas qui cassait la
# résolution lexicale (seul le numéro distingue les documents).
_CORPUS_NUMERIQUE = [
    "cquae_doc_219.txt",
    "cquae_doc_156.txt",
    "cquae_doc_2920.txt",
    "cquae_doc_460.txt",
]


def test_1_nom_fichier_exact_avec_numero_resout_unique():
    cat = _catalogue(_CORPUS_NUMERIQUE)
    p = cat.resoudre("Classe le document cquae_doc_219.txt.")
    assert p.statut == "exact"
    assert p.unique is True
    assert p.origine == "identifiant_exact"
    assert len(p.valeurs_filtre) == 1
    assert cat.par_identifiant("cquae_doc_219.txt").document_id in p.valeurs_filtre


def test_2_nom_fichier_proche_mais_absent_ne_declenche_pas_le_fastpath():
    cat = _catalogue(_CORPUS_NUMERIQUE)
    # cquae_doc_218.txt n'existe pas : pas de correspondance exacte -> fallback
    # lexical (qui, lui, reste incapable de trancher -> pas 'exact').
    p = cat.resoudre("Classe le document cquae_doc_218.txt.")
    assert p.origine != "identifiant_exact"
    assert not (p.statut == "exact" and p.unique)


def test_3_identifiant_absent_conserve_le_fallback_historique():
    cat = _catalogue(_CORPUS_NUMERIQUE)
    p_sans_fastpath = cat.resoudre("Quelle est la date de la bataille de Valmy ?")
    # Aucune référence d'identifiant : le fast-path ne s'active pas du tout,
    # le résultat est celui de l'algorithme lexical historique.
    assert p_sans_fastpath.origine != "identifiant_exact"


def test_4_nom_fichier_non_numerique_pas_de_regression():
    cat = _catalogue(
        ["contrat_dupont.pdf", "contrat_martin.pdf", "facture_0042.pdf"]
    )
    p = cat.resoudre("Résume le document contrat_dupont.pdf s'il te plaît.")
    assert p.statut == "exact"
    assert p.unique is True
    assert p.origine == "identifiant_exact"
    assert cat.par_identifiant("contrat_dupont.pdf").document_id in p.valeurs_filtre


@pytest.mark.parametrize(
    "nom",
    ["facture_0042.pdf", "scan_00187.pdf", "PO_000123.pdf", "1912.01214.pdf"],
)
def test_5_prefixe_partage_l_exact_reste_unique(nom):
    corpus = [
        "facture_0042.pdf",
        "facture_0043.pdf",
        "facture_0044.pdf",
        "scan_00187.pdf",
        "scan_00188.pdf",
        "PO_000123.pdf",
        "PO_000124.pdf",
        "1912.01214.pdf",
        "1912.01215.pdf",
    ]
    cat = _catalogue(corpus)
    p = cat.resoudre(f"Peux-tu classer le document {nom} ?")
    assert p.statut == "exact"
    assert p.unique is True
    assert p.valeurs_filtre == (cat.par_identifiant(nom).document_id,)


def test_mot_ordinaire_ne_declenche_pas_par_identifiant():
    # Un mot courant ne doit jamais être testé comme identifiant (gate de forme).
    cat = _catalogue(_CORPUS_NUMERIQUE)
    p = cat.resoudre("Parle-moi de Rome et de son histoire.")
    assert p.origine != "identifiant_exact"
