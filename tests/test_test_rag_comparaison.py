"""
Tests de `comparer_reponse` (test_rag.py), la comparaison canonique
réponse-vérité-terrain réutilisée par tout le harnais d'évaluation.

Régression : un attendu court ("No", "Yes"...) matchait par simple
sous-chaîne, donc à l'intérieur de n'importe quel mot qui le contient
("nommees", "know", "annotation"...), sans rapport avec la question. Sur
l'échantillon end-to-end (150 questions UDA), cela classait à tort 12
réponses comme correctes (catégorie CORRECT_SANS_EVIDENCE) : l'attendu
"No" matchait par exemple dans "entites **nom**mees" ("named entities"),
sans lien avec la question posée.
"""

from __future__ import annotations

from test_rag import comparer_reponse


def test_attendu_court_ne_matche_pas_un_mot_qui_le_contient():
    """'No' ne doit pas matcher dans 'entites nommees' (nommees contient 'no')."""
    ok, _ = comparer_reponse(
        "La question dépend des types d'entités nommées présents.", "No"
    )
    assert ok is False


def test_attendu_court_matche_comme_mot_entier():
    ok, _ = comparer_reponse("La réponse est non, ce n'est pas équilibré.", "non")
    assert ok is True


def test_attendu_matche_en_debut_de_reponse():
    ok, _ = comparer_reponse("Yes, LSTM can be bidirectional.", "Yes")
    assert ok is True


def test_attendu_matche_en_fin_de_reponse():
    ok, _ = comparer_reponse("D'après les sources, la réponse est non", "non")
    assert ok is True


def test_phrase_attendue_matche_comme_sequence_de_mots_entiers():
    ok, _ = comparer_reponse(
        "Ils comparent leur approche aux méthodes de pivoting multilingue.",
        "pivoting multilingue",
    )
    assert ok is True


def test_phrase_attendue_ne_matche_pas_si_mots_non_contigus():
    ok, _ = comparer_reponse(
        "Ils comparent leur approche aux méthodes de pivoting, puis multilingue.",
        "pivoting direct multilingue",
    )
    assert ok is False


def test_attendu_numerique_reste_gere_par_la_branche_numerique():
    """Les comparaisons numériques ('2.7') ne passent pas par le matching textuel."""
    ok, motif = comparer_reponse("Le score obtenu est de 2.7 points.", "2.7")
    assert ok is True
    assert "conforme" in motif
