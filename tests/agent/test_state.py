"""
Tests de l'état de workflow de l'agent.

Aucun LLM, aucun Qdrant, aucun fichier : l'état est une structure pure.
"""

from __future__ import annotations

import pytest

from src.agent.state import (
    BudgetTentativesEpuise,
    ErreurEtatAgent,
    EtapeTrace,
    EtatAgent,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_etat_initial_coherent():
    etat = EtatAgent(
        requete_initiale="Quel est le niveau B-BBEE en 2022 ?",
        requete_courante="Quel est le niveau B-BBEE en 2022 ?",
    )

    assert etat.requete_initiale == etat.requete_courante
    assert etat.tentatives == 0
    assert not etat.a_ete_reformulee
    assert etat.peut_reessayer
    assert etat.trace == []


def test_requete_courante_absente_reprend_l_initiale():
    etat = EtatAgent(requete_initiale="Question", requete_courante="")
    assert etat.requete_courante == "Question"


def test_espaces_reduits():
    etat = EtatAgent(
        requete_initiale="  Quelle   est\n la  date ? ",
        requete_courante="",
    )
    assert etat.requete_initiale == "Quelle est la date ?"


def test_requete_vide_refusee():
    with pytest.raises(ErreurEtatAgent):
        EtatAgent(requete_initiale="   ", requete_courante="")


def test_max_tentatives_invalide_refuse():
    with pytest.raises(ErreurEtatAgent):
        EtatAgent(
            requete_initiale="Question",
            requete_courante="Question",
            max_tentatives=0,
        )


# ---------------------------------------------------------------------------
# Budget de boucle — aucune boucle infinie possible
# ---------------------------------------------------------------------------


def test_incrementer_consomme_le_budget():
    etat = EtatAgent(
        requete_initiale="Question",
        requete_courante="Question",
        max_tentatives=3,
    )

    assert etat.incrementer_tentative() == 1
    assert etat.tentatives_restantes == 2
    assert etat.incrementer_tentative() == 2
    assert etat.incrementer_tentative() == 3

    assert not etat.peut_reessayer
    assert etat.tentatives_restantes == 0


def test_depassement_du_budget_leve_une_erreur():
    etat = EtatAgent(
        requete_initiale="Question",
        requete_courante="Question",
        max_tentatives=1,
    )
    etat.incrementer_tentative()

    with pytest.raises(BudgetTentativesEpuise):
        etat.incrementer_tentative()

    # Le compteur ne dépasse jamais la borne, même après l'échec.
    assert etat.tentatives == 1


def test_boucle_typique_se_termine_toujours():
    """Simule la boucle de l'étape 3 : elle doit s'arrêter d'elle-même."""
    etat = EtatAgent(
        requete_initiale="Question",
        requete_courante="Question",
        max_tentatives=4,
    )

    passages = 0
    while etat.peut_reessayer:
        etat.incrementer_tentative()
        passages += 1
        assert passages <= 4, "La boucle aurait dû être bornée."

    assert passages == 4


# ---------------------------------------------------------------------------
# Reformulation
# ---------------------------------------------------------------------------


def test_reformulation_conserve_la_requete_initiale():
    etat = EtatAgent(
        requete_initiale="B-BBEE 2022 ?",
        requete_courante="B-BBEE 2022 ?",
    )

    etat.reformuler("Niveau B-BBEE d'Absa en 2022", motif="Preuves insuffisantes")

    assert etat.requete_initiale == "B-BBEE 2022 ?"
    assert etat.requete_courante == "Niveau B-BBEE d'Absa en 2022"
    assert etat.a_ete_reformulee
    assert "reformulation" in etat.noms_etapes()


def test_reformulation_vide_refusee():
    etat = EtatAgent(requete_initiale="Question", requete_courante="Question")

    with pytest.raises(ErreurEtatAgent):
        etat.reformuler("   ")

    assert etat.requete_courante == "Question"


# ---------------------------------------------------------------------------
# Profil de domaine — contexte métier, jamais une preuve
# ---------------------------------------------------------------------------


class _ProfilFactice:
    profile_name = "finance"
    domain = "Finance et comptabilité"


def test_sans_profil_de_domaine_l_etat_reste_utilisable():
    etat = EtatAgent(requete_initiale="Question", requete_courante="Question")

    assert not etat.a_un_profil_domaine
    assert etat.nom_profil_domaine is None

    etat.ajouter_trace("session", "Construite sans contexte métier.")
    assert "Profil métier: aucun" in etat.resume_trace()


def test_avec_profil_de_domaine():
    etat = EtatAgent(
        requete_initiale="Question",
        requete_courante="Question",
        profil_domaine=_ProfilFactice(),
    )

    assert etat.a_un_profil_domaine
    assert etat.nom_profil_domaine == "finance"


def test_serialisation_ne_contient_que_le_nom_du_profil():
    """
    Le profil ne doit jamais ressembler à une donnée de réponse : seul son
    nom est sérialisé, pas sa description ni ses mots-clés.
    """
    etat = EtatAgent(
        requete_initiale="Question",
        requete_courante="Question",
        profil_domaine=_ProfilFactice(),
    )

    donnees = etat.vers_dict()

    assert donnees["profil_domaine"] == "finance"
    assert "Finance et comptabilité" not in str(donnees)


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


def test_trace_conserve_l_ordre():
    etat = EtatAgent(requete_initiale="Question", requete_courante="Question")

    etat.ajouter_trace("session", "Construite.")
    etat.ajouter_trace("outil", "search : succès", outil="search", succes=True)
    etat.ajouter_trace("evaluation", "Preuves suffisantes.")

    assert etat.noms_etapes() == ["session", "outil", "evaluation"]
    assert etat.trace[1].donnees["outil"] == "search"


def test_etape_trace_serialisable():
    etape = EtapeTrace(nom="outil", message="search : succès", donnees={"n": 6})
    donnees = etape.vers_dict()

    assert donnees["nom"] == "outil"
    assert donnees["donnees"]["n"] == 6
    assert donnees["horodatage"]


def test_resume_trace_lisible():
    etat = EtatAgent(
        requete_initiale="Question",
        requete_courante="Question",
        profil_technique="generic",
        max_tentatives=6,
    )
    etat.ajouter_trace("session", "Construite.")

    resume = etat.resume_trace()

    assert "Question" in resume
    assert "generic" in resume
    assert "0/6" in resume