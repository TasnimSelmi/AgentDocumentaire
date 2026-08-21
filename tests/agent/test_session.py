"""
Tests de la fabrique de session agentique.

Aucun test de ce fichier n'exige un serveur Ollama ni une collection Qdrant
peuplée : le LLM est injecté et aucun outil n'est exécuté contre le corpus.
C'est précisément l'intérêt d'avoir séparé l'assemblage de l'exécution.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from src.agent.session import (
    ErreurSession,
    SessionAgent,
    construire_registre,
    construire_session,
)
from src.agent.state import EtatAgent
from src.profiling.models import DomainProfile
from src.tools.base import (
    ContexteOutil,
    DefinitionOutil,
    ResultatOutil,
    SourceOutil,
)


# ---------------------------------------------------------------------------
# Doublures
# ---------------------------------------------------------------------------


class FauxLLM:
    """LLM injecté : la session ne doit jamais joindre le réseau."""

    def __init__(self) -> None:
        self.appels = 0

    def invoke(self, messages):  # pragma: no cover - non sollicité ici
        self.appels += 1
        raise AssertionError(
            "La construction d'une session ne doit appeler aucun LLM."
        )


PROFIL_TEST = DomainProfile(
    profile_name="finance_test",
    domain="Finance et comptabilité",
    description="Domaine de test, sans lien avec un corpus.",
    keywords=["budget", "trésorerie", "audit"],
    output_language="fr",
)


class _ArgsFactices(BaseModel):
    valeur: str = Field(default="", description="Argument de test.")


def _outil_lecture() -> DefinitionOutil:
    return DefinitionOutil(
        nom="lecture_factice",
        description="Outil de lecture factice.",
        schema_arguments=_ArgsFactices,
        fonction=lambda *, contexte=None, **kw: ResultatOutil(
            outil="lecture_factice",
            succes=True,
            message="ok",
            sources=[
                SourceOutil(
                    doc_id="d1",
                    source="doc.pdf",
                    nom_fichier="doc.pdf",
                    page=1,
                    extrait="Un extrait.",
                )
            ],
        ),
        lecture_seule=True,
    )


def _outil_action() -> DefinitionOutil:
    return DefinitionOutil(
        nom="action_factice",
        description="Outil qui modifierait un système externe.",
        schema_arguments=_ArgsFactices,
        fonction=lambda *, contexte=None, **kw: ResultatOutil(
            outil="action_factice", succes=True
        ),
        lecture_seule=False,
    )


def _outil_qui_casse() -> DefinitionOutil:
    raise ValueError("Profil YAML illisible.")


# ---------------------------------------------------------------------------
# Registre
# ---------------------------------------------------------------------------


def test_registre_contient_les_quatre_outils_du_projet():
    registre = construire_registre(contexte=ContexteOutil(question="test"))

    assert registre.noms() == ["classify", "extract", "search", "summarize"]


def test_registre_convertible_en_outils_langchain():
    """Prépare directement bind_tools à l'étape suivante."""
    registre = construire_registre(contexte=ContexteOutil(question="test"))
    outils = registre.vers_langchain()

    assert len(outils) == 4
    assert {outil.name for outil in outils} == {
        "search",
        "extract",
        "summarize",
        "classify",
    }
    for outil in outils:
        assert outil.description
        assert outil.args_schema is not None


def test_outils_d_action_masques_par_defaut():
    registre = construire_registre(
        contexte=ContexteOutil(question="test"),
        fabriques=[_outil_lecture, _outil_action],
    )

    assert registre.noms() == ["lecture_factice"]
    assert "action_factice" not in registre


def test_outils_d_action_exposes_sur_autorisation_explicite():
    registre = construire_registre(
        contexte=ContexteOutil(question="test"),
        fabriques=[_outil_lecture, _outil_action],
        inclure_outils_ecriture=True,
    )

    assert registre.noms() == ["action_factice", "lecture_factice"]


def test_fabrique_en_echec_remonte_clairement():
    with pytest.raises(ErreurSession, match="Profil YAML illisible"):
        construire_registre(fabriques=[_outil_qui_casse])


def test_registre_sans_fabrique_refuse():
    with pytest.raises(ErreurSession):
        construire_registre(fabriques=[])


# ---------------------------------------------------------------------------
# Session : assemblage
# ---------------------------------------------------------------------------


def test_session_complete_sans_reseau():
    session = construire_session(
        "Quel est le niveau B-BBEE d'Absa en 2022 ?",
        llm=FauxLLM(),
        profil_domaine=PROFIL_TEST,
    )

    assert isinstance(session, SessionAgent)
    assert isinstance(session.etat, EtatAgent)
    assert session.noms_outils() == ["classify", "extract", "search", "summarize"]
    assert session.etat.requete_courante == "Quel est le niveau B-BBEE d'Absa en 2022 ?"
    assert not session.a_des_preuves
    assert session.llm.appels == 0


def test_le_contexte_est_bien_partage_avec_le_registre():
    """
    Sans ce partage, un outil écrirait ses sources dans un contexte que la
    session ne lit pas : extract ne verrait jamais le résultat de search.
    """
    session = construire_session("Question", llm=FauxLLM(), charger_profil_domaine=False)

    assert session.registre.contexte is session.contexte


def test_requete_vide_refusee():
    with pytest.raises(ErreurSession):
        construire_session("   ", llm=FauxLLM())


def test_budget_par_defaut_vient_de_la_configuration():
    from src.config import get_config_technique

    session = construire_session("Question", llm=FauxLLM(), charger_profil_domaine=False)

    assert session.etat.max_tentatives == get_config_technique().agent.max_iterations


def test_budget_surchargeable():
    session = construire_session(
        "Question",
        llm=FauxLLM(),
        charger_profil_domaine=False,
        max_tentatives=2,
    )
    assert session.etat.max_tentatives == 2


def test_deux_sessions_sont_isolees():
    a = construire_session("Question A", llm=FauxLLM(), charger_profil_domaine=False)
    b = construire_session("Question B", llm=FauxLLM(), charger_profil_domaine=False)

    a.contexte.ajouter_resultat(
        ResultatOutil(
            outil="lecture_factice",
            succes=True,
            sources=[
                SourceOutil(doc_id="d1", source="a.pdf", nom_fichier="a.pdf")
            ],
        )
    )

    assert a.nombre_preuves == 1
    assert b.nombre_preuves == 0
    assert a.contexte is not b.contexte
    assert a.etat is not b.etat


# ---------------------------------------------------------------------------
# Session : profil de domaine
# ---------------------------------------------------------------------------


def test_avec_profil_actif_le_contexte_metier_est_disponible():
    session = construire_session(
        "Question", llm=FauxLLM(), profil_domaine=PROFIL_TEST
    )

    bloc = session.bloc_domaine()

    assert bloc
    assert "Finance et comptabilité" in bloc
    assert "trésorerie" in bloc
    assert session.etat.nom_profil_domaine == "finance_test"


def test_sans_profil_actif_la_session_fonctionne_normalement():
    """Contrainte du projet : le profil ne doit jamais être bloquant."""
    session = construire_session(
        "Question", llm=FauxLLM(), charger_profil_domaine=False
    )

    assert session.etat.profil_domaine is None
    assert session.bloc_domaine() == ""
    assert session.noms_outils() == ["classify", "extract", "search", "summarize"]


def test_profil_actif_charge_depuis_la_configuration(monkeypatch):
    """Le chargement passe bien par le loader existant, non réécrit."""
    appels: list[str] = []

    def faux_chargement():
        appels.append("appelé")
        return PROFIL_TEST

    monkeypatch.setattr(
        "src.agent.session.load_active_domain_profile", faux_chargement
    )

    session = construire_session("Question", llm=FauxLLM())

    assert appels == ["appelé"]
    assert session.etat.nom_profil_domaine == "finance_test"


def test_profil_explicite_court_circuite_le_chargement(monkeypatch):
    def ne_doit_pas_etre_appele():  # pragma: no cover
        raise AssertionError("Le loader ne devait pas être sollicité.")

    monkeypatch.setattr(
        "src.agent.session.load_active_domain_profile", ne_doit_pas_etre_appele
    )

    session = construire_session(
        "Question", llm=FauxLLM(), profil_domaine=PROFIL_TEST
    )
    assert session.etat.nom_profil_domaine == "finance_test"


# ---------------------------------------------------------------------------
# Session : exécution d'outil et trace
# ---------------------------------------------------------------------------


def test_executer_outil_alimente_preuves_et_trace():
    session = construire_session(
        "Question",
        llm=FauxLLM(),
        charger_profil_domaine=False,
        fabriques=[_outil_lecture],
    )

    resultat = session.executer_outil("lecture_factice", valeur="x")

    assert resultat.succes
    assert session.a_des_preuves
    assert session.nombre_preuves == 1
    assert session.outils_utilises() == ["lecture_factice"]
    assert "outil" in session.etat.noms_etapes()
    assert session.etat.trace[-1].donnees["outil"] == "lecture_factice"


def test_outil_inconnu_ne_leve_pas_d_exception():
    """Un mauvais appel doit être lisible par l'agent, pas casser la boucle."""
    session = construire_session("Question", llm=FauxLLM(), charger_profil_domaine=False)

    resultat = session.executer_outil("outil_inexistant")

    assert not resultat.succes
    assert "Outil inconnu" in resultat.message
    assert session.etat.trace[-1].donnees["succes"] is False


def test_trace_de_construction_presente():
    session = construire_session(
        "Question", llm=FauxLLM(), profil_domaine=PROFIL_TEST
    )

    premiere = session.etat.trace[0]

    assert premiere.nom == "session"
    assert premiere.donnees["profil_domaine"] == "finance_test"
    assert premiere.donnees["outils"] == [
        "classify",
        "extract",
        "search",
        "summarize",
    ]


def test_resume_lisible():
    session = construire_session(
        "Question",
        llm=FauxLLM(),
        charger_profil_domaine=False,
        fabriques=[_outil_lecture],
    )
    session.executer_outil("lecture_factice")

    resume = session.resume()

    assert "lecture_factice" in resume
    assert "Preuves      : 1" in resume