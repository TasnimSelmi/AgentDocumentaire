"""
Helpers de mapping (`src/api/errors.py`) testés isolément — sans HTTP.
"""

from __future__ import annotations

from src.agent.response import (
    STATUT_ERREUR,
    STATUT_REFUS,
    STATUT_SUCCES,
    AgentResponse,
)
from src.api.errors import (
    ERREUR_INTERNE_PUBLIQUE,
    corps_reponse_query,
    statut_http_pour,
)


def test_statut_succes_et_refus_sont_200():
    assert statut_http_pour(AgentResponse(status=STATUT_SUCCES, capability="search")) == 200
    assert statut_http_pour(AgentResponse(status=STATUT_REFUS, capability="extract")) == 200


def test_statut_requete_invalide_est_422():
    r = AgentResponse(
        status=STATUT_ERREUR, capability="", error={"code": "requete_invalide", "message": "…"}
    )
    assert statut_http_pour(r) == 422


def test_statut_autre_erreur_est_500():
    r = AgentResponse(
        status=STATUT_ERREUR, capability="", error={"code": "ConnectionError", "message": "…"}
    )
    assert statut_http_pour(r) == 500
    assert statut_http_pour(AgentResponse(status=STATUT_ERREUR, capability="", error=None)) == 500


def test_corps_laisse_succes_intact():
    r = AgentResponse(status=STATUT_SUCCES, capability="search", answer="ok", data={"a": 1})
    assert corps_reponse_query(r) == r.vers_dict()


def test_corps_laisse_requete_invalide_intact():
    r = AgentResponse(
        status=STATUT_ERREUR,
        capability="",
        error={"code": "requete_invalide", "message": "La requête doit être une chaîne non vide."},
    )
    assert corps_reponse_query(r) == r.vers_dict()


def test_corps_masque_l_erreur_technique():
    r = AgentResponse(
        status=STATUT_ERREUR,
        capability="",
        error={"code": "ValueError", "message": "chemin /home/x/secret introuvable"},
    )
    corps = corps_reponse_query(r)
    assert corps["error"] == ERREUR_INTERNE_PUBLIQUE
    assert corps["status"] == "error"
    # les autres champs du contrat restent présents et natifs
    assert corps["capability"] == ""
    assert corps["sources"] == []
