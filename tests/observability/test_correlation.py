"""
Corrélation — validation d'`X-Request-Id`, génération, liaison / reset des
`ContextVar`, isolation sous concurrence asyncio, `portee_correlation`.
"""

from __future__ import annotations

import asyncio

import pytest

from src.observability.correlation import (
    REQUEST_ID_MAX_LEN,
    CorrelationIds,
    current_execution_id,
    current_request_id,
    delier_correlation,
    lier_correlation,
    nouvel_id,
    portee_correlation,
    request_id_valide,
)


def test_nouvel_id_est_hex_32_unique():
    ids = {nouvel_id() for _ in range(1000)}
    assert len(ids) == 1000
    assert all(len(i) == 32 and all(c in "0123456789abcdef" for c in i) for i in ids)


@pytest.mark.parametrize(
    "valeur",
    [
        "abc",
        "ABC-123_x.y",
        "a" * REQUEST_ID_MAX_LEN,
        "req-2026-09-03T10.00.00",
    ],
)
def test_request_id_valide_conserve_les_ids_conformes(valeur):
    assert request_id_valide(valeur) == valeur


@pytest.mark.parametrize(
    "valeur",
    [
        None,
        "",
        "   ",
        "a" * (REQUEST_ID_MAX_LEN + 1),
        "bad id with spaces",
        "slash/inside",
        "quote'inside",
        "unicodé",
        "semi;colon",
        "amp&ersand",
        123,
    ],
)
def test_request_id_valide_rejette_les_ids_non_conformes(valeur):
    assert request_id_valide(valeur) is None


def test_lier_puis_delier_restaure_l_absence_de_contexte():
    assert current_request_id() is None
    assert current_execution_id() is None

    tokens = lier_correlation("r-1", "e-1")
    assert current_request_id() == "r-1"
    assert current_execution_id() == "e-1"

    delier_correlation(tokens)
    assert current_request_id() is None
    assert current_execution_id() is None


def test_portee_correlation_hors_contexte_genere_et_reset():
    with portee_correlation() as ids:
        assert isinstance(ids, CorrelationIds)
        assert ids.request_id and ids.execution_id
        assert current_request_id() == ids.request_id
        assert current_execution_id() == ids.execution_id
    assert current_request_id() is None
    assert current_execution_id() is None


def test_portee_correlation_reutilise_le_contexte_lie():
    tokens = lier_correlation("r-http", "e-http")
    try:
        with portee_correlation() as ids:
            assert (ids.request_id, ids.execution_id) == ("r-http", "e-http")
        # Le contexte HTTP n'a pas été réinitialisé par la portée interne.
        assert current_request_id() == "r-http"
        assert current_execution_id() == "e-http"
    finally:
        delier_correlation(tokens)


def test_portee_correlation_reset_propre_sur_exception():
    with pytest.raises(ValueError):
        with portee_correlation():
            assert current_request_id() is not None
            raise ValueError("boom")
    assert current_request_id() is None
    assert current_execution_id() is None


def test_isolation_entre_taches_asyncio_concurrentes():
    """Chaque tâche asyncio a sa propre copie de contexte : aucun id ne fuit
    d'une tâche à l'autre, même en s'entrelaçant."""

    async def tache(nom: str, resultats: dict[str, tuple[str, str]]) -> None:
        with portee_correlation(request_id=f"req-{nom}", execution_id=f"exe-{nom}"):
            await asyncio.sleep(0)  # laisse les autres tâches s'exécuter
            rid, eid = current_request_id(), current_execution_id()
            await asyncio.sleep(0)
            # inchangé après réentrelacement
            assert current_request_id() == rid
            assert current_execution_id() == eid
            resultats[nom] = (rid, eid)

    async def scenario() -> dict[str, tuple[str, str]]:
        resultats: dict[str, tuple[str, str]] = {}
        await asyncio.gather(*(tache(str(i), resultats) for i in range(25)))
        return resultats

    resultats = asyncio.run(scenario())
    assert len(resultats) == 25
    for i in range(25):
        assert resultats[str(i)] == (f"req-{i}", f"exe-{i}")
    # Aucune fuite dans le contexte principal.
    assert current_request_id() is None
    assert current_execution_id() is None
