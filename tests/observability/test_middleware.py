"""
`CorrelationMiddleware` — middleware ASGI pur, testé en l'invoquant
directement avec des `scope` / `receive` / `send` fabriqués.

Vérifie : id entrant valide conservé / invalide remplacé, `X-Execution-Id`
toujours généré serveur, recopie dans `scope["state"]`, en-têtes de réponse,
reset des `ContextVar` dans `finally` (y compris sur exception), isolation
entre requêtes concurrentes, transparence hors HTTP.
"""

from __future__ import annotations

import asyncio

import pytest

from src.observability.correlation import current_execution_id, current_request_id
from src.observability.middleware import CorrelationMiddleware


def _scope(headers=None, type_="http"):
    return {
        "type": type_,
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or [])],
        "state": {},
    }


async def _receive():
    return {"type": "http.request", "body": b"", "more_body": False}


class _AppEnregistreuse:
    """ASGI app minimale : mémorise les ids vus (contextvars + scope state) et
    renvoie une réponse 200 vide."""

    def __init__(self, *, exception: BaseException | None = None):
        self.exception = exception
        self.vu_context: tuple[str | None, str | None] | None = None
        self.vu_state: dict | None = None

    async def __call__(self, scope, receive, send):
        self.vu_context = (current_request_id(), current_execution_id())
        self.vu_state = dict(scope.get("state") or {})
        if self.exception is not None:
            raise self.exception
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": b""})


async def _run(mw, scope):
    envois = []

    async def send(message):
        envois.append(message)

    await mw(scope, _receive, send)
    return envois


def _entetes_reponse(envois):
    for m in envois:
        if m["type"] == "http.response.start":
            return {k.decode().lower(): v.decode() for k, v in m["headers"]}
    return {}


def test_id_entrant_valide_est_conserve():
    app = _AppEnregistreuse()
    mw = CorrelationMiddleware(app)
    scope = _scope([("X-Request-Id", "client-abc.123_XYZ")])

    envois = asyncio.run(_run(mw, scope))

    entetes = _entetes_reponse(envois)
    assert entetes["x-request-id"] == "client-abc.123_XYZ"
    assert scope["state"]["request_id"] == "client-abc.123_XYZ"
    assert app.vu_context[0] == "client-abc.123_XYZ"


@pytest.mark.parametrize(
    "mauvais",
    ["with space", "a" * 129, "slash/x", "quote'", "unicodé", ""],
)
def test_id_entrant_invalide_est_remplace(mauvais):
    app = _AppEnregistreuse()
    mw = CorrelationMiddleware(app)
    scope = _scope([("X-Request-Id", mauvais)])

    envois = asyncio.run(_run(mw, scope))

    genere = _entetes_reponse(envois)["x-request-id"]
    assert genere != mauvais
    assert len(genere) == 32  # uuid4().hex


def test_execution_id_toujours_genere_serveur():
    app = _AppEnregistreuse()
    mw = CorrelationMiddleware(app)
    # même si le client tente d'imposer un X-Execution-Id
    scope = _scope([("X-Execution-Id", "client-triche")])

    envois = asyncio.run(_run(mw, scope))
    entetes = _entetes_reponse(envois)

    assert entetes["x-execution-id"] != "client-triche"
    assert len(entetes["x-execution-id"]) == 32
    assert scope["state"]["execution_id"] == entetes["x-execution-id"]


def test_ids_recopies_dans_scope_state():
    app = _AppEnregistreuse()
    mw = CorrelationMiddleware(app)
    scope = _scope()

    asyncio.run(_run(mw, scope))

    assert set(scope["state"]) >= {"request_id", "execution_id"}
    assert app.vu_state["request_id"] == scope["state"]["request_id"]
    assert app.vu_state["execution_id"] == scope["state"]["execution_id"]


def test_ids_uniques_entre_requetes_successives():
    mw = CorrelationMiddleware(_AppEnregistreuse())
    vus = set()
    for _ in range(50):
        scope = _scope()
        asyncio.run(_run(mw, scope))
        vus.add((scope["state"]["request_id"], scope["state"]["execution_id"]))
    assert len(vus) == 50


def test_reset_des_contextvar_apres_requete():
    mw = CorrelationMiddleware(_AppEnregistreuse())
    asyncio.run(_run(mw, _scope()))
    assert current_request_id() is None
    assert current_execution_id() is None


def test_middleware_n_avale_pas_l_exception_et_reset_quand_meme():
    app = _AppEnregistreuse(exception=RuntimeError("panne applicative"))
    mw = CorrelationMiddleware(app)

    with pytest.raises(RuntimeError, match="panne applicative"):
        asyncio.run(_run(mw, _scope()))

    # `finally` a bien réinitialisé le contexte malgré l'exception
    assert current_request_id() is None
    assert current_execution_id() is None


def test_requete_non_http_passe_sans_correlation():
    app = _AppEnregistreuse()
    mw = CorrelationMiddleware(app)
    scope = {"type": "lifespan"}

    asyncio.run(_run(mw, scope))

    assert app.vu_context == (None, None)


def test_isolation_entre_requetes_concurrentes():
    """Plusieurs passages de middleware entrelacés : chaque requête garde ses
    propres ids, aucune fuite via les `ContextVar`."""

    class _AppLente:
        async def __call__(self, scope, receive, send):
            await asyncio.sleep(0)
            rid = current_request_id()
            await asyncio.sleep(0)
            assert current_request_id() == rid  # stable sous entrelacement
            scope["state"]["_vu"] = (rid, current_execution_id())
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

    mw = CorrelationMiddleware(_AppLente())

    async def scenario():
        scopes = [_scope([("X-Request-Id", f"req-{i}")]) for i in range(30)]
        await asyncio.gather(*(_run(mw, s) for s in scopes))
        return scopes

    scopes = asyncio.run(scenario())
    for i, scope in enumerate(scopes):
        rid, eid = scope["state"]["_vu"]
        assert rid == f"req-{i}"
        assert scope["state"]["request_id"] == f"req-{i}"
        assert scope["state"]["execution_id"] == eid
    # unicité des execution ids générés
    assert len({s["state"]["execution_id"] for s in scopes}) == 30
    assert current_request_id() is None
