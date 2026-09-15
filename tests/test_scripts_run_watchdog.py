"""
Régression : `scripts/run.py::_surveiller_arret_force`.

Couvre le bug diagnostiqué le 2026-09-13 — une requête synchrone bloquante
(ex. génération LLM sans timeout) tourne dans un thread non-démon
(pool AnyIO/Starlette) ; l'arrêt gracieux d'uvicorn (`Server.shutdown`)
attend indéfiniment sa fin avant de rendre la main, ce qui laisse un
process zombie qui retient le port ET le verrou Qdrant local
(`data/vectordb/.lock`), bloquant tout relancement de `scripts/run.py`.

Charge `scripts/run.py` par chemin de fichier (plutôt qu'un import de
package) : `scripts/` n'a pas de `__init__.py` et le module a des effets de
bord d'import volontairement inertes (résolution de `sys.path`, lecture de
`Settings` via `get_settings()`) — jamais de réseau ni d'écriture disque
avant `main()`.
"""

from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path
from types import SimpleNamespace

RACINE = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location("scripts_run", RACINE / "scripts" / "run.py")
assert _SPEC is not None and _SPEC.loader is not None
scripts_run = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(scripts_run)


def test_surveiller_arret_force_ne_declenche_rien_tant_que_le_serveur_tourne(monkeypatch):
    """Cas normal : `should_exit` reste `False` -> jamais de sortie forcée."""
    appels: list[int] = []
    monkeypatch.setattr(scripts_run.os, "_exit", lambda code: appels.append(code))

    serveur = SimpleNamespace(should_exit=False)
    thread = threading.Thread(
        target=scripts_run._surveiller_arret_force,
        args=(serveur, 0.05),
        daemon=True,
    )
    thread.start()
    time.sleep(0.2)

    assert appels == []
    assert thread.is_alive()  # reste en attente, aucune sortie déclenchée


def test_surveiller_arret_force_force_la_sortie_apres_le_delai(monkeypatch, capsys):
    """
    Cas du bug : l'arrêt est demandé (`should_exit=True`, posé par uvicorn
    sur Ctrl+C) mais une requête en cours ne termine jamais avant le délai
    -> `os._exit(1)` doit être appelé, et l'avertissement doit être visible
    (flush explicite : `os._exit` ne vide jamais les tampons stdio).
    """
    appels: list[int] = []
    monkeypatch.setattr(scripts_run.os, "_exit", lambda code: appels.append(code))

    serveur = SimpleNamespace(should_exit=True)
    delai = 0.05
    thread = threading.Thread(
        target=scripts_run._surveiller_arret_force,
        args=(serveur, delai),
        daemon=True,
    )
    thread.start()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert appels == [1]
    assert "sortie forcée" in capsys.readouterr().out


def test_surveiller_arret_force_pas_de_sortie_forcee_si_larret_normal_a_deja_fini(monkeypatch):
    """
    Cas heureux (idle, ~1s en pratique) : `should_exit` passe à `True` puis
    le process se termine normalement bien avant `delai_s` -> le thread
    n'a plus de raison d'exister, mais s'il tournait encore il ne doit PAS
    forcer la sortie avant l'échéance."""
    appels: list[int] = []
    monkeypatch.setattr(scripts_run.os, "_exit", lambda code: appels.append(code))

    serveur = SimpleNamespace(should_exit=True)
    thread = threading.Thread(
        target=scripts_run._surveiller_arret_force,
        args=(serveur, 10.0),
        daemon=True,
    )
    thread.start()
    time.sleep(0.2)

    assert appels == []  # toujours dans la fenêtre de grâce de 10s
    assert thread.is_alive()
