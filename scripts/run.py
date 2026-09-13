"""
Lancement unique de l'agent documentaire : API FastAPI + frontend statique
(`src/ui/`) dans UN seul process, sur UN seul port.

    python scripts/run.py

Ouvre ensuite http://127.0.0.1:8000/ (redirige vers la page de gestion des
corpus). Ctrl+C arrête proprement le serveur — aucun autre process à gérer.

Vérifie avant de démarrer : Ollama joignable, modèle configuré disponible,
dossiers runtime créés (`data/...`). N'écrit, ne réinitialise et ne
supprime jamais de collection Qdrant : c'est un lanceur, pas un outil
d'administration.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from src.config import get_settings  # noqa: E402

HOST = "127.0.0.1"
PORT = 8000


def _url_ollama(base_url: str | None) -> str:
    return (base_url or "http://localhost:11434").strip().rstrip("/")


def _verifier_ollama(base_url: str, modele: str) -> None:
    """Erreur claire et immédiate si Ollama n'est pas joignable, ou si le
    modèle configuré n'est pas disponible localement — mieux vaut échouer
    ici qu'au milieu de la première requête utilisateur."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as reponse:
            donnees = json.loads(reponse.read())
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        print(f"[ERREUR] Ollama n'est pas joignable sur {base_url} ({exc}).")
        print("         Démarrez le serveur Ollama avant de relancer (`ollama serve`).")
        raise SystemExit(1) from exc
    except json.JSONDecodeError as exc:
        print(f"[ERREUR] Réponse inattendue d'Ollama sur {base_url}.")
        raise SystemExit(1) from exc

    noms_disponibles = {m.get("name", "") for m in donnees.get("models", [])}
    prefixes_disponibles = {n.split(":")[0] for n in noms_disponibles}
    if modele not in noms_disponibles and modele.split(":")[0] not in prefixes_disponibles:
        print(f"[ERREUR] Le modèle '{modele}' n'est pas disponible sur Ollama ({base_url}).")
        print(f"         Récupérez-le avec : ollama pull {modele}")
        raise SystemExit(1)


def main() -> None:
    settings = get_settings()  # lit .env + crée les dossiers runtime (data/...)
    base_url = _url_ollama(settings.llm_base_url)
    _verifier_ollama(base_url, settings.llm_model)

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover — dépendance déclarée dans requirements.txt
        print("[ERREUR] uvicorn n'est pas installé : pip install -r requirements.txt")
        raise SystemExit(1) from exc

    print("=" * 64)
    print("Agent documentaire — démarrage")
    print(f"  LLM       : {settings.llm_provider}/{settings.llm_model} @ {base_url}")
    print(f"  Qdrant    : {settings.qdrant_mode} -> {settings.qdrant_path}")
    print(f"  Documents : {settings.documents_dir}")
    print(f"  URL       : http://{HOST}:{PORT}/")
    print("=" * 64)

    uvicorn.run("src.api:create_app", factory=True, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
