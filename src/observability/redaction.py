"""
Redaction centralisée — **allow-list d'abord**, scrub de motifs ensuite.

Politique (design P2.4 validé)
------------------------------
Un événement n'est jamais sérialisé par `asdict()` : `evenement_vers_log()`
recopie **champ par champ** une liste blanche stricte. Un champ ajouté à une
dataclass d'attributs mais absent d'ici n'est **pas** journalisé.

Classification des champs journalisés :

- ``CLIENT_SAFE``      : `request_id`, `operation_id`, `outcome`, `capability`,
                         `duration_ms`, timestamps.
- ``BACKEND_AUDIT``    : `documents`, `citations`, `refusal_code`,
                         `error_category`, compteurs d'ingestion, `source`,
                         `profile`, `source_count`.
- ``SENSITIVE_BACKEND``: `error_code` (nom de type d'exception — conservé tel
                         quel), `error_message` (**scrub**), `error_stack`
                         (**scrub + troncature**).
- ``FORBIDDEN``        : secrets, clés d'API, jetons, credentials, en-têtes
                         `Authorization` / `Cookie`, `.env`, prompts,
                         chain-of-thought / `<think>`, réponses LLM brutes,
                         requêtes reformulées internes, contenu documentaire
                         complet, dumps `SessionAgent` / `EtatGraphe`, chemins
                         locaux sensibles. Aucun de ces champs n'a de chemin
                         vers la sortie : ils ne figurent pas dans l'allow-list.

`str(exc)` n'est **jamais** journalisé directement : il passe par
`scrub_texte()`. Le scrub cible des motifs explicites (Bearer, JWT, `clé=valeur`
sensible, credentials d'URL, chemins absolus) ; il n'applique **aucune** règle
du type « toute chaîne longue est un secret » — `uuid4().hex` et les
identifiants de corrélation restent intacts.
"""

from __future__ import annotations

import posixpath
import re
from typing import Any

from src.observability.events import (
    AgentExecutionAttributes,
    HttpErrorAttributes,
    IngestionAttributes,
    ObservabilityEvent,
)

#: Marqueur de substitution.
REDACTED = "[REDACTED]"
_REDACTED_JWT = "[REDACTED_JWT]"
_PATH = "[PATH]"

#: Longueur maximale d'une stack journalisée (après scrub).
STACK_MAX_CHARS = 8000

# --- Classification documentaire (référence + tests) ------------------------
CLIENT_SAFE = frozenset(
    {
        "schema_version",
        "event",
        "request_id",
        "operation_id",
        "started_at",
        "finished_at",
        "duration_ms",
        "outcome",
        "capability",
    }
)
BACKEND_AUDIT = frozenset(
    {
        "documents",
        "citations",
        "source_count",
        "refusal_code",
        "error_category",
        "source",
        "profile",
        "files_found",
        "files_processed",
        "files_failed",
        "files_skipped",
        "files_empty",
        "files_ocr",
        "files_deleted",
        "chunks_indexed",
    }
)
SENSITIVE_BACKEND = frozenset({"error_code", "error_message", "error_stack"})
FORBIDDEN = frozenset(
    {
        "authorization",
        "cookie",
        "api_key",
        "apikey",
        "token",
        "secret",
        "password",
        "credentials",
        "prompt",
        "chain_of_thought",
        "think",
        "llm_raw_response",
        "reformulated_query",
        "document_content",
        "session_dump",
        "graph_state",
        "env",
        "local_path",
    }
)

# ---------------------------------------------------------------------------
# Scrub de motifs
# ---------------------------------------------------------------------------
# Un JWT compact : trois segments base64url séparés par des points, préfixe
# `eyJ` (header `{"` encodé). Cible sûre, ne matche pas un uuid hex.
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}")

# Credentials dans une URL : `scheme://user:pass@host`.
_URL_CREDS = re.compile(r"://[^\s/:@]+:[^\s/:@]+@")

# En-têtes sensibles : `Authorization: ...`, `Cookie: ...`.
_HEADER_SECRET = re.compile(
    r"(?i)\b(authorization|proxy-authorization|cookie|set-cookie|x-api-key)\b\s*:\s*[^\r\n]+"
)

# `clé = valeur` / `clé: valeur` pour un nom de clé sensible.
_KV_SECRET = re.compile(
    r"(?i)\b("
    r"authorization|api[_-]?key|apikey|access[_-]?key|access[_-]?token|"
    r"refresh[_-]?token|secret[_-]?key|client[_-]?secret|private[_-]?key|"
    r"password|passwd|pwd|passphrase|token|secret|credential|credentials"
    r")\b\s*[:=]\s*[^\s,;)\]}\"']+"
)

# `Bearer <jeton>` (hors en-tête déjà couvert).
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{6,}")

# Chemins absolus POSIX (>= 2 segments) et Windows.
_POSIX_PATH = re.compile(r"(?<![\w./])(?:/[A-Za-z0-9_.\-]+){2,}/?")
_WIN_PATH = re.compile(r"[A-Za-z]:\\(?:[^\\\s]+\\?)+")

# Ligne de traceback Python : `File "<chemin>", line N` — on ne garde que le
# nom de fichier (aucune arborescence locale, même relative).
_TB_FILE = re.compile(r'File "([^"\r\n]+)"')


def scrub_texte(valeur: Any) -> str | None:
    """Nettoie une chaîne libre des motifs sensibles connus. `None` → `None`.
    Ne lève jamais : en dernier recours renvoie `REDACTED`."""
    if valeur is None:
        return None
    try:
        texte = valeur if isinstance(valeur, str) else str(valeur)
        texte = _JWT.sub(_REDACTED_JWT, texte)
        texte = _URL_CREDS.sub("://" + REDACTED + "@", texte)
        texte = _HEADER_SECRET.sub(lambda m: f"{m.group(1)}: {REDACTED}", texte)
        texte = _KV_SECRET.sub(lambda m: f"{m.group(1)}={REDACTED}", texte)
        texte = _BEARER.sub("Bearer " + REDACTED, texte)
        texte = _WIN_PATH.sub(_PATH, texte)
        texte = _POSIX_PATH.sub(_PATH, texte)
        return texte
    except Exception:  # noqa: BLE001 — la redaction ne casse jamais l'appelant
        return REDACTED


def _tb_file_basename(match: "re.Match[str]") -> str:
    chemin = match.group(1).replace("\\", "/")
    return f'File "{posixpath.basename(chemin) or REDACTED}"'


def scrub_stack(stack: Any) -> str | None:
    """Scrub d'une stack technique : les lignes `File "<chemin>"` sont réduites
    au nom de fichier, puis scrub de motifs et troncature à `STACK_MAX_CHARS`."""
    if not stack:
        return None
    try:
        texte = stack if isinstance(stack, str) else str(stack)
        texte = _TB_FILE.sub(_tb_file_basename, texte)
    except Exception:  # noqa: BLE001
        texte = REDACTED
    texte = scrub_texte(texte) or REDACTED
    if len(texte) > STACK_MAX_CHARS:
        texte = texte[:STACK_MAX_CHARS] + "…[tronqué]"
    return texte


# ---------------------------------------------------------------------------
# Allow-list de sérialisation
# ---------------------------------------------------------------------------
def _agent_attrs_vers_log(a: AgentExecutionAttributes) -> dict[str, Any]:
    return {
        "capability": a.capability,
        "documents": list(a.documents),
        "citations": list(a.citations),
        "source_count": a.source_count,
        "refusal_code": a.refusal_code,
        "error_category": a.error_category,
        "error_code": a.error_code,
        "error_message": scrub_texte(a.error_message),
        "error_stack": scrub_stack(a.error_stack),
    }


def _ingestion_attrs_vers_log(a: IngestionAttributes) -> dict[str, Any]:
    return {
        "source": a.source,
        "profile": a.profile,
        "files_found": a.files_found,
        "files_processed": a.files_processed,
        "files_failed": a.files_failed,
        "files_skipped": a.files_skipped,
        "files_empty": a.files_empty,
        "files_ocr": a.files_ocr,
        "files_deleted": a.files_deleted,
        "chunks_indexed": a.chunks_indexed,
        "error_category": a.error_category,
        "error_code": a.error_code,
        "error_message": scrub_texte(a.error_message),
        "error_stack": scrub_stack(a.error_stack),
    }


def _http_attrs_vers_log(a: HttpErrorAttributes) -> dict[str, Any]:
    return {
        "error_category": a.error_category,
        "error_code": a.error_code,
        "error_message": scrub_texte(a.error_message),
        "error_stack": scrub_stack(a.error_stack),
    }


_DISPATCH = (
    (AgentExecutionAttributes, _agent_attrs_vers_log),
    (IngestionAttributes, _ingestion_attrs_vers_log),
    (HttpErrorAttributes, _http_attrs_vers_log),
)


def attributs_vers_log(attributes: Any) -> dict[str, Any] | None:
    """Sérialise un bloc `attributes` typé via son allow-list dédiée. Un type
    inconnu (ou `None`) → `None` (rien ne fuit)."""
    for type_attendu, fonction in _DISPATCH:
        if isinstance(attributes, type_attendu):
            return fonction(attributes)
    return None


def evenement_vers_log(event: ObservabilityEvent) -> dict[str, Any]:
    """
    Représentation JSON-sûre d'un `ObservabilityEvent`, prête pour un sink.

    Enveloppe recopiée champ par champ (allow-list) ; `attributes` délégué à
    `attributs_vers_log`. Aucun `asdict()`, aucun passage générique.
    """
    charge: dict[str, Any] = {
        "schema_version": event.schema_version,
        "event": event.event,
        "request_id": event.request_id,
        "operation_id": event.operation_id,
        "started_at": event.started_at,
        "finished_at": event.finished_at,
        "duration_ms": event.duration_ms,
        "outcome": event.outcome,
    }
    attributs = attributs_vers_log(event.attributes)
    if attributs is not None:
        charge["attributes"] = attributs
    return charge


__all__ = [
    "REDACTED",
    "STACK_MAX_CHARS",
    "CLIENT_SAFE",
    "BACKEND_AUDIT",
    "SENSITIVE_BACKEND",
    "FORBIDDEN",
    "scrub_texte",
    "scrub_stack",
    "attributs_vers_log",
    "evenement_vers_log",
]
