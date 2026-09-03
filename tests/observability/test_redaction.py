"""
Redaction — scrub de motifs sensibles, non-destruction des identifiants
légitimes, et sérialisation par allow-list de l'enveloppe.
"""

from __future__ import annotations

import json

import pytest

from src.observability.correlation import nouvel_id
from src.observability.events import (
    AgentExecutionAttributes,
    IngestionAttributes,
    ObservabilityEvent,
)
from src.observability.redaction import (
    REDACTED,
    evenement_vers_log,
    scrub_stack,
    scrub_texte,
)


@pytest.mark.parametrize(
    ("brut", "interdit"),
    [
        ("api_key=AKIA1234567890ABCDEF reste", "AKIA1234567890ABCDEF"),
        ("apikey: sk-proj-abcdef123456", "sk-proj-abcdef123456"),
        ("Authorization: Bearer abcdefghijklmnop", "abcdefghijklmnop"),
        ("token=hunter2secretvalue", "hunter2secretvalue"),
        ("password = p@ssw0rd!", "p@ssw0rd!"),
        ("client_secret:zzzTOPSECRETzzz", "zzzTOPSECRETzzz"),
        (
            "connexion https://user:motdepasse@db.interne:5432/x échouée",
            "motdepasse",
        ),
        (
            "jeton eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NX0.SflKxwRJSMeKKF2QT4",
            "SflKxwRJSMeKKF2QT4",
        ),
        ("fichier /home/jawher/corpus/secret.pdf absent", "/home/jawher/corpus"),
        ("chemin C:\\Users\\admin\\secret.txt introuvable", "admin\\secret.txt"),
        ("cookie: session=deadbeefcafe; other=1", "deadbeefcafe"),
    ],
)
def test_scrub_supprime_les_motifs_sensibles(brut, interdit):
    nettoye = scrub_texte(brut)
    assert interdit not in nettoye
    assert REDACTED in nettoye or "[PATH]" in nettoye or "[REDACTED_JWT]" in nettoye


def test_scrub_exception_inconnue_ne_laisse_rien_passer():
    exc = RuntimeError("échec appel https://svc:tok3n@api/host token=abcXYZ /var/run/x")
    nettoye = scrub_texte(str(exc))
    for interdit in ("tok3n", "abcXYZ", "/var/run/x"):
        assert interdit not in nettoye


@pytest.mark.parametrize(
    "legitime",
    [
        nouvel_id(),
        nouvel_id().upper(),
        "d41d8cd98f00b204e9800998ecf8427e",
        "req-2026-09-03-0001",
        "execution-00000000000000000000000000000000",
        "a" * 64,
    ],
)
def test_scrub_ne_casse_pas_les_identifiants_legitimes(legitime):
    """Aucune règle « chaîne longue = secret » : uuid4().hex et ids de
    corrélation traversent le scrub intacts."""
    assert scrub_texte(f"contexte {legitime} fin") == f"contexte {legitime} fin"


def test_scrub_none_et_non_str():
    assert scrub_texte(None) is None
    assert isinstance(scrub_texte(12345), str)


def test_scrub_stack_tronque_et_nettoie():
    stack = "Traceback...\n  File /home/x/app.py\n" + "A" * 20000
    nettoye = scrub_stack(stack)
    assert "/home/x/app.py" not in nettoye
    assert len(nettoye) <= 8000 + len("…[tronqué]")
    assert scrub_stack(None) is None
    assert scrub_stack("") is None


def test_evenement_vers_log_enveloppe_par_allow_list():
    event = ObservabilityEvent(
        event="agent_execution_completed",
        operation_id="op-1",
        request_id="req-1",
        started_at="2026-09-03T10:00:00+00:00",
        finished_at="2026-09-03T10:00:01+00:00",
        duration_ms=1000.0,
        outcome="success",
        attributes=AgentExecutionAttributes(
            capability="search",
            documents=("a.pdf", "b.pdf"),
            citations=("S1", "S2"),
            source_count=2,
        ),
    )
    charge = evenement_vers_log(event)
    assert set(charge) == {
        "schema_version",
        "event",
        "request_id",
        "operation_id",
        "started_at",
        "finished_at",
        "duration_ms",
        "outcome",
        "attributes",
    }
    assert charge["schema_version"] == "1.0"
    assert charge["attributes"]["documents"] == ["a.pdf", "b.pdf"]
    # JSON-sérialisable.
    json.dumps(charge)


def test_evenement_vers_log_redacte_message_et_stack():
    event = ObservabilityEvent(
        event="ingestion_failed",
        operation_id="op-1",
        request_id="req-1",
        outcome="error",
        attributes=IngestionAttributes(
            source="LocalDocumentSource",
            error_category="unexpected",
            error_code="RuntimeError",
            error_message="échec /home/jawher/db token=s3cr3t",
            error_stack="File /home/jawher/app.py line 1\nBearer eyJa.bbb.ccc",
        ),
    )
    attrs = evenement_vers_log(event)["attributes"]
    # error_code (nom de type) conservé ; message et stack scrubés.
    assert attrs["error_code"] == "RuntimeError"
    assert attrs["error_category"] == "unexpected"
    for interdit in ("/home/jawher", "s3cr3t"):
        assert interdit not in json.dumps(attrs)


def test_evenement_sans_attributs_n_a_pas_de_cle_attributes():
    event = ObservabilityEvent(
        event="agent_execution_started",
        operation_id="op",
        request_id="req",
        started_at="2026-09-03T10:00:00+00:00",
    )
    assert "attributes" not in evenement_vers_log(event)
