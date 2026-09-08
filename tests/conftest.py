"""
Garde globale de test : aucune suite ne doit jamais écrire dans le vrai
`config/corpus.yaml` du dépôt.

Un test qui exerce `POST /ingestion` — même un test qui ne mentionne jamais
"corpus" — passe désormais par `enregistrer_rapport_ingestion`
(`src/rag/corpus.py`), qui persiste `derniere_ingestion` via
`mettre_a_jour_corpus` -> `ecrire_registre_corpus`. Sans cette garde, une
suite qui n'a pas explicitement câblé un registre en mémoire
(`tests/rag/conftest.cabler_registre_corpus*`) corromprait silencieusement
le fichier réel du dépôt à chaque exécution.

Reproduit `get_registre_corpus`/`ecrire_registre_corpus` en mémoire,
initialisée avec le contenu RÉEL au démarrage du test (pour que
`corpus_id="default"` se comporte identiquement à l'exécution réelle), puis
absorbe toute écriture — jamais sur disque. Un test qui câble explicitement
son propre registre (`cabler_registre_corpus`/`cabler_registre_corpus_inscriptible`)
patche par-dessus cette garde, sans conflit (monkeypatch empile et dépile
correctement).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _proteger_le_registre_corpus_reel(monkeypatch):
    import src.config as config_module

    etat = dict(config_module.get_registre_corpus())

    def _lire() -> dict:
        return dict(etat)

    _lire.cache_clear = lambda: None  # type: ignore[attr-defined]

    def _ecrire(nouveau: dict) -> None:
        etat.clear()
        etat.update(nouveau)

    monkeypatch.setattr(config_module, "get_registre_corpus", _lire)
    monkeypatch.setattr(config_module, "ecrire_registre_corpus", _ecrire)
    yield
