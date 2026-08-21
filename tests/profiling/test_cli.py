"""
Tests de la commande `suggest` de la CLI.

Aucun appel réseau : `suggest_domain_profile` est remplacé par un double qui
mémorise les arguments reçus.
"""

from __future__ import annotations

import pytest

from src.profiling import cli
from src.profiling.models import DomainProfile
from src.profiling.storage import load_domain_profile


class SuggestionFactice:
    """Remplace le service : renvoie un profil fixe, mémorise les appels."""

    def __init__(self, profile_name: str = "domaine_propose"):
        self.profile_name = profile_name
        self.appels: list[tuple[str, str]] = []

    def __call__(self, domain: str, output_language: str = "fr") -> DomainProfile:
        self.appels.append((domain, output_language))
        return DomainProfile(
            profile_name=self.profile_name,
            domain=domain,
            description="Description générique produite par le double de test.",
            keywords=["concept a", "concept b", "concept c"],
            output_language=output_language,
        )

    @property
    def derniere_langue(self) -> str:
        return self.appels[-1][1]


@pytest.fixture()
def cli_isolee(monkeypatch, tmp_path):
    """CLI branchée sur un faux service et sur un dossier de profils jetable."""
    faux = SuggestionFactice()
    monkeypatch.setattr(cli, "suggest_domain_profile", faux)
    monkeypatch.setattr(cli, "langue_sortie_par_defaut", lambda: "fr")

    import src.profiling.storage as storage

    monkeypatch.setattr(storage, "dossier_profils", lambda dossier=None: tmp_path)
    return faux


def test_langue_issue_de_la_configuration(cli_isolee, monkeypatch):
    monkeypatch.setattr(cli, "langue_sortie_par_defaut", lambda: "en")

    assert cli.main(["suggest", "--domain", "Finance", "--yes"]) == 0
    assert cli_isolee.derniere_langue == "en"


def test_option_language_prioritaire_sur_la_configuration(cli_isolee, monkeypatch):
    monkeypatch.setattr(cli, "langue_sortie_par_defaut", lambda: "fr")

    assert cli.main(["suggest", "--domain", "Finance", "--language", "en", "--yes"]) == 0
    assert cli_isolee.derniere_langue == "en"


def test_aucune_langue_codee_en_dur_dans_la_cli():
    """Sans --language, la CLI ne doit fournir aucune valeur par défaut propre."""
    args = cli.construire_parseur().parse_args(["suggest", "--domain", "X"])
    assert args.language is None
    assert args.name is None


def test_nom_impose_remplace_celui_du_llm(cli_isolee, tmp_path):
    code = cli.main(
        ["suggest", "--domain", "Finance", "--name", "finance_v2", "--save", "--yes"]
    )
    assert code == 0
    assert (tmp_path / "finance_v2.yaml").is_file()
    assert not (tmp_path / "domaine_propose.yaml").exists()


def test_nom_impose_normalise(cli_isolee, tmp_path):
    cli.main(["suggest", "--domain", "Finance", "--name", "Finance V2", "--save", "--yes"])
    assert (tmp_path / "finance_v2.yaml").is_file()


def test_sans_nom_impose_le_nom_du_llm_est_conserve(cli_isolee, tmp_path):
    cli.main(["suggest", "--domain", "Finance", "--save", "--yes"])
    assert (tmp_path / "domaine_propose.yaml").is_file()


@pytest.mark.parametrize("nom", ["../evasion", "sous/dossier", "..", "الأمن"])
def test_nom_impose_invalide_refuse(cli_isolee, nom, capsys):
    code = cli.main(["suggest", "--domain", "Finance", "--name", nom, "--save", "--yes"])

    assert code == 2
    assert "Entrée invalide" in capsys.readouterr().err
    assert cli_isolee.appels == []  # le LLM n'est même pas appelé


def test_domaine_arabe_avec_nom_ascii(cli_isolee, tmp_path):
    code = cli.main(
        [
            "suggest",
            "--domain",
            "الأمن السيبراني",
            "--name",
            "cybersecurite_ar",
            "--language",
            "ar",
            "--save",
            "--yes",
        ]
    )
    assert code == 0

    profil = load_domain_profile("cybersecurite_ar", dossier=tmp_path)
    assert profil.domain == "الأمن السيبراني"
    assert profil.output_language == "ar"
    assert profil.profile_name == "cybersecurite_ar"


def test_sans_save_aucun_fichier_cree(cli_isolee, tmp_path):
    assert cli.main(["suggest", "--domain", "Finance", "--yes"]) == 0
    assert list(tmp_path.iterdir()) == []
