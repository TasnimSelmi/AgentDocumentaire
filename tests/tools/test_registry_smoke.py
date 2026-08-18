from src.tools.base import ContexteOutil, RegistreOutils
from src.tools.classify import definir_classify
from src.tools.extract import definir_extract
from src.tools.search import definir_search
from src.tools.summarize import definir_summarize


def main() -> None:
    contexte = ContexteOutil(
        question="test registre"
    )

    registre = RegistreOutils(
        contexte=contexte
    )

    registre.enregistrer(
        definir_search()
    )
    registre.enregistrer(
        definir_extract()
    )
    registre.enregistrer(
        definir_summarize()
    )
    registre.enregistrer(
        definir_classify()
    )

    print("Outils :", registre.noms())

    assert set(registre.noms()) == {
        "search",
        "extract",
        "summarize",
        "classify",
    }

    outils_langchain = registre.vers_langchain()

    assert len(outils_langchain) == 4

    print("✅ registre complet valide.")


if __name__ == "__main__":
    main()