from src.profiling import load_active_domain_profile
from src.rag.ingestion import construire_llm
from src.tools.base import ContexteOutil
from src.tools.extract import definir_extract
from src.tools.search import definir_search


def main() -> None:
    question = "Compare Absa's B-BBEE level between 2022 and 2020."

    # -------------------------------------------------------
    # 1. Contexte partagé unique pour toute l'exécution
    # -------------------------------------------------------

    contexte = ContexteOutil(
        question=question,
        llm=construire_llm(),
        profil_domaine=load_active_domain_profile(),
    )

    print("\n=== ÉTAT INITIAL ===")
    print("Sources :", len(contexte.sources))
    print("Résultats :", len(contexte.resultats))

    # -------------------------------------------------------
    # 2. SEARCH
    # -------------------------------------------------------

    search = definir_search()

    resultat_search = search.executer(
        contexte=contexte,
        requete=question,
    )

    print("\n=== APRÈS SEARCH ===")
    print("Succès :", resultat_search.succes)
    print("Message :", resultat_search.message)
    print("Sources search :", len(resultat_search.sources))
    print("Sources contexte :", len(contexte.sources))
    print("Résultats contexte :", len(contexte.resultats))

    if not resultat_search.succes:
        raise RuntimeError(
            f"Search a échoué : {resultat_search.message}"
        )

    if not contexte.sources:
        raise RuntimeError(
            "Search n'a placé aucune source dans le contexte."
        )

    # -------------------------------------------------------
    # 3. EXTRACT
    # -------------------------------------------------------

    extract = definir_extract()

    resultat_extract = extract.executer(
        contexte=contexte,
        champs=[
            "B-BBEE level en 2022",
            "B-BBEE level en 2020",
        ],
        instruction=(
            "Extraire séparément les niveaux correspondant aux deux années. "
            "Ne pas utiliser une valeur provenant d'une autre entreprise."
        ),
    )

    print("\n=== APRÈS EXTRACT ===")
    print("Succès :", resultat_extract.succes)
    print("Message :", resultat_extract.message)

    print("\nEXTRACTIONS")
    print(resultat_extract.donnees)

    print("\nSOURCES UTILISÉES PAR EXTRACT")
    for index, source in enumerate(
        resultat_extract.sources,
        start=1,
    ):
        print(
            f"{index}. {source.localisation} "
            f"| score={source.score:.4f}"
        )

    print("\nCONTEXTE PARTAGÉ FINAL")
    print("Sources :", len(contexte.sources))
    print("Résultats :", len(contexte.resultats))

    # -------------------------------------------------------
    # 4. Vérifications structurelles
    # -------------------------------------------------------

    assert resultat_extract.succes, (
        f"Extract a échoué : {resultat_extract.message}"
    )

    assert len(contexte.resultats) == 2, (
        "Le contexte devrait contenir exactement "
        "le résultat de search et celui de extract."
    )

    assert contexte.resultats[0].outil == "search"
    assert contexte.resultats[1].outil == "extract"

    assert "extractions" in resultat_extract.donnees

    print(
        "\n✅ TEST RÉUSSI : "
        "search → ContexteOutil → extract fonctionne."
    )


if __name__ == "__main__":
    main()