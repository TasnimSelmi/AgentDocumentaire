
from src.tools.base import ContexteOutil
from src.tools.search import definir_search


def main() -> None:
    question = "Compare Absa's B-BBEE level between 2022 and 2020."

    # 1. Nouveau contexte pour cette requête utilisateur
    contexte = ContexteOutil(
        question=question,
    )

    print("AVANT SEARCH")
    print("Sources dans le contexte :", len(contexte.sources))
    print("Résultats dans le contexte :", len(contexte.resultats))

    # 2. Construction de l'outil search
    search = definir_search()

    # 3. Exécution de search AVEC le contexte partagé
    resultat = search.executer(
        contexte=contexte,
        requete=question,
    )

    print("\nAPRÈS SEARCH")
    print("Succès :", resultat.succes)
    print("Message :", resultat.message)

    print("\nRésultat search")
    print("Sources retournées :", len(resultat.sources))

    print("\nContexte partagé")
    print("Sources stockées :", len(contexte.sources))
    print("Résultats stockés :", len(contexte.resultats))

    # 4. Vérifications
    assert contexte.resultats, (
        "ERREUR : le résultat de search n'a pas été enregistré "
        "dans ContexteOutil."
    )

    assert contexte.resultats[-1] is resultat, (
        "ERREUR : le résultat conservé dans le contexte "
        "n'est pas celui retourné par search."
    )

    if resultat.sources:
        assert contexte.sources, (
            "ERREUR : search a retourné des sources mais "
            "ContexteOutil.sources est vide."
        )

        assert len(contexte.sources) == len(resultat.sources), (
            "ERREUR : le nombre de sources stockées dans le contexte "
            "ne correspond pas au résultat de search."
        )

    print("\n--- SOURCES CONSERVÉES ---")

    for index, source in enumerate(contexte.sources, start=1):
        print(
            f"{index}. {source.localisation} "
            f"| score={source.score:.4f}"
        )

    print("\n✅ TEST RÉUSSI : search → ContexteOutil.sources fonctionne.")


if __name__ == "__main__":
    main()