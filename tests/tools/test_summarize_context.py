from src.profiling import load_active_domain_profile
from src.rag.ingestion import construire_llm
from src.tools.base import ContexteOutil
from src.tools.search import definir_search
from src.tools.summarize import definir_summarize


def main() -> None:
    question = (
        "What does Absa report about B-BBEE levels "
        "between 2020 and 2022?"
    )

    contexte = ContexteOutil(
        question=question,
        llm=construire_llm(),
        profil_domaine=load_active_domain_profile(),
    )

    print("\n=== ÉTAT INITIAL ===")
    print("Sources :", len(contexte.sources))
    print("Résultats :", len(contexte.resultats))

    # -------------------------------------------------------
    # 1. SEARCH
    # -------------------------------------------------------

    search = definir_search()

    resultat_search = search.executer(
        contexte=contexte,
        requete=question,
    )

    print("\n=== APRÈS SEARCH ===")
    print("Succès :", resultat_search.succes)
    print("Message :", resultat_search.message)
    print("Sources contexte :", len(contexte.sources))
    print("Résultats contexte :", len(contexte.resultats))

    if not resultat_search.succes:
        raise RuntimeError(resultat_search.message)

    if not contexte.sources:
        raise RuntimeError(
            "Aucune source disponible après search."
        )

    # -------------------------------------------------------
    # 2. SUMMARIZE
    # -------------------------------------------------------

    summarize = definir_summarize()

    print("\n>>> DÉBUT SUMMARIZE")

    resultat_resume = summarize.executer(
    contexte=contexte,
    objectif=(
        "Résumer uniquement les informations concernant "
        "les niveaux B-BBEE d'Absa entre 2020 et 2022."
    ),
    format="points_cles",
    documents=["Absa"],
)

    print(">>> FIN SUMMARIZE")

    print("\n=== APRÈS SUMMARIZE ===")
    print("Succès :", resultat_resume.succes)
    print("Message :", resultat_resume.message)

    print("\nRÉSUMÉ")
    print(resultat_resume.donnees.get("resume"))

    print("\nCITATIONS VALIDES")
    print(
        resultat_resume.donnees.get(
            "citations_valides"
        )
    )

    print("\nSOURCES UTILISÉES")
    for index, source in enumerate(
        resultat_resume.sources,
        start=1,
    ):
        print(
            f"{index}. {source.localisation} "
            f"| score={source.score:.4f}"
        )

    if resultat_resume.avertissements:
        print("\nAVERTISSEMENTS")
        for avertissement in resultat_resume.avertissements:
            print("-", avertissement)

    print("\nCONTEXTE FINAL")
    print("Sources :", len(contexte.sources))
    print("Résultats :", len(contexte.resultats))

    # -------------------------------------------------------
    # 3. Vérifications
    # -------------------------------------------------------

    assert resultat_resume.succes, (
        f"Summarize a échoué : {resultat_resume.message}"
    )

    assert len(contexte.resultats) == 2

    assert contexte.resultats[0].outil == "search"
    assert contexte.resultats[1].outil == "summarize"

    assert resultat_resume.donnees.get("resume")

    print(
        "\n✅ TEST RÉUSSI : "
        "search → ContexteOutil → summarize fonctionne."
    )


if __name__ == "__main__":
    main()