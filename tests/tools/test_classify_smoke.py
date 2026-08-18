from langchain_core.messages import AIMessage

from src.tools.base import ContexteOutil, SourceOutil
from src.tools.classify import definir_classify


class FakeLLM:
    def invoke(self, messages):
        return AIMessage(
            content="""
{
  "categorie": "Rapport ESG",
  "confiance": 0.92,
  "sources": ["S1"],
  "justification": "Le document contient des indicateurs ESG, sociaux et de gouvernance."
}
""".strip()
        )

class FakeLLMInvalidCategory:
    def invoke(self, messages):
        return AIMessage(
            content="""
{
  "categorie": "Document secret",
  "confiance": 0.99,
  "sources": ["S1"],
  "justification": "Catégorie inventée."
}
""".strip()
        )
def main() -> None:
    contexte = ContexteOutil(
        question="Quel est le type de ce document ?",
        llm=FakeLLM(),
        sources=[
            SourceOutil(
                doc_id="doc-absa",
                source="absa_esg.pdf",
                nom_fichier="absa_esg.pdf",
                page=1,
                categorie="rapport",
                score=0.9,
                extrait=(
                    "Environmental, Social and Governance indicators. "
                    "The report presents sustainability and governance metrics."
                ),
            )
        ],
    )

    outil = definir_classify()

    resultat = outil.executer(
        contexte=contexte,
        categories=[
            "Rapport ESG",
            "Rapport financier",
            "Contrat",
        ],
        document="absa",
        critere="type de document",
    )

    print("\n=== CLASSIFY ===")
    print("Succès :", resultat.succes)
    print("Message :", resultat.message)
    print("Données :", resultat.donnees)
    print("Sources :", len(resultat.sources))
    print("Résultats contexte :", len(contexte.resultats))

    assert resultat.succes
    assert resultat.donnees["categorie"] == "Rapport ESG"
    assert resultat.donnees["confiance"] == 0.92
    assert resultat.donnees["citations"] == ["S1"]
    assert len(resultat.sources) == 1
    assert contexte.resultats[-1].outil == "classify"

    print("\n✅ classify fonctionne avec contexte synthétique.")

    contexte_invalid = ContexteOutil(
        question="Classifie ce document",
        llm=FakeLLMInvalidCategory(),
        sources=[
            SourceOutil(
                doc_id="doc-absa",
                source="absa_esg.pdf",
                nom_fichier="absa_esg.pdf",
                page=1,
                categorie="rapport",
                score=0.9,
                extrait="Environmental, Social and Governance indicators.",
            )
        ],
    )

    resultat_invalid = definir_classify().executer(
        contexte=contexte_invalid,
        categories=[
            "Rapport ESG",
            "Rapport financier",
            "Contrat",
        ],
        document="absa",
    )

    assert resultat_invalid.donnees["categorie"] is None
    assert resultat_invalid.donnees["confiance"] == 0.0



    contexte_multi = ContexteOutil(
        question="Classifie ce document",
        llm=FakeLLM(),
        sources=[
            SourceOutil(
                doc_id="1",
                source="absa.pdf",
                nom_fichier="absa.pdf",
                page=1,
                categorie="rapport",
                score=0.9,
                extrait="ESG indicators.",
            ),
            SourceOutil(
                doc_id="2",
                source="sasol.pdf",
                nom_fichier="sasol.pdf",
                page=1,
                categorie="rapport",
                score=0.8,
                extrait="Sustainability report.",
            ),
        ],
    )

    resultat_multi = definir_classify().executer(
        contexte=contexte_multi,
        categories=[
            "Rapport ESG",
            "Rapport financier",
            "Contrat",
        ],
    )

    assert not resultat_multi.succes

    print("✅ plusieurs documents sans cible → refus.")

    print("✅ catégorie inventée correctement rejetée.")


if __name__ == "__main__":
    main()