from langchain_core.tools import tool
from src.llm.factory import construire_llm

@tool
def chercher(requete: str, categorie: str | None = None) -> str:
    """Recherche des passages pertinents dans le corpus documentaire."""
    return "ok"

llm = construire_llm().bind_tools([chercher])
r = llm.invoke("cherche les contrats qui parlent de résiliation")
print(r.tool_calls)