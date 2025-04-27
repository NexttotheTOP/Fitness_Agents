from typing import Any, Dict, List

from graph.state import GraphState
from ingestion import retriever, get_retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]

    # Check if retriever is available
    if retriever is None:
        print("WARNING: Vector database not initialized or empty")
        return {"documents": [], "question": question, "web_search": True}

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}