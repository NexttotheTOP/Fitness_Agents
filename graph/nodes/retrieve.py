from typing import Any, Dict, List

from graph.state import GraphState
from ingestion import retriever, get_retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]
    
    # Create a copy of the full state
    new_state = state.copy()
    
    # Check if retriever is available
    if retriever is None:
        print("WARNING: Vector database not initialized or empty")
        new_state["documents"] = []
        new_state["web_search"] = True
        return new_state

    documents = retriever.invoke(question)
    print(f"DEBUG: Retrieved {len(documents)} documents")
    print(f"DEBUG: Documents: {documents}")
    new_state["documents"] = documents
    return new_state