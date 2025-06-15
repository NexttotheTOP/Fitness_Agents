from typing import Any, Dict, List

from graph.state import GraphState
from supabase_retriever import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]
    
    # Create a copy of the full state
    new_state = state.copy()
    conversation_history = state.get("conversation_history", [])
    new_state = state.copy()

    # Check if retriever is available
    if retriever is None:
        print("WARNING: Vector database not initialized or empty")
        new_state["documents"] = []
        new_state["web_search"] = True
        return new_state

    documents = retriever.invoke(question, conversation_history)
    print(f"DEBUG: Retrieved {len(documents)} documents")
    print(f"DEBUG: Documents: {documents}")
    new_state["documents"] = documents

    if hasattr(retriever, "last_subqueries"):
        new_state["subqueries"] = retriever.last_subqueries

    return new_state