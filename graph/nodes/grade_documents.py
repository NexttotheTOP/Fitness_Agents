from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question or subqueries.
    If any document is not relevant, or if fewer than 5 relevant documents remain, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO SUBQUERIES (BATCHED)---")
    question = state["question"]
    documents = state["documents"]
    subqueries = state.get("subqueries")

    # Always pass a list of queries (subqueries if present, else [question])
    queries = subqueries if subqueries and isinstance(subqueries, list) and len(subqueries) > 0 else [question]

    # Create a copy of the full state
    new_state = state.copy()

    filtered_docs = []
    web_search = False
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": queries, "document": d.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue

    # If fewer than 5 relevant documents, trigger web search
    if len(filtered_docs) < 5:
        web_search = True

    new_state["documents"] = filtered_docs
    new_state["web_search"] = web_search
    return new_state