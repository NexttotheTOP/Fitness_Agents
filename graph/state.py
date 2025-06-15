from typing import List, TypedDict, Dict, Any, Optional


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
        conversation_history: list of message objects with role and content
        thread_id: unique identifier for the conversation thread
        user_id: unique identifier for the user
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]
    conversation_history: Optional[List[Dict[str, Any]]]
    thread_id: Optional[str]
    user_id: Optional[str]
    web_search_raw_results: Optional[List[Dict[str, Any]]] 
    subqueries: Optional[List[str]]