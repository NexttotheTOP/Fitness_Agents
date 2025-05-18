from typing import Any, Dict
from urllib.parse import urlparse

from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from graph.state import GraphState
from dotenv import load_dotenv

load_dotenv()
web_search_tool = TavilySearchResults(max_results=5, search_depth="advanced", max_age_days=365)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    new_state = state.copy()
    
    # Get raw Tavily results with full metadata
    tavily_results = web_search_tool.invoke({"query": question})
    
    # Create documents with proper metadata
    web_documents = []
    for i, result in enumerate(tavily_results):
        doc = Document(
            page_content=result["content"],
            metadata={
                "source": result["url"],
                "title": result.get("title", f"Web Search Result {i+1}"),
                "source_type": "web_search",
                "search_rank": i+1,
                "domain": urlparse(result["url"]).netloc,
                "result_score": result.get("score", 1.0 - (i * 0.1))  # Estimated score if not provided
            }
        )
        web_documents.append(doc)
    
    # Add to existing documents or create new list
    documents = state.get("documents", [])
    documents.extend(web_documents)
    
    # Store raw results for frontend display
    new_state["web_search_raw_results"] = tavily_results
    new_state["documents"] = documents
    return new_state


if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None})