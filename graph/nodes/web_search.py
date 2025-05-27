from typing import Any, Dict
from urllib.parse import urlparse

from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from graph.state import GraphState
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
import os


load_dotenv()
web_search_tool = TavilySearchResults(max_results=5, search_depth="advanced", max_age_days=365)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a world-class search strategist for fitness and nutrition. "
            "Given a user question, generate **three** distinct, concise web-search queries "
            "that, together, capture different angles, synonyms, or sub-topics. "
            "Return them as a plain list with no extra text.",
        ),
        ("human", "{question}"),
    ]
)

def generate_search_queries(question: str) -> list[str]:
    response = llm(
        _QUERY_PROMPT.format_messages(question=question)
    ).content.strip()
    queries = [
        line.split(".", 1)[1].strip() if "." in line else line.strip()
        for line in response.splitlines()
        if line.strip()
    ]
    return queries[:3]

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    new_state = state.copy()

    queries = generate_search_queries(question)

    per_query_results = []
    for q in queries:
        per_query_results.extend(
            web_search_tool.invoke({"query": q})[:2]
        )

    seen = set()
    tavily_results = []
    for r in per_query_results:
        if r["url"] not in seen:
            seen.add(r["url"])
            tavily_results.append(r)
        if len(tavily_results) == 6:  # stop at 6
            break
    
    # Get raw Tavily results with full metadata
    #tavily_results = web_search_tool.invoke({"query": question})
    
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