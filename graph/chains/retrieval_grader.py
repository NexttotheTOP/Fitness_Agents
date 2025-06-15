from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing the relevance of a retrieved document to a list of user queries.
 
If the document contains keyword(s) or semantic meaning related to ANY of the queries in the list, grade it as relevant.

Give a binary score 'yes' or 'no' to indicate whether the document is relevant to at least one of the queries.

You should almost ALWAYS output yes unless its really about another topic completely not related to anything of the body or fitness or nutrition, foods, whataver at all."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "My User queries: \n\n {question}"),
        ("system", "Retrieved document: \n\n {document}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader








