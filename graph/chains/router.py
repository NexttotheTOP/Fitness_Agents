from typing import Literal # a variable can only take one of predefined values

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import os
from langchain_core.prompts import MessagesPlaceholder

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "generate"] = Field(
        ...,
        description="Given a user question choose to route it to our vectorstore or directly to generate.",
    )


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are a very good router/decider. You excel at routing a user question either to our vector database with all our fitness data/knowledge or to generate.

The vectorstore contains a LOT of data related to fitness (25 000+ docs), its science behind, nutrition, really really broad in all these fields.

You should ALWAYS output vectorstore unless the user is just greeting without actual question, or the user specifically requests for the most up to date info which won't happen a lot."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("system", "Start of the conversation history"),
        MessagesPlaceholder(variable_name="chat_history"),
    ]
)

question_router = route_prompt | structured_llm_router
