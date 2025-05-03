from typing import Literal # a variable can only take one of predefined values

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import os

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or web search.

The vectorstore contains documents related to fitness, workout routines, nutrition, and exercise techniques from popular fitness YouTubers including:
- Jeff Nippard
- AthleanX (Jeff Cavaliere)
- Renaissance Periodization (Dr. Mike Israetel)

Use the vectorstore for ANY questions related to:
- Workout routines and programming
- Exercise techniques and form
- Muscle growth and strength training
- Fitness nutrition and diet advice
- Weight loss and body composition
- General fitness concepts and principles

Only use web-search if the question is completely unrelated to fitness or requires very recent/current information that wouldn't be in the vectorstore."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
