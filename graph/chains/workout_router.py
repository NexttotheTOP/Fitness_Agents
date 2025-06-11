from typing import Dict, Any, Literal
from graph.workout_state import StateForWorkoutApp
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import logging

def create_conversation_router():
    """Create the LLM router for conversation decisions"""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,  # Low temperature for consistent decisions
        disable_streaming=True   # We just need the final decision
    )

def conversation_router_node(state):
    router_llm = create_conversation_router()
    messages = []

        # Add the router decision prompt
    decision_prompt = """Read the latest **assistant** message and the user's **most recent reply/feedback**.  
Output exactly one lowercase word: **continue** or **proceed**.

- return **continue** when the assistant is still asking importantquestions, clarifying, waiting for input, or when key info / safety issues are unresolved.
- return **continue** when the user's reply introduces new uncertainties, requests more explanation, or asks a question that requires a response or different approach before proceeding.

- return **proceed** when the assistant states it has enough info to design the workout OR the user signals readiness (e.g. "yes", "sounds good", "let's start"), **or when the user provides a minor tweak, addition, or preference that does not require further clarification (e.g., "please include mobility as well", "add more core work", "make it 4 days instead of 3")**.

**If the user's feedback is clear, actionable, and does not require a follow-up question, treat it as proceed—even if it is a request for a small to medium sized change.**

Edge case: if the user generally agrees but adds small preferences, treat it as **proceed**.

Respond with nothing else.
"""

    messages.append(SystemMessage(content=decision_prompt))

    # Add conversation history if it exists
    conversation_history = state.get("analysis_conversation_history", []) or []
    if conversation_history:
        messages.append(SystemMessage(content="CONVERSATION HISTORY START:"))
        for msg in conversation_history:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=str(msg.get("content", ""))))
            elif msg.get("role") == "assistant":
                messages.append(SystemMessage(content=str(msg.get("content", ""))))
        messages.append(SystemMessage(content="CONVERSATION HISTORY END:"))

    # Add feedback as the latest user message if present
    feedback = state.get("feedback", "")
    if feedback:
        messages.append(HumanMessage(content=f"The user's feedback/answer: {feedback}"))

    # Call the LLM
    response = router_llm.invoke(messages)
    decision = response.content.strip().lower()
    if "continue" in decision:
        decision = "continue"
    elif "proceed" in decision:
        decision = "proceed"
    else:
        decision = "continue"
    return {"decision": decision}

