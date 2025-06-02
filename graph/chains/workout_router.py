from typing import Dict, Any, Literal
from graph.workout_state import StateForWorkoutApp
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import logging

def create_conversation_router():
    """Create the LLM router for conversation decisions"""
    return ChatOpenAI(
        model="gpt-4",
        temperature=0.1,  # Low temperature for consistent decisions
        disable_streaming=True   # We just need the final decision
    )

async def should_continue_conversation(state: StateForWorkoutApp):
    """
    Generator node for routing/feedback:
    - Always yields dicts with a 'type' key.
    - Yields {"type": "await_feedback", ...} when waiting for user feedback.
    - Yields {"type": "route", "decision": ...} when a routing decision is made ("continue" or "proceed").
    The graph runner or API should interpret these yields and handle routing or pausing accordingly.
    """
    print("\n---ROUTER DECISION---")

    # If waiting for user feedback, process it
    if state.get("needs_user_input"):
        feedback = state.get("user_feedback")
        decision = state.get("router_decision")
        if feedback:
            # Optionally add feedback to conversation history
            if feedback.lower() in ["agree", "proceed", "yes"]:
                # Clear feedback state
                state["needs_user_input"] = False
                state["user_feedback"] = None
                state["router_decision"] = None
                yield {"type": "route", "decision": "proceed"}
                return
            else:
                # User wants to continue/clarify
                state["needs_user_input"] = False
                state["user_feedback"] = None
                state["router_decision"] = None
                yield {"type": "route", "decision": "continue"}
                return
        else:
            # Still waiting for feedback, emit event and pause
            yield {
                "type": "await_feedback",
                "assistant_message": state.get("last_assistant_message", ""),
                "decision": state.get("router_decision", ""),
                "thread_id": state.get("thread_id"),
            }
            return

    # Get conversation data
    conversation_history = state.get("analysis_conversation_history", [])
    workout_prompt = state.get("workout_prompt", "")

    if not conversation_history:
        print("No conversation history, proceeding directly")
        yield {"type": "route", "decision": "proceed"}
        return

    # Get the latest assistant response
    latest_assistant_response = ""
    for msg in reversed(conversation_history):
        if msg["role"] == "assistant":
            latest_assistant_response = msg["content"]
            break

    if not latest_assistant_response:
        print("No assistant response found, proceeding")
        yield {"type": "route", "decision": "proceed"}
        return

    print(f"Analyzing conversation with {len(conversation_history)} messages")
    print(f"Latest response preview: {latest_assistant_response[:200]}...")

    # Create router LLM
    router_llm = create_conversation_router()

    # Build conversation summary for the router
    conversation_summary = ""
    for i, msg in enumerate(conversation_history):
        role = msg["role"].upper()
        content = msg["content"][:300] + "..." if len(msg["content"]) > 300 else msg["content"]
        conversation_summary += f"{role}: {content}\n\n"

    # Create decision prompt
    decision_prompt = f"""
You are analyzing a fitness coaching conversation to decide the next step.

CONVERSATION SO FAR:
{conversation_summary}

DECISION CRITERIA:
- CONTINUE if the assistant is asking questions, needs clarification, or waiting for user input
- CONTINUE if there are unresolved safety concerns, unclear goals, or missing critical information
- PROCEED if the assistant has gathered enough information and seems ready to create a workout plan
- PROCEED if the user has confirmed they're ready to move forward with planning

Look at the LATEST ASSISTANT RESPONSE carefully. Does it:
- Ask specific questions? → CONTINUE
- Request more information? → CONTINUE  
- Express uncertainty about user goals? → CONTINUE
- Indicate readiness to create a plan? → PROCEED
- Summarize understanding and confirm next steps? → PROCEED

Respond with exactly one word: "continue" or "proceed"
"""

    try:
        # Get router decision
        messages = [
            SystemMessage(content="You are a decision router for fitness coaching conversations. Analyze the conversation and decide whether to continue asking questions or proceed to workout planning."),
            HumanMessage(content=decision_prompt)
        ]

        response = await router_llm.ainvoke(messages)
        decision = response.content.strip().lower()

        # Ensure we get a valid decision
        if "continue" in decision:
            decision = "continue"
        elif "proceed" in decision:
            decision = "proceed"
        else:
            print(f"Unclear router response: {decision}, defaulting to continue")
            decision = "continue"

        print(f"Router decision: {decision.upper()}")

        # Save state for feedback
        state["needs_user_input"] = True
        state["router_decision"] = decision
        state["last_assistant_message"] = latest_assistant_response
        state["user_feedback"] = None

        # Emit feedback event and pause
        yield {
            "type": "await_feedback",
            "assistant_message": latest_assistant_response,
            "decision": decision,
            "thread_id": state.get("thread_id"),
        }
        return

    except Exception as e:
        print(f"Error in router decision: {str(e)}")
        yield {"type": "route", "decision": "continue"}
        return

def route_by_workflow_type(state: StateForWorkoutApp):
    """Route based on workflow type - kept for compatibility"""
    workflow_type = state.get("workflow_type", "create")

    if workflow_type == "variation":
        return "generate_variations"
    else:
        return "create_workout"
