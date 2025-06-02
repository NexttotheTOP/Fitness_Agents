from typing import Dict, Any, List
import json
from graph.workout_state import StateForWorkoutApp
from graph.nodes.workout_analysis import build_context_info
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import StreamWriter  # for typing reference
from langgraph.config import get_stream_writer

async def propose_workout_plan(state: StateForWorkoutApp):
    writer = get_stream_writer()

    # Step update
    writer({"type": "step", "content": "Proposing workout plan structure..."})
    
    # Simple status message
    writer({"type": "progress", "content": "Starting workout plan proposal..."})

    # Gather all relevant info from state
    user_profile = state.get("user_profile", {})
    profile_assessment = state.get("profile_assessment", "")
    body_analysis = state.get("body_analysis", "")
    workout_profile_analysis = state.get("workout_profile_analysis", "")
    workout_prompt = state.get("workout_prompt", "")
    context = state.get("context", {})
    
    print(f"PROPOSAL NODE - State ID: {id(state)}")
    print(f"State keys: {list(state.keys())}")
    print(f"workout_profile_analysis: {state.get('workout_profile_analysis', 'NOT_FOUND')[:100]}...")
    
    # Build formatted context information
    context_info = build_context_info(context, workout_prompt)
    overview = state.get("previous_complete_response", "")

    # Build messages
    messages: List[Any] = []

    # 1. Bring over relevant conversation history so the assistant continues naturally
    conversation_history = state.get("analysis_conversation_history", []) or []
    for msg in conversation_history:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=str(msg.get("content", ""))))
        elif msg.get("role") == "assistant":
            # Log the assistant messages as system messages to preserve context
            messages.append(SystemMessage(content=str(msg.get("content", ""))))

    # 2. Main instruction prompt – emphasise continuity & analysis focus
    messages.append(SystemMessage(content="""
You are a highly skilled, supportive fitness coach. 
Your job is to take the agreed workout plan or structure that the analysis agent has provided and go into depth on each workout component, turning it into a detailed user-facing proposal.

Convert the agreed structure into a detailed workout proposal focused primarily on the insights found in the analysis that follows.
Present the plan in an engaging, clear, and motivating way, speaking directly to the user.
                                  
- Add rich descriptions for each part of the plan, including the purpose, benefits, and what to expect.
- Explain your rationale for the choices you made, and highlight how the plan fits the user's goals, needs, and preferences.
- Include safety notes, tips for success, and encouragement.
- Make the proposal exciting and easy to understand, using clear language and a positive tone.
- Output only user-facing text/markdown. Do not include any code or JSON.
                                  
You are participating in an ongoing conversation with the user. Do not greet the user again or restart the conversation. Assume continuity.
"""))

    # 3. Supply analysis first (highest priority)
    messages.append(SystemMessage(content=f"Analysis Summary (primary reference):\n{workout_profile_analysis}"))

    # 4. Include profile & other context for reference (secondary)
    profile_summary = (
        f"User profile metadata: {json.dumps(user_profile)}\n"
        f"Profile assessment: {profile_assessment}\n"
        f"Body analysis: {body_analysis}\n"
    )
    messages.append(SystemMessage(content=f"Additional Reference Data:\n{profile_summary}"))

    if context_info:
        messages.append(SystemMessage(content=f"Referenced Exercises / Workouts:\n{context_info}"))

    # 5. Original user request for completeness
    messages.append(HumanMessage(content=f"User Request (original prompt):\n{workout_prompt}"))

    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        response_text = ""
        section_break = "\n\n--------------------------------\n\n"
        writer({"type": "token", "content": section_break})
        response_text += section_break
        async for chunk in llm.astream(messages):
            token = getattr(chunk, "content", None)
            if token:
                # Stream token to the client
                writer({"type": "token", "content": token})
            response_text += token
        
        # Simple completion message
        writer({"type": "progress", "content": "Workout plan proposal completed"})
        
        print("Read workout_profile_analysis:", state.get("workout_profile_analysis"))
        
        yield {"type": "result", "content": response_text}

        yield {
            "plan_proposal_markdown": response_text,
            "analysis_conversation_history": conversation_history,
        }

    except Exception as e:
        writer({"type": "error", "content": str(e)})
        yield {"type": "error", "content": str(e)}