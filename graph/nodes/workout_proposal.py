from typing import Dict, Any
import json
from graph.workout_state import StateForWorkoutApp
from graph.nodes.workout_analysis import build_context_info
from langchain_core.messages import SystemMessage, HumanMessage

async def propose_workout_plan(state: StateForWorkoutApp):
    yield {"type": "step", "content": "Proposing workout plan structure..."}
    
    # Simple status message
    yield {"type": "progress", "content": "Starting workout plan proposal..."}

    # Gather all relevant info from state
    user_profile = state.get("user_profile", {})
    profile_assessment = state.get("profile_assessment", "")
    body_analysis = state.get("body_analysis", "")
    workout_profile_analysis = state.get("workout_profile_analysis", "")
    workout_prompt = state.get("workout_prompt", "")
    context = state.get("context", {})
    
    # Build formatted context information
    context_info = build_context_info(context, workout_prompt)
    overview = state.get("previous_complete_response", "")

    # Build messages
    messages = []
    messages.append(SystemMessage(content="""
You are a highly skilled, friendly, and motivating fitness coach. Your job is to take the agreed workout plan or structure and turn it into a detailed, user-facing proposal.

Instructions:
- Present the plan in an engaging, clear, and motivating way, speaking directly to the user.
- Add rich descriptions for each part of the plan, including the purpose, benefits, and what to expect.
- Explain your rationale for the choices you made, and highlight how the plan fits the user's goals, needs, and preferences.
- Include safety notes, tips for success, and encouragement.
- Make the proposal exciting and easy to understand, using clear language and a positive tone.
- Output only user-facing text/markdown. Do not include any code or JSON.
- Your goal is to make the user feel confident, informed, and excited to start their fitness journey!
"""))
    
    analysis_summary = (
        f"Workout profile analysis: {workout_profile_analysis}\n"
    )
    profile_summary = (
        f"User profile: {json.dumps(user_profile)}\n"
        f"Profile assessment: {profile_assessment}\n"
        f"Body analysis: {body_analysis}\n"
    )
    messages.append(SystemMessage(content=f"Profile Summary:\n{profile_summary}"))
    messages.append(HumanMessage(content=f"User Request:\n{workout_prompt}"))
    if context_info:
        messages.append(SystemMessage(content=f"Referenced Context:\n{context_info}"))
    messages.append(SystemMessage(content=f"The Analysis to follow, made by the analysis agent:\n{analysis_summary}"))

    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        response_text = ""
        async for chunk in llm.astream(messages):
            token = getattr(chunk, "content", None) or str(chunk)
            response_text += token
            yield {"type": "token", "content": token}

        # The proposal is markdown, so just yield the full result at the end
        state["plan_proposal_markdown"] = response_text
        
        # Simple completion message
        yield {"type": "progress", "content": "Workout plan proposal completed"}
        
        yield {"type": "result", "content": response_text}

    except Exception as e:
        yield {"type": "error", "content": str(e)}