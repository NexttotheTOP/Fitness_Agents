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
    feedback = state.get("feedback", "")
    
    print(f"PROPOSAL NODE - FEEDBACK: {feedback}")
    print(f"State keys: {list(state.keys())}")
    print(f"workout_profile_analysis: {state.get('workout_profile_analysis', 'NOT_FOUND')[:100]}...")
    
    # Build formatted context information
    context_info = build_context_info(context, workout_prompt)
    overview = state.get("previous_complete_response", "")

    messages: List[Any] = []

    messages.append(SystemMessage(content="""You are an expert fitness coach and personal trainer with extensive experience creating personalized workout plans. 
You excel at designing safe, effective, and personalized workouts for your clients.
                                  
Your job is to take the agreed workout plan or structure that the analysis agent (your AI colleague) has provided and go into depth on each workout component, turning it into a highly detailed user-facing proposal.

Present the plan in an informative and clear way, speaking directly to the user.
                                  
- Add rich descriptions for each part of the plan, including the purpose, benefits, and what to expect.
- Explain your rationale for the choices you made, and highlight how the plan fits the user's goals, needs, and preferences.
- Include safety notes, tips for success, and encouragement.
- Make the proposal exciting and easy to understand, using clear language and a positive tone.
- Output only user-facing text/markdown. Do not include any code or JSON.
                                  
The user has provided feedback on the analysis and proposed plan. This feedback may be a simple button press (either "continue" or "redo") or custom written text. You should take this feedback into account when generating your response. At the beginning of your message, you are encouraged (but not required) to acknowledge or reference the user's feedback in a natural way.
                                  
You are participating in an ongoing conversation with the user. Do not greet the user again or restart the conversation. Assume continuity.
"""))
    
    profile_summary = (
        f"The user's Profile Assessment: {profile_assessment}\n"
    )
    messages.append(SystemMessage(profile_summary))
    
    messages.append(SystemMessage(content=f"CONVERSATION HISTORY START:"))


    conversation_history = state.get("analysis_conversation_history", []) or []
    for msg in conversation_history:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=str(msg.get("content", ""))))
        elif msg.get("role") == "assistant":
            # Log the assistant messages as system messages to preserve context
            messages.append(SystemMessage(content=str(msg.get("content", ""))))

    messages.append(SystemMessage(content=f"CONVERSATION HISTORY END"))

    messages.append(HumanMessage(content=f"My Original Request:\n{workout_prompt}"))
    if context_info:
        messages.append(HumanMessage(content=f"Referenced Exercises / Workouts:\n{context_info}"))

    messages.append(SystemMessage(content=f"Analysis Summary (yourprimary reference):\n{workout_profile_analysis}"))
    messages.append(SystemMessage(content=f"The user's feedback on the analysis:\n{feedback}"))


    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        response_text = ""
        feedback_block = (
            "\n\n--------------------------------\n\n"
            f"> 💬 **User Feedback:**  \n"
            f"> {feedback}\n"
            "\n\n--------------------------------\n\n"
        )
        writer({"type": "token", "content": feedback_block})
        response_text += feedback_block
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