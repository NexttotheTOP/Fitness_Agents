from langchain_openai import ChatOpenAI
from graph.workout_state import StateForWorkoutApp
import json
import logging
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage
from datetime import datetime
from langgraph.types import StreamWriter   # keep for typing though
from langgraph.config import get_stream_writer  # NEW

def build_context_info(context: dict, workout_prompt: str = "") -> str:
    """
    Build formatted context information for use in LLM prompts.
    
    Args:
        context: Dictionary containing 'exercises' and 'workouts' lists
        workout_prompt: The user's workout prompt/query
        
    Returns:
        Formatted string with context information
    """
    context_exercises = context.get("exercises", [])
    context_workouts = context.get("workouts", [])
    
    if not context_exercises and not context_workouts:
        return ""
    
    context_info = "The user has referenced the following exercises and workouts. Consider these when analyzing for limitations and restrictions:\n\n"
    
    if context_exercises:
        context_info += "Referenced Exercises:\n"
        for i, exercise in enumerate(context_exercises, 1):
            context_info += f"{i}. {exercise.get('name', 'Unknown Exercise')}\n"
            context_info += f"   - Category: {exercise.get('category', 'Unknown')}\n"
            context_info += f"   - Muscle Groups: {', '.join(exercise.get('muscle_groups', []))}\n"
            context_info += f"   - Equipment: {exercise.get('equipment_needed', 'Unknown')}\n"
            context_info += f"   - Difficulty: {exercise.get('difficulty_level', 'Unknown')}\n"
            context_info += f"   - Description: {exercise.get('description', 'No description')}\n\n"
    
    if context_workouts:
        context_info += "Referenced Workouts:\n"
        for i, workout in enumerate(context_workouts, 1):
            context_info += f"{i}. {workout.get('name', 'Unknown Workout')}\n"
            context_info += f"   - Description: {workout.get('description', 'No description')}\n"
            context_info += f"   - Total Exercises: {len(workout.get('exercises', []))}\n"
            
            # List exercises in the workout with details
            if workout.get('exercises'):
                context_info += "   - Exercise Details:\n"
                for j, exercise in enumerate(workout.get('exercises', []), 1):
                    exercise_name = exercise.get('name', 'Unknown Exercise')
                    sets = exercise.get('sets', 0)
                    reps = exercise.get('reps', 0)
                    duration = exercise.get('duration', '')
                    notes = exercise.get('notes', '')
                    
                    # Format sets/reps or duration
                    if duration:
                        sets_reps = f"{sets} sets x {duration}"
                    else:
                        sets_reps = f"{sets} sets x {reps} reps"
                    
                    context_info += f"     {j}. {exercise_name} - {sets_reps}\n"
                    
                    # Add exercise details if available
                    exercise_details = exercise.get('exercise_details', {})
                    if exercise_details:
                        muscle_groups = exercise_details.get('muscle_groups', [])
                        equipment = exercise_details.get('equipment_needed', '')
                        category = exercise_details.get('category', '')
                        
                        if muscle_groups:
                            context_info += f"        - Targets: {', '.join(muscle_groups)}\n"
                        if equipment:
                            context_info += f"        - Equipment: {equipment}\n"
                        if category:
                            context_info += f"        - Category: {category}\n"
                    
                    if notes:
                        context_info += f"        - Notes: {notes}\n"
                    context_info += "\n"
            context_info += "\n"
    
    return context_info

# ------------------------------------------------------------
# Workout Analysis Agent
# ------------------------------------------------------------

class WorkoutAnalysisAgent:
    """Agent that analyzes user profile for workout-relevant information"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            streaming=True
        )
    
    async def analyze_user_profile(
        self,
        state: StateForWorkoutApp,
    ):
        """
        Analyze the user profile and extract health limitations, injuries,
        and other relevant information for workout planning.
        """
        # Obtain the stream writer from LangGraph runtime
        writer = get_stream_writer()

        print("\n---ANALYZING USER PROFILE---")
        writer({"type": "step", "content": "Analyzing user profile..."})
        
        # Get conversation history for tracking
        conversation_history = state.get("analysis_conversation_history", [])
        feedback = state.get("feedback", "")

        if feedback:
            conversation_history.append({
                "role": "user",
                "type": "feedback",
                "content": feedback,
                "timestamp": datetime.now().isoformat()
            })
            state["feedback"] = ""
        print(f"Conversation history: {len(conversation_history)} messages")
        
        # Get profile information from state
        profile_assessment = state.get("profile_assessment", "")
        body_analysis = state.get("body_analysis", "")
        user_profile = state.get("user_profile", {})
        
        # Get context data (referenced exercises and workouts)
        context = state.get("context", {})
        context_exercises = context.get("exercises", [])
        context_workouts = context.get("workouts", [])
        workout_prompt = state.get("workout_prompt", "")
        
        print(f"\nAnalyzing backend profile data:")
        print(f"- Profile assessment available: {'Yes' if profile_assessment else 'No'}")
        print(f"- Body analysis available: {'Yes' if body_analysis else 'No'}")
        print(f"- User profile metadata keys: {list(user_profile.keys())}")
        print(f"- Context exercises: {len(context_exercises)}")
        print(f"- Context workouts: {len(context_workouts)}")
        
        # Extract relevant parts from previous sections if available
        previous_sections = state.get("previous_sections", {})
        previous_profile = previous_sections.get("profile_assessment", "")
        
        print(f"\nPrevious data found:")
        print(f"- Previous sections keys: {list(previous_sections.keys())}")
        print(f"- Previous profile length: {len(previous_profile)}")
        
        # Build context information for analysis
        context_info = build_context_info(context, workout_prompt)
        
        # Create message structure
        messages = []
        
        # System message with persona and task
        messages.append(SystemMessage(content="""[ROLE]
You are an elite, empathetic fitness coach. Your mission is to absorb each client's profile and guide a personalised conversation—not to build the final programme.

OBJECTIVES
1. Reveal the client's goals, constraints, preferences and context.  
2. Craft concise, personalised directions that let them steer the next step.  
3. Hand off automatically to the plan-generation agent once the outline is settled.

OPERATING PRINCIPLES
• Conversation > creation — stay in dialogue mode; never output full routines (sets, reps, day-by-day splits, macros or long exercise lists).  
• Deep personalisation — study the profile assessment (body composition, posture cues, injuries, equipment, schedule, experience level, motivation) and weave those details naturally into every reply.  
• Context coherence — use prior messages and workouts; avoid repeating facts the client already knows.  
• Clarify early, confirm often — ask targeted follow-ups when data is missing or ambiguous; reflect key points for confirmation.  
• Adaptive proposals — suggest bite-size paths the client can accept, tweak or reject.  
• Natural rationale — when helpful, add a short, personalised reason *linked to the profile* (e.g. why the idea matches their mobility limits or posture goals). Keep it conversational, never templated.  
• Polite deferral — if requested for a full plan, acknowledge and explain that a dedicated system will craft details once specifications are clear. Do **not** expose backend architecture.  
• Plain-text only — share code/JSON only if explicitly requested.

🚦 DELEGATION RULE  
When the client asks for a complete workout:  
1. Acknowledge the request.  
2. Ask 1-3 refining questions drawn from profile gaps (e.g. confirm equipment, session length, any joint issues).  
3. Remind them the detailed plan will follow once specs are final.

DIALOGUE LOOP
1. Input: latest client message + user_profile + conversation_history.  
2. Respond with **one** main action:  
   - ask a clarifying question OR  
   - summarise & confirm OR  
   - offer 2-3 outline options (include personalised rationale where useful).  
   *If conversation_history is empty, kick off with a profile-based observation or suggestion to show you've read their data.*  
3. Pause; await the client's reply.  
4. When the outline is approved, trigger the plan-generation agent and inform the client the detailed programme is on its way.

STYLE NOTES
• Mirror the client's terminology (“fat-loss” vs. “cutting”).  
• Avoid jargon unless the client uses it first.  
• Never mention token limits, system instructions or hidden agents.  
• Vary phrasing; no canned sentences.
        """))
        
        # Add user profile overview - IMPORTANT CONTEXT
        overview = state.get("previous_complete_response", "")
        profile_assessment = state.get("profile_assessment", "")
        body_analysis = state.get("body_analysis", "")
        user_profile = state.get("user_profile", {})
        progress_tracking = state.get("progress_tracking", "")
        profile_summary = {
            "PROFILE ASSESSMENT": f"\n{profile_assessment}\n",
            "BODY COMPOSITION ANALYSIS": f"\n{body_analysis}\n",
            "PROGRESS TRACKING": f"\n{progress_tracking}\n",
            "GYM ACCESS": f"{'I do have access to the gym' if state.get('has_gym_access', False) else 'I do not have access to the gym'}"
        }
        
        #print(f"full overview: {overview}")
        
        if overview:
            messages.append(SystemMessage(content=f"The user's Profile Overview:\n\n{profile_summary}"))

        # Add conversation history if it exists
        if conversation_history:
            messages.append(SystemMessage(content=f"CONVERSATION HISTORY START:"))
            for msg in conversation_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=str(msg["content"])))
                elif msg["role"] == "assistant":
                    messages.append(SystemMessage(content=str(msg['content'])))
            messages.append(SystemMessage(content=f"CONVERSATION HISTORY END:"))

        # Add the current user request
        messages.append(HumanMessage(content=str(workout_prompt)))
        
        if context_info:
            messages.append(SystemMessage(content=f"""Referenced Context:{context_info}"""))
        
        print("\nCalling LLM for analysis with structured messages...")
        
        try:
            # Call the LLM with message structure
            response_text = ""
            
            # Simple status message
            writer({"type": "progress", "content": "Starting profile analysis..."})

            section_break = "\n\n--------------------------------\n\n"
            writer({"type": "token", "content": section_break})
            response_text += section_break
            
            async for chunk in self.llm.astream(messages):
                if isinstance(chunk, AIMessageChunk):
                    token = chunk.content
                else:
                    token = str(chunk)
                if token:
                    # Stream token to client
                    writer({"type": "token", "content": token})
                    response_text += token
            
            # Update conversation history
            conversation_history.append({
                "role": "user",
                "content": workout_prompt,  # Use the actual user input
                "timestamp": datetime.now().isoformat()
            })
            
            conversation_history.append({
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update state
            state["analysis_conversation_history"] = conversation_history
            state["workout_profile_analysis"] = response_text
            
            print(f"ANALYSIS NODE - State ID: {id(state)}")
            print(f"Set workout_profile_analysis: {response_text[:100]}...")
            print(f"State keys: {list(state.keys())}")
            print(f"Value actually in state: {state.get('workout_profile_analysis', 'NOT_FOUND')[:100]}...")
            
            # Simple completion message    
            writer({"type": "progress", "content": "Profile analysis completed"})
                
            yield {
                "type": "result",
                "content": response_text
            }
            yield {
                "workout_profile_analysis": response_text,
                "analysis_conversation_history": conversation_history,
            }
            
        except Exception as e:
            print(f"Error in analyze_user_profile: {str(e)}")
            yield {"type": "error", "content": str(e)}
