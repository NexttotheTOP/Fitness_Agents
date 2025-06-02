from langchain_openai import ChatOpenAI
from graph.workout_state import StateForWorkoutApp
import json
import logging
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage

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

class WorkoutAnalysisAgent:
    """Agent that analyzes user profile for workout-relevant information"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            streaming=True
        )
    
    async def analyze_user_profile(self, state: StateForWorkoutApp):
        """
        Analyze the user profile and extract health limitations, injuries,
        and other relevant information for workout planning.
        """
        print("\n---ANALYZING USER PROFILE---")
        yield {"type": "step", "content": "Analyzing user profile..."}
        
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
        messages.append(SystemMessage(content="""You are a highly skilled, friendly, and collaborative fitness coach. Your job is to help the user design the best possible workout experience for their needs, goals, and preferences. 
                                    You have access to the user's profile, goals, and any context or previous workouts they provide. Your role is to analyze this information, ask clarifying questions if needed, and propose ideas, plans, or suggestions that fit what the user is looking for—whether that's a full workout plan, a single workout, exercise variations, or just advice.
                                    Engage in a conversation with the user:
                                    - If anything is unclear or you need more information, ask the user directly.
                                    - Suggest possible approaches or options, and explain your reasoning in simple, honest language.
                                    - Wait for the user's feedback or confirmation before finalizing any plan or moving forward.
                                    - Be flexible: adapt your suggestions based on the user's responses, preferences, and feedback.
                                    - Your goal is to work together with the user to find the best solution for them, step by step.

                                    Do not assume the user always wants a full plan—sometimes they may want just a split, a single workout, a variation, or advice. Always clarify and confirm before proceeding.

                                    Output your responses in clear, conversational language, as if you are chatting directly with the user. Do not output code or JSON unless the user specifically asks for it.
        """))
        
        # # System message with all user profile information
        # profile_content = "User Profile Information:\n\n"
        
        # if user_profile:
        #     profile_content += f"User Profile Data:\n{json.dumps(user_profile, indent=2)}\n\n"
        
        # if profile_assessment:
        #     profile_content += f"Profile Assessment:\n{profile_assessment}\n\n"
        
        # if body_analysis:
        #     profile_content += f"Body Analysis:\n{body_analysis}\n\n"
        
        # if previous_profile:
        #     profile_content += f"Previous Profile Assessment:\n{previous_profile}\n\n"
        
        #     messages.append(SystemMessage(content=profile_content.strip()))
        overview = state.get("previous_complete_response", "")
        if overview:
            messages.append(SystemMessage(content=f"User Profile Overview:\n{overview}"))

        messages.append(HumanMessage(content=f"""Please analyze my fitness profile for workout planning. My request: "{workout_prompt}"."""))
        
        if context_info:
            messages.append(SystemMessage(content=f"""Referenced Context:{context_info}"""))
        
        print("\nCalling LLM for analysis with structured messages...")
        
        try:
            # Call the LLM with message structure
            response_text = ""
            
            # Simple status message
            yield {"type": "progress", "content": "Starting profile analysis..."}
            
            async for chunk in self.llm.astream(messages):
                if isinstance(chunk, AIMessageChunk):
                    token = chunk.content
                else:
                    token = str(chunk)
                if token:
                    response_text += token
                    yield {"type": "token", "content": token}
            
            # Store the full analysis
            state["workout_profile_analysis"] = response_text
            
            # Simple completion message    
            yield {"type": "progress", "content": "Profile analysis completed"}
                
            yield {"type": "result", "content": response_text}
            
        except Exception as e:
            print(f"Error in analyze_user_profile: {str(e)}")
            yield {"type": "error", "content": str(e)}
