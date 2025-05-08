from typing import Dict, Any, List
import json
from graph.workout_state import StateForWorkoutApp

def propose_workout_plan(state: StateForWorkoutApp) -> StateForWorkoutApp:
    # Gather all relevant info from state
    user_profile = state.get("user_profile", {})
    profile_assessment = state.get("profile_assessment", "")
    body_analysis = state.get("body_analysis", "")
    health_limitations = state.get("health_limitations", [])
    exercise_restrictions = state.get("exercise_restrictions", [])
    recommended_modifications = state.get("recommended_modifications", [])
    equipment_preferences = state.get("equipment_preferences", [])
    focus_areas = state.get("focus_areas", [])
    workout_prompt = state.get("workout_prompt", "")

    prompt = f"""
[Persona]
You are a highly skilled fitness coach and program architect. The client's profile has already been thoroughly analyzed by our dedicated profile analysis agent.
Your job is to review the analysis, the user's direct workout prompt, and all available context, and then prepare a comprehensive, highly detailed workout plan structure for the workout creation agent using all this information of the client.

[User Request]
{workout_prompt}

[Task]
- Summarize the client's profile assessment and body analysis, highlighting all relevant health considerations, goals, and physical characteristics.
- Propose a highly detailed, comprehensive workout plan structure (number and type of workouts per week, split style such as upper/lower, push/pull/legs, full body, etc.), referencing the client's goals, profile, any limitations or preferences, and the user's direct request above.
- Always propose a variety of workout structures and splits, ensuring the plan is thorough and covers all relevant aspects for the client's goals and profile.
- Provide comprehensive workout structures that thoroughly address the client's goals and needs and together cover the full body, unless the user specifically requests a focused workout.
- Include multiple workout variations and splits to ensure variety and progression.
- Vary the number of exercises per workout (not always 4), and vary the estimated durations (e.g., some 30, 45, 60, or 90 minutes).
- For each proposed structure, justify your choices with reference to the user's profile, goals, and analysis.
- Be exhaustive and technical; this output is for another agent, not for the client.

[User Profile]
{json.dumps(user_profile, indent=2)}

[Profile Assessment]
{profile_assessment}

[Body Analysis]
{body_analysis}

[Results from Profile Analysis Agent]
- Health limitations: {json.dumps(health_limitations)}
- Exercise restrictions: {json.dumps(exercise_restrictions)}
- Recommended modifications: {json.dumps(recommended_modifications)}
- Equipment preferences: {json.dumps(equipment_preferences)}
- Focus areas: {json.dumps(focus_areas)}

[Instructions]
- Be extremely comprehensive and detailed in your summary and plan.
- Always propose multiple workout structures or splits if appropriate.
- Use technical, precise, and exhaustive language suitable for another expert agent.
- Output only markdown.
"""

    # Call LLM
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    response = llm.invoke(prompt)
    plan_proposal_markdown = response.content

    # Store in state
    state["plan_proposal_markdown"] = plan_proposal_markdown
    return state