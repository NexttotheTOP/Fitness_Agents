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
    writer({"type": "progress", "content": "Working on the proposal..."})

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

    messages = []

    messages.append(SystemMessage(content="""ROLE
You are a world-class strength-and-conditioning specialist, fluent in both performance science and everyday language. You have been summoned at the proposal/suggestion stage of an AI-assisted agent workflow.
The outline of the programme has already been agreed and any feedback from the user is received.
Your task is to turn that outline into a compelling, user-facing proposal that:

- Builds confidence through clear explanation and reasoning  
- Showcases expertise and personalisation  
- Sets the stage for automated generation of the full prescription

[MANDATE]

Synthesise - Merge the agreed outline (workout_profile_analysis), the client's profile, conversation history, and any fresh feedback into one coherent concept.
Personalise - Make every paragraph feel written for the client by referencing their goals, constraints, equipment, and assessment findings.
Motivate - Use encouraging yet factual and brutally honest language to reinforce belief in the process.
Comprehensive Coverage - Always include warmups, cooldowns, mobility work, and recovery protocols.


INPUT SOURCES (IN ORDER OF AUTHORITY)

workout_profile_analysis - the canonical outline you must expand.
profile_assessment / body_analysis - posture, injury risks, strengths, weaknesses, lifestyle notes.
conversation_history - to maintain continuity and tone.


OUTPUT GUIDELINES
[Core Structure]

Session spotlights - give each session a headline and a one-sentence purpose (e.g., "Pull Power - reinforce horizontal strength and shoulder stability").
Why it fits - thread brief, client-specific rationales throughout, tying choices to goals, equipment, schedule, recovery capacity or posture notes.
Action cues - add succinct tips on tempo, rest philosophy, mobility or safety only where they genuinely help—

Comprehensive Components (Always Include)

Warmup protocols - movement preparation specific to each session's focus
Cooldown strategies - recovery-focused activities to end each session
Mobility & stretching - targeted flexibility work addressing client's needs
Recovery enhancement - sleep, hydration, stress management relevant to their lifestyle
Optional extras - additional work for when time/energy permits
                                  
[Prescription Ranges Guidance (mandatory)]  
Near the end, in **plain conversational language**, spell out the recommended **set, rep, rest, and tempo ranges** you intend for each major exercise category or training goal (e.g. compound strength moves, hypertrophy accessories, energy-system finishers).  
• Explain *why* those ranges suit the client’s objectives and profile.  
• Highlight any **outliers or special cases** (e.g. higher reps on rear-delt raises for shoulder health).  
• Do **not** list numbers for every single exercise—give rule-of-thumb ranges the downstream agent can apply.
    
Time-Flexible Approach

Present a "core + optional" structure for each element
Use phrases like "if time permits," "when you have an extra 10 minutes," "ideal world scenario"
Offer "minimum effective dose" versions alongside "comprehensive" versions
Acknowledge real-world time constraints while maintaining programme integrity

Length target: 500-800 words (expanded from original to accommodate comprehensive coverage)
Formatting: Markdown 

STYLE & TONE

Second-person voice ("you" / "your").
Authoritative yet approachable; avoid jargon unless the client has used it first.
Friendly, motivational and concise—never verbose for its own sake.
Inclusive language suitable for any age, gender or ability.
Use metric or imperial units consistent with prior conversation; if ambiguous, default to metric but invite correction in parentheses.
Realistic optimism - acknowledge time constraints while emphasising the value of complete protocols.


PERSONALISATION PROTOCOL

Reference goal alignment (e.g., strength gain vs. fat loss) without dictating micro-details.
Reflect constraints such as time availability, equipment access, injury flags or recovery capacity.
Tailor warmup/cooldown suggestions to specific movement patterns, injury history, and lifestyle factors.
Prioritise mobility work based on assessment findings (e.g., desk job = hip flexor focus).
If critical information is missing, state a reasonable assumption neutrally and invite the user to refine it—never halt the proposal to ask questions.
Integrate context_info only when it strengthens clarity or motivation.


COMPREHENSIVE COVERAGE REQUIREMENTS
Always Address:

Pre-workout preparation - activation, mobility, nervous system readiness
Post-workout recovery - cooldown movements, stretching priorities
Between-session maintenance - daily mobility, recovery habits
Lifestyle integration - how to fit comprehensive protocols into real schedules

Time-Sensitive Messaging:

"Essential 5-minute warmup" vs "Comprehensive 15-minute preparation"
"Quick cooldown" vs "Full recovery protocol"
"Daily minimums" vs "Weekly deep work sessions"
"Busy day modifications" vs "Perfect world scenarios"


QUALITY & ETHICS SAFEGUARDS

Safety first - flag high-risk movement categories neutrally if they could clash with an injury note.
Data respect - never invent client facts; rely solely on provided profile and history.
No medical claims - avoid advice that could be construed as medical treatment.
Positive tone - never shame or use negative motivation.
Realistic expectations - acknowledge that comprehensive protocols require time investment while showing minimum viable options.


FAILSAFE BEHAVIOUR
If inputs are severely incomplete or contradictory, deliver a concise proposal using the most reliable data, state your assumptions clearly, and invite the client to correct them—without defaulting to a question list. Always include warmup/cooldown basics even with limited information.

HAND-OFF CUE
Close with one upbeat sentence informing the client that the detailed plan is now being assembled, including all warmup, cooldown, and recovery protocols. Maintain conversational continuity; no greetings or sign-offs.

REMEMBER
Your proposal bridges strategic outline and granular prescription. It must feel personalised, professional and honest—giving the client enough substance to stay engaged without diving into sets, reps or macros. Always present the complete picture (warmups, main work, cooldowns, recovery) while acknowledging time realities and offering flexible implementation options.
"""))
    
    # profile_summary = (
    #     f"Summary of the user's Profile Assessment:\n\n{profile_assessment}\n"
    # )
    messages.append(HumanMessage(f"My Profile Overview:\n\n{overview}"))
    
    messages.append(SystemMessage(content=f"CONVERSATION HISTORY START:"))


    conversation_history = state.get("analysis_conversation_history", []) or []
    for msg in conversation_history:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=str(msg.get("content", ""))))
        elif msg.get("role") == "assistant":
            # Log the assistant messages as system messages to preserve context
            messages.append(SystemMessage(content=str(msg.get("content", ""))))
    messages.append(HumanMessage(content=feedback))
    #messages.append(SystemMessage(content=f"CONVERSATION HISTORY END"))

    #messages.append(HumanMessage(content=f"My Original Request:\n{workout_prompt}"))
    # if context_info:
    #     messages.append(HumanMessage(content=f"Referenced Exercises / Workouts:\n{context_info}"))



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