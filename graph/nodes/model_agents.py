from __future__ import annotations

import copy
from typing import Dict, List, Any, Tuple, Literal, Union, Optional, Callable, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, InjectedState
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.types import Command, StreamWriter
from graph.model_state import ModelState
from graph.tools import (
    MODEL_CONTROL_TOOL_FUNCTIONS,
    select_muscles_tool,
    toggle_muscle_tool,
    set_animation_frame_tool,
    toggle_animation_tool,
    set_camera_position_tool,
    set_camera_target_tool,
    reset_camera_tool,
    set_camera_view_tool
)
import json
import asyncio

__all__ = [
    "SYSTEM_PROMPT",
    "RESPONSE_PROMPT",
    "MODEL_CONTROL_TOOL_FUNCTIONS_NO_ANIMATION",
    "TOOL_MAP",
    "select_muscles_tool",
    "toggle_muscle_tool",
    "set_animation_frame_tool",
    "toggle_animation_tool",
    "set_camera_position_tool",
    "set_camera_target_tool", 
    "set_camera_view_tool",
    "reset_camera_tool",
    "model_agent",
    "MUSCLE_MAPPING_STR",
    "MUSCLE_PAIRING_RULES",
    "MUSCLE_NAMING_RULES",
]

# Add muscle mapping/grouping variables for prompt context
MUSCLE_MAPPING_STR = '''
FACE:
  - General: Orbicularis_Oculi, Orbicularis_Oris, Corrugator_Supercilii, Occipitofrontalis_Frontal, Depressor_Supercilii, Eyes, Procerus, Auricular_Cartilage, Occipitofrontalis_Occipital
  - Mouth: Depressor_Anguli_Oris, Mentalis, Depressor_Labii_Inferioris, Risorius, Zygomaticus_Major, Zygomaticus_Minor, Levator_Labii_Superioris, Buccinator
  - Nose: Depressor_Septi_Nasi, Greater_Alar_Cartilage, Lateral_Nasal_Cartilage, Nasalis_Transverese, Septal_Cartilage, Nasalis_Alar, Levator_Labii_Superioris_Alaeque_Nasi
  - Jaw: Masseter_Deep, Masseter_Superficial, Genioglossus, Temporalis, Mylohyoid, Digastric

TRUNK:
  - Chest (Left): Pectoralis_Major_01_Clavicular, Pectoralis_Major_02_Sternocostal, Pectoralis_Major_03_Abdominal, Pectoralis_Minor
  - Chest (Right): Pectoralis_Major_01_Clavicular_R, Pectoralis_Major_02_Sternocostal_R, Pectoralis_Major_03_Abdominal_R, Pectoralis_Minor_R
  - Abdomen (Left): External_Oblique, Rectus_Abdominis
  - Abdomen (Right): External_Oblique_R, Rectus_Abdominis_R
  - Back (Left): Iliocostalis_Lumborum, Latissimus_Dorsi, Longissimus_Thoracis, Rhomboideus_Major, Rhomboideus_Minor, Serratus_Posterior_Inferior, Serratus_Posterior_Superior, Spinalis_Thoracis, Splenius_Capitis, Splenius_Cervicis, Trapezius_01_Upper, Trapezius_02_Middle, Trapezius_03_Lower
  - Back (Right): Iliocostalis_Lumborum_R, Latissimus_Dorsi_R, Longissimus_Thoracis_R, Rhomboideus_Major_R, Rhomboideus_Minor_R, Serratus_Posterior_Inferior_R, Serratus_Posterior_Superior_R, Spinalis_Thoracis_R, Splenius_Capitis_R, Splenius_Cervicis_R, Trapezius_01_Upper_R, Trapezius_02_Middle_R, Trapezius_03_Lower_R
  - Neck (Left): Levator_Scapulae, Omohyoid, Scalenus_Medius, Semispinalis_Capitis_Lateral, Semispinalis_Capitis_Medial, Sternocleidomastoid, Sternohyoid
  - Neck (Right): Levator_Scapulae_R, Omohyoid_R, Scalenus_Medius_R, Semispinalis_Capitis_Lateral_R, Semispinalis_Capitis_Medial_R, Sternocleidomastoid_R, Sternohyoid_R
  - Shoulders (Left): Trapezius_01_Upper, Trapezius_02_Middle, Trapezius_03_Lower, Serratus_Anterior
  - Shoulders (Right): Trapezius_01_Upper_R, Trapezius_02_Middle_R, Trapezius_03_Lower_R, Serratus_Anterior_R

PELVIS:
  - Pelvic Floor: Ischium, Iliococygeus, Coccygeus, Anal_Sphincter_External_Superficial, Pubococcygeus, Anal_Sphincter_External, Puborectalis, Superficial_Transverse_Perineal, Deep_Transverse_Perineal
  - Hip Flexors (Left): Psoas_Major, Psoas_Minor
  - Hip Flexors (Right): Psoas_Major_R, Psoas_Minor_R

ARMS:
  - Upper Arm (Left): Triceps_Medial_Head, Triceps_Lateral_Long_Heads, Biceps_Brachii, Corabrachialis, Brachialis
  - Upper Arm (Right): Triceps_Medial_Head_R, Triceps_Lateral_Long_Heads_R, Biceps_Brachii_R, Corabrachialis_R, Brachialis_R
  - Shoulder (Left): Deltoid_Anterior, Deltoid_Middle, Deltoid_Posterior, Infraspinatus, Subscapularis, Supraspinatus, Teres_Major, Teres_Minor
  - Shoulder (Right): Deltoid_Anterior_R, Deltoid_Middle_R, Deltoid_Posterior_R, Infraspinatus_R, Subscapularis_R, Supraspinatus_R, Teres_Major_R, Teres_Minor_R
  - Posterior Forearm (Left): Extensor_Indicis, Extensor_Carpi_Radialis_Brevis, Extensor_Carpi_Radialis_Longus, Extensor_Digitorum, Extensor_Digiti_Minimi, Extensor_Carpi_Ulnaris, Anconeus, Abductor_Pollicis_Longus, Extensor_Pollicis_Longus, Extensor_Pollicis_Brevis, Supinator
  - Posterior Forearm (Right): Extensor_Indicis_R, Extensor_Carpi_Radialis_Brevis_R, Extensor_Carpi_Radialis_Longus_R, Extensor_Digitorum_R, Extensor_Digiti_Minimi_R, Extensor_Carpi_Ulnaris_R, Anconeus_R, Abductor_Pollicis_Longus_R, Extensor_Pollicis_Longus_R, Extensor_Pollicis_Brevis_R, Supinator_R
  - Anterior Forearm (Left): Flexor_Carpi_Radialis, Flexor_Carpi_Ulnaris, Pronator_Teres, Palmaris_Longus, Brachioradialis, Flexor_Digitorum_Superficialis, Flexor_Digitorum_Profundus, Flexor_Pollicis_Longus, Pronator_Quadratus
  - Anterior Forearm (Right): Flexor_Carpi_Radialis_R, Flexor_Carpi_Ulnaris_R, Pronator_Teres_R, Palmaris_Longus_R, Brachioradialis_R, Flexor_Digitorum_Superficialis_R, Flexor_Digitorum_Profundus_R, Flexor_Pollicis_Longus_R, Pronator_Quadratus_R

LEGS:
  - Thigh (Left): Adductor_Brevis, Adductor_Longus, Adductor_Magnus, Biceps_Femoris_Long_Head, Biceps_Femoris_Short_Head, Gracilis, Iliacus, Illiotibial_Tract, Pectineus, Rectus_Femoris, Sartorius, Semimembranosus, Semitendinosus, Tensor_Fascia_Lata, Vastus_Intermedius, Vastus_Lateralis, Vastus_Medialis
  - Thigh (Right): Adductor_Brevis_R, Adductor_Longus_R, Adductor_Magnus_R, Biceps_Femoris_Long_Head_R, Biceps_Femoris_Short_Head_R, Gracilis_R, Iliacus_R, Illiotibial_Tract_R, Pectineus_R, Rectus_Femoris_R, Sartorius_R, Semimembranosus_R, Semitendinosus_R, Tensor_Fascia_Lata_R, Vastus_Intermedius_R, Vastus_Lateralis_R, Vastus_Medialis_R
  - Gluteal (Left): Gluteus_Maximus, Gluteus_Medius, Gluteus_Minimus
  - Gluteal (Right): Gluteus_Maximus_R, Gluteus_Medius_R, Gluteus_Minimus_R
  - Lower Leg (Left): Extensor_Digitorum_Longus, Extensor_Hallucis_Longus, Fibularis_Brevis, Fibularis_Longus, Flexor_Digitorum_Longus, Flexor_Hallucis_Longus, Gastrocnemius_Lateral_Medial, Patellar_Ligament, Soleus, Tibialis_Anterior
  - Lower Leg (Right): Extensor_Digitorum_Longus_R, Extensor_Hallucis_Longus_R, Fibularis_Brevis_R, Fibularis_Longus_R, Flexor_Digitorum_Longus_R, Flexor_Hallucis_Longus_R, Gastrocnemius_Lateral_Medial_R, Patellar_Ligament_R, Soleus_R, Tibialis_Anterior_R
'''

FUNCTIONAL_GROUPS_STR = '''
PUSHING_MUSCLES:
  - Chest: Pectoralis_Major_01_Clavicular, Pectoralis_Major_01_Clavicular_R, Pectoralis_Major_02_Sternocostal, Pectoralis_Major_02_Sternocostal_R, Pectoralis_Major_03_Abdominal, Pectoralis_Major_03_Abdominal_R
  - Shoulders: Deltoid_Anterior, Deltoid_Anterior_R
  - Arms: Triceps_Medial_Head, Triceps_Medial_Head_R, Triceps_Lateral_Long_Heads, Triceps_Lateral_Long_Heads_R

PULLING_MUSCLES:
  - Back: Latissimus_Dorsi, Latissimus_Dorsi_R, Rhomboideus_Major, Rhomboideus_Major_R, Rhomboideus_Minor, Rhomboideus_Minor_R, Trapezius_01_Upper, Trapezius_01_Upper_R, Trapezius_02_Middle, Trapezius_02_Middle_R, Trapezius_03_Lower, Trapezius_03_Lower_R
  - Arms: Biceps_Brachii, Biceps_Brachii_R

CORE_MUSCLES:
  - Abdomen: Rectus_Abdominis, Rectus_Abdominis_R, External_Oblique, External_Oblique_R
  - Lower Back: Iliocostalis_Lumborum, Iliocostalis_Lumborum_R, Longissimus_Thoracis, Longissimus_Thoracis_R

LOWER_BODY_PUSH:
  - Quadriceps: Rectus_Femoris, Rectus_Femoris_R, Vastus_Lateralis, Vastus_Lateralis_R, Vastus_Medialis, Vastus_Medialis_R, Vastus_Intermedius, Vastus_Intermedius_R
  - Calves: Gastrocnemius_Lateral_Medial, Gastrocnemius_Lateral_Medial_R, Soleus, Soleus_R

LOWER_BODY_PULL:
  - Hamstrings: Biceps_Femoris_Long_Head, Biceps_Femoris_Long_Head_R, Biceps_Femoris_Short_Head, Biceps_Femoris_Short_Head_R, Semimembranosus, Semimembranosus_R, Semitendinosus, Semitendinosus_R
  - Glutes: Gluteus_Maximus, Gluteus_Maximus_R
'''

MUSCLE_PAIRING_RULES = '''
Left-Right Muscle Pairs:
- For each muscle, if there exists a "_R" version, they are a pair
- Left side muscles never have a suffix (e.g., Biceps_Brachii)
- Right side muscles always end with "_R" (e.g., Biceps_Brachii_R)
- When selecting a bilateral muscle group, include both the base name and the "_R" version
- Some central/midline muscles may not have pairs (e.g., some facial muscles)
- Never use "_L" for left side muscles
'''

MUSCLE_NAMING_RULES = '''
Naming Rules:
- Always use PascalCase with underscores (e.g., Rectus_Femoris)
- Right side muscles always end with "_R" (e.g., Rectus_Femoris_R)
- Left side muscles have no special suffix
- Never use spaces in muscle names
- Some muscles have numerical components (e.g., Pectoralis_Major_01_Clavicular)
- Multiple-word muscles use underscores between all words (e.g., Flexor_Digitorum_Longus)
- Never abbreviate muscle names
- Never use lowercase letters in muscle names
'''

# Unified system prompt for all model control
SYSTEM_PROMPT = f"""
[Persona]
You're an enthusiastic, experienced fitness coach who LOVES using the 3D anatomy model tools to help clients understand their muscles better. You're energetic, motivating, and speak like a real gym trainer - not a medical textbook. 
You get excited about using the muscle highlighting tools and camera controls to create the perfect demonstration!

[Core Focus]
- Help users achieve their fitness goals by explaining exercises, movements, and muscle functions.
- Use the tools to control the 3D model to show muscles in the context of workouts and training, not just for anatomy education.
- Connect every muscle explanation to real-world fitness benefits and practical exercises.
- Use the muscle highlighting tools first, then adjust the camera for the clearest view!

[Task]
- Proactively demonstrate relevant fitness muscles and anatomy for the user's question using the 3D model.
- For simple, direct questions about specific muscles or movements, immediately highlight the relevant muscles.
- Unless the user asks for a specific muscle, prioritize highlighting the whole muscle group(s) related to the question.
- For complex questions requiring multiple steps or extensive highlighting, first explain your approach.
- Always use the exact muscle names as defined in the [Available Muscles] section.
- When the user mentions a muscle group, ambiguous muscle, or common name (e.g., "bicep"), expand it to all relevant anatomical muscles using the [Available Muscles], [Functional Muscle Groups], and [Exercise-Specific Muscle Groups] sections.
- After highlighting muscles, adjust the camera to the best angle to demonstrate the relevant anatomy.
- Use distinct, visually clear colors for each muscle (unless the user requests a specific color).
- Report back what changes you made, including which muscles were highlighted and their colors.

[Naming Instructions]
- Use PascalCase with underscores for all muscle names (e.g., Zygomaticus_Major, Pectoralis_Major_01_Clavicular).
- For right-side muscles, append _R (e.g., Gluteus_Maximus_R).
- For left-side muscles, use the base name with NO suffix (e.g., Gluteus_Maximus).
- Do NOT use _L, spaces, lowercase, or any other formats.

[Available Muscles]
The 3D model has the following muscles arranged by region. Always use these exact names:
{MUSCLE_MAPPING_STR}

[Functional Muscle Groups]
When highlighting muscles related to specific movements or exercises, use these predefined groups:
{FUNCTIONAL_GROUPS_STR}

[Muscle Pairing Rules]
{MUSCLE_PAIRING_RULES}

[Naming Rules]
{MUSCLE_NAMING_RULES}

[Current Model State]
- Highlighted muscles (with colors): {{highlighted_muscles_str}}
- Camera: {{camera_str}}

[Tool Usage Instructions]
- **select_muscles(muscle_names: list, colors: dict)**: Highlight specific muscles. Always use the exact muscle names from [Available Muscles]. The `colors` argument should be a dictionary mapping each muscle name to a hex color (e.g., `{{"Biceps_Brachii": "#FFD600"}}`). If the user does not specify colors, assign a distinct, visually clear color to each muscle.
- **toggle_muscle(muscle_name: str, color: str)**: Toggle highlight for a single muscle. Use the correct muscle name and a hex color.
- **set_camera_view(position_x: float, position_y: float, position_z: float, target_x: float, target_y: float, target_z: float)**: Set both camera position and target in a single call. Use the predefined presets from the [Camera Control Guidelines] section.
- **reset_camera()**: Reset the camera to the default full body front view.

[Camera Control Guidelines]
For the clearest demonstrations, ALWAYS use these specific presets based on the muscle group:

Upper Body Front View (for chest, biceps, abs):
- Position: x: -0.03, y: 0.83, z: 3.48
- Target: x: -0.03, y: 0.83, z: ~0

Upper Body Back View (for back, shoulders):
- Position: x: 0.20, y: 1.53, z: -3.70
- Target: x: 0.07, y: 0.77, z: 0.16

Lower Body Front View (for quads, calves):
- Position: x: -0.0007, y: -0.50, z: 4.45
- Target: x: 0.0006, y: -0.50, z: 0.00006

Lower Body Back View (for glutes, hamstrings):
- Position: x: 0.20, y: 0.26, z: -4.21
- Target: x: 0.06, y: -0.56, z: -0.11

[Best Practices for Tool Use]
- First highlight the relevant muscles, then adjust the camera for the best view!
- Never invent muscle names. Only use names from [Available Muscles].
- When highlighting multiple muscles, always provide a `colors` dictionary mapping each muscle to a color.
- If the user requests a group or region, expand it to the correct list of muscle names using the [Functional Muscle Groups] or [Exercise-Specific Muscle Groups] sections.
- If the user requests "both sides" or "bilateral", include both the base name and the "_R" version for each muscle.
- For simple questions about specific muscles or movements, immediately highlight the relevant muscles.
- For complex questions requiring multiple steps, first explain your approach before making changes.
- If unsure about a muscle name or group, ask the user for clarification.

[Reporting]
After using any tools, provide a concise summary of what you changed in the model, including which muscles were highlighted and their colors, and any camera changes.

- For every user request to highlight, show, or demonstrate a muscle or group, you MUST use the select_muscles or toggle_muscle tool, even if you have already described it in text.
- Never just say you will highlight a muscle—always call the tool to actually do it.
- After highlighting muscles, adjust the camera to the appropriate view using the presets above!
- After highlighting muscles and adjusting the camera, provide a detailed explanation of what was highlighted and why it's relevant to the user's question.
"""

# Response generation prompt - used after tools are executed
RESPONSE_PROMPT = """
[Persona]
You are a friendly, brutally honest fitness coach who uses a 3D anatomy model to help clients understand their muscles, bodies, and exercises better. 
You are knowledgeable but conversational, using gym-friendly language that balances technical accuracy with practical advice.

[Task]
- Respond directly to the user's question based on the now-visible highlighted muscles
- Reference the conversation history to maintain context and avoid repeating information
- For each muscle, include both its scientific name and common name when first mentioning it
- Keep explanations concise but informative

[Context]
- User question: {user_question}
- Your initial assessment: {initial_assessment}
- Changes made to the model: {model_changes}
- Conversation history provides important context for your response

[Response Style]
- Be conversational and dynamic - vary your structure and approach based on the query
- Use appropriate fitness terminology while remaining accessible to all knowledge levels
- Mix scientific knowledge with practical coaching advice
- When highlighting bilateral muscles (left/right pairs), describe their function once rather than repeating
- Avoid the repetitive format of listing every individual muscle - integrate information naturally
- Reference previous conversation points where relevant to build continuity

"""

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, streaming=True)

# Map of tool names to their actual function implementations
TOOL_MAP = {
    "select_muscles": select_muscles_tool,
    "toggle_muscle": toggle_muscle_tool,
    "set_animation_frame": set_animation_frame_tool,
    "toggle_animation": toggle_animation_tool,
    "set_camera_position": set_camera_position_tool,
    "set_camera_target": set_camera_target_tool,
    "set_camera_view": set_camera_view_tool,
    "reset_camera": reset_camera_tool
}

# Remove animation tools from tool selection
MODEL_CONTROL_TOOL_FUNCTIONS_NO_ANIMATION = [
    select_muscles_tool,
    toggle_muscle_tool,
    set_camera_view_tool,
    reset_camera_tool
]

def summarize_changes(old_state, new_state):
    messages = []
    # Muscles
    old_muscles = old_state.get("highlighted_muscles", {})
    new_muscles = new_state.get("highlighted_muscles", {})
    if new_muscles != old_muscles:
        if new_muscles:
            muscle_str = ", ".join(f"{name} (color: {color})" for name, color in new_muscles.items())
            messages.append(f"I've highlighted the following muscles for you: {muscle_str}.")
        else:
            messages.append("I've cleared the muscle highlights.")
    # Camera position
    old_pos = old_state.get("camera", {}).get("position", {})
    new_pos = new_state.get("camera", {}).get("position", {})
    if new_pos != old_pos:
        messages.append("I've moved the camera to give you a better view.")
    # Camera target
    old_target = old_state.get("camera", {}).get("target", {})
    new_target = new_state.get("camera", {}).get("target", {})
    if new_target != old_target:
        messages.append("I've adjusted what the camera is looking at.")
    # Animation
    old_anim = old_state.get("animation", {})
    new_anim = new_state.get("animation", {})
    if new_anim != old_anim:
        if new_anim.get("isPlaying", False):
            messages.append(f"I've started the animation at frame {new_anim.get('frame', 0)}.")
        else:
            messages.append(f"I've paused the animation at frame {new_anim.get('frame', 0)}.")
    return " ".join(messages) if messages else "I've made the requested changes to the 3D model."

async def model_agent(state: ModelState, writer: Optional[StreamWriter] = None) -> Command[Literal[END]]:
    print("MODEL_AGENT: writer is", writer)
    if writer: 
        writer({"tester": "Streaming custom data while generating, this is fro testing purposes"})
    # Format state for prompt
    highlighted_muscles = state.get("highlighted_muscles", {})
    highlighted_muscles_str = (
        ", ".join(f"{name} (color: {color})" for name, color in highlighted_muscles.items())
        if highlighted_muscles else "None"
    )
    animation = state.get("animation", {"frame": 0, "isPlaying": False})
    animation_str = f"Frame: {animation.get('frame', 0)}, Playing: {animation.get('isPlaying', False)}"
    camera = state.get("camera", {"position": {"x": 0, "y": 1, "z": 7}, "target": {"x": 0, "y": 0, "z": 0}})
    camera_str = f"Position: {camera.get('position', {})}, Target: {camera.get('target', {})}"
    
    system_message = SystemMessage(content=SYSTEM_PROMPT) #.format(
    #     highlighted_muscles=highlighted_muscles_str, 
    #     camera=camera_str
    # ))
    messages = state.get("messages", [])
    
    print(f"\n{'='*50}\nModel agent received state with {len(messages)} messages")
    for i, msg in enumerate(messages):
        print(f"Message {i}: role={msg.get('role', 'unknown')}, content={msg.get('content', '')[:50]}...")
    
    if not messages or messages[-1]["role"] != "user":
        print("No messages or last message is not from user, returning without action")
        return Command(goto=END)
        
    user_message = HumanMessage(content=messages[-1]["content"])
    
    # Create full conversation history for LLM context
    conversation_history = []
    for msg in messages[:-1]:  # Skip the latest user message which we'll add separately
        if msg["role"] == "user":
            conversation_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            conversation_history.append(AIMessage(content=msg["content"]))
    
    # Start with system message, add conversation history, then add latest user message
    full_prompt = (
        [system_message]
        + [SystemMessage(content=f"[Current states of the model]\n\nHighlighted muscles: {highlighted_muscles_str}\nAnimation: {animation_str}\nCamera: {camera_str}")]
        + [SystemMessage(content="[Conversation History Start]")]
        + conversation_history
        + [user_message]
        + [SystemMessage(content="[Conversation History End]")]
    )
    
    print(f"Sending {len(full_prompt)} messages to LLM: 1 system + {len(conversation_history)} history + 1 new user")
    
    if writer:
        writer({"type": "thinking", "content": "Processing your request..."})

    print(f"Processing user message: {user_message.content}")
    print(f"Current state before tool call: highlighted_muscles={highlighted_muscles}, events={state.get('events', [])}")
    
    # Step 1: Get initial LLM assessment for tools and capture initial thoughts
    response = await llm.bind_tools(
        MODEL_CONTROL_TOOL_FUNCTIONS_NO_ANIMATION,
        tool_choice="auto"
    ).ainvoke(
        full_prompt
    )
    
    print(f"LLM response type: {type(response)}")
    print(f"LLM response: {response}")

    # Initialize state for chaining
    current_state = state.copy()
    all_events = current_state.get("events", []).copy()
    initial_assessment = getattr(response, 'content', str(response)) or ""
    
    # Step 2: Execute any tools the LLM decided to use
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"Tool calls detected: {response.tool_calls}")
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"Executing tool: {tool_name} with args: {tool_args}")

            try:
                if tool_name == "select_muscles":
                    muscle_names = tool_args.get("muscle_names", [])
                    colors = tool_args.get("colors", None)
                    print(f"Selecting muscles: {muscle_names} with colors: {colors}")
                    # Default color if not specified
                    default_color = "#FFD600"
                    muscle_color_map = {name: (colors[name] if colors and name in colors else default_color) for name in muscle_names}
                    event = {
                        "type": "model:selectMuscles",
                        "payload": {"muscleNames": muscle_names, "colors": muscle_color_map}
                    }
                    all_events.append(event)
                    current_state["highlighted_muscles"] = muscle_color_map
                elif tool_name == "toggle_muscle":
                    muscle_name = tool_args.get("muscle_name", "")
                    color = tool_args.get("color", None)
                    print(f"Toggling muscle: {muscle_name} with color: {color}")
                    event = {
                        "type": "model:toggleMuscle",
                        "payload": {"muscleName": muscle_name, "color": color or '#FFD600'}
                    }
                    all_events.append(event)
                    highlighted_muscles = current_state.get("highlighted_muscles", {}).copy()
                    if muscle_name in highlighted_muscles:
                        highlighted_muscles.pop(muscle_name)
                    else:
                        highlighted_muscles[muscle_name] = color or '#FFD600'
                    current_state["highlighted_muscles"] = highlighted_muscles
                elif tool_name == "set_camera_position":
                    x = tool_args.get("x", 0)
                    y = tool_args.get("y", 0)
                    z = tool_args.get("z", 0)
                    print(f"Setting camera position: x={x}, y={y}, z={z}")
                    
                    # Ensure values are within reasonable ranges
                    # Clamp x between -7 and 7
                    x = max(-7, min(7, x))
                    # Clamp y between -1 and 2 (avoiding positions below or too high)
                    y = max(-1, min(2, y))
                    # Clamp z between -7 and 7
                    z = max(-7, min(7, z))
                    
                    # Create position object
                    position = {"x": x, "y": y, "z": z}
                    
                    # Create event
                    event = {
                        "type": "model:setCameraPosition",
                        "payload": {"position": position}
                    }
                    
                    # Add to events list
                    all_events.append(event)
                    
                    # Update camera state
                    camera = current_state.get("camera", {"position": {}, "target": {}}).copy()
                    camera["position"] = position
                    
                    # Update state
                    current_state["camera"] = camera
                
                elif tool_name == "set_camera_target":
                    x = tool_args.get("x", 0)
                    y = tool_args.get("y", 0)
                    z = tool_args.get("z", 0)
                    print(f"Setting camera target: x={x}, y={y}, z={z}")
                    
                    # Ensure values are within reasonable ranges
                    # Keep target.x close to center (±0.2 range)
                    x = max(-0.2, min(0.2, x))
                    # Clamp y between -0.6 and 1 (avoiding targets too low or high)
                    y = max(-0.6, min(1, y))
                    # Keep target.z close to center (±0.2 range)
                    z = max(-0.2, min(0.2, z))
                    
                    # Create target object
                    target = {"x": x, "y": y, "z": z}
                    
                    # Create event
                    event = {
                        "type": "model:setCameraTarget",
                        "payload": {"target": target}
                    }
                    
                    # Add to events list
                    all_events.append(event)
                    
                    # Update camera state
                    camera = current_state.get("camera", {"position": {}, "target": {}}).copy()
                    camera["target"] = target
                    
                    # Update state
                    current_state["camera"] = camera
                
                elif tool_name == "set_camera_view":
                    # Extract position coordinates
                    position_x = tool_args.get("position_x", 0)
                    position_y = tool_args.get("position_y", 0)
                    position_z = tool_args.get("position_z", 0)
                    
                    # Extract target coordinates
                    target_x = tool_args.get("target_x", 0)
                    target_y = tool_args.get("target_y", 0)
                    target_z = tool_args.get("target_z", 0)
                    
                    print(f"Setting camera view: position: ({position_x}, {position_y}, {position_z}), target: ({target_x}, {target_y}, {target_z})")
                    
                    # Ensure position values are within reasonable ranges
                    position_x = max(-7, min(7, position_x))
                    position_y = max(-1, min(2, position_y))
                    position_z = max(-7, min(7, position_z))
                    
                    # Ensure target values are within reasonable ranges
                    target_x = max(-0.2, min(0.2, target_x))
                    target_y = max(-0.6, min(1, target_y))
                    target_z = max(-0.2, min(0.2, target_z))
                    
                    # Create position and target objects
                    position = {"x": position_x, "y": position_y, "z": position_z}
                    target = {"x": target_x, "y": target_y, "z": target_z}
                    
                    # Create a single combined event
                    event = {
                        "type": "model:setCameraView",
                        "payload": {"position": position, "target": target}
                    }
                    
                    # Add to events list
                    all_events.append(event)
                    
                    # Update camera state
                    camera = current_state.get("camera", {"position": {}, "target": {}}).copy()
                    camera["position"] = position
                    camera["target"] = target
                    
                    # Update state
                    current_state["camera"] = camera
                
                elif tool_name == "reset_camera":
                    print("Resetting camera")
                    
                    # Create event
                    event = {
                        "type": "model:resetCamera",
                        "payload": {}
                    }
                    
                    # Add to events list
                    all_events.append(event)
                    
                    # Use the optimal Full Body Front view from guidelines
                    default_position = {"x": 0, "y": 1, "z": 6.5}
                    default_target = {"x": 0, "y": 0, "z": 0}
                    camera = {
                        "position": default_position,
                        "target": default_target
                    }
                    
                    # Update state
                    current_state["camera"] = camera
                
                print(f"State updates after tool execution: {current_state}")
                
            except Exception as e:
                print(f"ERROR executing tool {tool_name}: {e}")

        # After all tool calls, update the state and events
        current_state["events"] = all_events
        
        # Step 3: Summarize the changes made to the model
        model_changes = summarize_changes(state, current_state)
        summary_of_changes = summarize_changes(state, current_state)
        # Build the new prompt
        system_message = SystemMessage(content=RESPONSE_PROMPT + f"\nHighlighted muscles: {highlighted_muscles_str}\nAnimation: {animation_str}\nCamera: {camera_str}\nChanges: {summary_of_changes}")
        user_message = HumanMessage(content=messages[-1]["content"])
        tool_info_message = HumanMessage(content=f"Tool actions taken: {model_changes}")
        final_prompt = [system_message] + conversation_history + [user_message, tool_info_message]
        if writer:
            async for chunk in llm.astream(final_prompt):
                print("STREAMING TOKEN:", chunk)
                token = chunk.content if hasattr(chunk, 'content') else chunk
                await writer({"type": "response", "content": token})
            final_message = ""
        else:
            final_response = await llm.ainvoke(final_prompt)
            final_message = final_response.content
    else:
        # If no tool calls, use the regular response content
        if writer:
            async for chunk in llm.astream(full_prompt):
                print("STREAMING TOKEN:", chunk)
                token = chunk.content if hasattr(chunk, 'content') else chunk
                await writer({"type": "response", "content": token})
            final_message = ""
        else:
            final_message = initial_assessment or "I'll help you with that right away."
    # Add the response to messages
    new_messages = messages.copy()
    new_messages.append({"role": "assistant", "content": final_message})
    current_state["messages"] = new_messages
    print(f"Final state updates: {current_state}")
    print(f"Final message count in state: {len(current_state.get('messages', []))}")
    current_state["current_agent"] = "model_agent"
    return Command(goto=END, update=current_state) 