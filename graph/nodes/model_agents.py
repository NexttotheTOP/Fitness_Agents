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
    reset_camera_tool
)
import json

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
  - Back (Left): Iliocostalis_Lumborum, Latissimus_Dorsi, Longissimus_Thoracis, Rhomboideus_Major, Rhomboideus_Minor, Serratus_Posterior_Inferior, Serratus_Posterior_Superior, Spinalis_Thoracis, Splenius_Capitis, Splenius_Cervicis
  - Back (Right): Iliocostalis_Lumborum_R, Latissimus_Dorsi_R, Longissimus_Thoracis_R, Rhomboideus_Major_R, Rhomboideus_Minor_R, Serratus_Posterior_Inferior_R, Serratus_Posterior_Superior_R, Spinalis_Thoracis_R, Splenius_Capitis_R, Splenius_Cervicis_R
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

EXERCISE_GROUPS_STR = '''
BENCH_PRESS:
  - Primary: Pectoralis_Major_01_Clavicular, Pectoralis_Major_01_Clavicular_R, Pectoralis_Major_02_Sternocostal, Pectoralis_Major_02_Sternocostal_R, Pectoralis_Major_03_Abdominal, Pectoralis_Major_03_Abdominal_R, Triceps_Medial_Head, Triceps_Medial_Head_R, Triceps_Lateral_Long_Heads, Triceps_Lateral_Long_Heads_R
  - Secondary: Deltoid_Anterior, Deltoid_Anterior_R

SQUAT:
  - Primary: Rectus_Femoris, Rectus_Femoris_R, Vastus_Lateralis, Vastus_Lateralis_R, Vastus_Medialis, Vastus_Medialis_R, Vastus_Intermedius, Vastus_Intermedius_R, Gluteus_Maximus, Gluteus_Maximus_R
  - Secondary: Adductor_Magnus, Adductor_Magnus_R, Soleus, Soleus_R

DEADLIFT:
  - Primary: Gluteus_Maximus, Gluteus_Maximus_R, Biceps_Femoris_Long_Head, Biceps_Femoris_Long_Head_R, Biceps_Femoris_Short_Head, Biceps_Femoris_Short_Head_R, Semimembranosus, Semimembranosus_R, Semitendinosus, Semitendinosus_R
  - Secondary: Latissimus_Dorsi, Latissimus_Dorsi_R, Trapezius_01_Upper, Trapezius_01_Upper_R, Trapezius_02_Middle, Trapezius_02_Middle_R, Trapezius_03_Lower, Trapezius_03_Lower_R, Rectus_Abdominis, Rectus_Abdominis_R

SHOULDER_PRESS:
  - Primary: Deltoid_Anterior, Deltoid_Anterior_R, Deltoid_Middle, Deltoid_Middle_R, Triceps_Medial_Head, Triceps_Medial_Head_R, Triceps_Lateral_Long_Heads, Triceps_Lateral_Long_Heads_R
  - Secondary: Trapezius_01_Upper, Trapezius_01_Upper_R, Serratus_Anterior, Serratus_Anterior_R

BICEP_CURL:
  - Primary: Biceps_Brachii, Biceps_Brachii_R
  - Secondary: Brachialis, Brachialis_R, Brachioradialis, Brachioradialis_R

TRICEP_EXTENSION:
  - Primary: Triceps_Lateral_Long_Heads, Triceps_Lateral_Long_Heads_R, Triceps_Medial_Head, Triceps_Medial_Head_R
  - Secondary: Anconeus, Anconeus_R
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
You are an expert fitness assistant specializing in human anatomy, physiology, and exercise science. You help users understand the human body and exercises by answering questions and visually demonstrating concepts using a 3D human anatomy model.

[Task]
- Execute requested model changes with precision.
- Select and highlight the correct muscles based on the user's request and context.
- Always use the exact muscle names as defined in the [Available Muscles] section.
- When the user requests a muscle group, ambiguous muscle, or common name (e.g., "bicep"), expand it to all relevant anatomical muscles using the [Available Muscles], [Functional Muscle Groups], and [Exercise-Specific Muscle Groups] sections. Highlight all relevant muscles in distinct, visually clear colors (unless the user requests a specific color).
- Choose appropriate camera angles to best demonstrate the relevant anatomy.
- When highlighting multiple muscles, use distinct, visually clear colors for each muscle (unless the user requests a specific color).
- Report back exactly what changes you made, including which muscles were highlighted and their colors.

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

[Exercise-Specific Muscle Groups]
For common exercises, these are the primary and secondary muscles involved:
{EXERCISE_GROUPS_STR}

[Muscle Pairing Rules]
{MUSCLE_PAIRING_RULES}

[Naming Rules]
{MUSCLE_NAMING_RULES}

[Current Model State]
- Highlighted muscles (with colors): {{highlighted_muscles}}
- Camera: {{camera}}

[Tool Usage Instructions]
- **select_muscles(muscle_names: list, colors: dict)**: Highlight specific muscles. Always use the exact muscle names from [Available Muscles]. The `colors` argument should be a dictionary mapping each muscle name to a hex color (e.g., `{{"Biceps_Brachii": "#FFD600"}}`). If the user does not specify colors, assign a distinct, visually clear color to each muscle.
- **toggle_muscle(muscle_name: str, color: str)**: Toggle highlight for a single muscle. Use the correct muscle name and a hex color.
- **set_camera_position(x: float, y: float, z: float)**: Move the camera to a specific position.
- **set_camera_target(x: float, y: float, z: float)**: Change what the camera is looking at.
- **reset_camera()**: Reset the camera to the default position and target.

[Best Practices for Tool Use]
- Never invent muscle names. Only use names from [Available Muscles].
- When highlighting multiple muscles, always provide a `colors` dictionary mapping each muscle to a color.
- If the user requests a group or region, expand it to the correct list of muscle names using the [Functional Muscle Groups] or [Exercise-Specific Muscle Groups] sections.
- If the user requests "both sides" or "bilateral", include both the base name and the "_R" version for each muscle.
- If unsure about a muscle name or group, ask the user for clarification.

[Reporting]
After using any tools, provide a concise summary of what you changed in the model, including which muscles were highlighted and their colors, and any camera changes.
"""

# Response generation prompt - used after tools are executed
RESPONSE_PROMPT = """
[Persona]
You are an expert fitness coach that helps his clients/users with their workouts and fitness questions. You use a 3D anatomy model to teach and engage.
You help users understand the human body and exercises by answering questions and visually demonstrating concepts using a 3D human anatomy model.
The demonstration fo the human model will be done by your AI collegeau and is already happened, you will find the changes made below.

[Task]
- Explain to the user what you did with the 3D model and be informative.
- For each highlighted muscle, mention its color and provide a brief, friendly explanation of its function or importance (e.g., "In blue you can see the Deltoid_Anterior, which ...").
- Be proactive, friendly, and educational. Use casual, supportive language as if chatting with a friend at the gym.
- Offer to answer more questions or provide more detail, and encourage the user to ask about other muscles, exercises, or movements.

[Context]
- User question: {user_question}
- Your initial assessment: {initial_assessment}
- Changes that your AI collegeau made to the model: {model_changes}

[Response Guidelines]
- Start with a direct, friendly answer that references the user's request.
- For each highlighted muscle, mention its color and give a one-sentence explanation of its function or importance.
- End by inviting the user to ask more questions or explore other muscles or exercises.
- Stay brutally honest, supportive, and approachable.
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
    "reset_camera": reset_camera_tool
}

# Remove animation tools from tool selection
MODEL_CONTROL_TOOL_FUNCTIONS_NO_ANIMATION = [
    select_muscles_tool,
    toggle_muscle_tool,
    set_camera_position_tool,
    set_camera_target_tool,
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

def model_agent(state: ModelState, writer: Optional[StreamWriter] = None) -> Command[Literal[END]]:
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
    
    system_message = SystemMessage(content=SYSTEM_PROMPT + f"\nHighlighted muscles: {highlighted_muscles_str}\nAnimation: {animation_str}\nCamera: {camera_str}")
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
    full_prompt = [system_message] + ["[Conversation History Start]"] + conversation_history + ["[Conversation History End]"] + [user_message]
    
    print(f"Sending {len(full_prompt)} messages to LLM: 1 system + {len(conversation_history)} history + 1 new user")
    
    if writer:
        writer({"type": "thinking", "content": "Processing your request..."})

    print(f"Processing user message: {user_message.content}")
    print(f"Current state before tool call: highlighted_muscles={highlighted_muscles}, events={state.get('events', [])}")
    
    # Step 1: Get initial LLM assessment for tools and capture initial thoughts
    response = llm.bind_tools(
        MODEL_CONTROL_TOOL_FUNCTIONS_NO_ANIMATION,
        tool_choice="auto"
    ).invoke(
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
                
                elif tool_name == "reset_camera":
                    print("Resetting camera")
                    
                    # Create event
                    event = {
                        "type": "model:resetCamera",
                        "payload": {}
                    }
                    
                    # Add to events list
                    all_events.append(event)
                    
                    # Create default camera settings
                    default_position = {"x": 0, "y": 1, "z": 7}
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
        
        # Step 4: Instead of a separate response prompt, re-invoke the LLM
        # Build a prompt that includes:
        # - The original user message
        # - The updated state (highlighted muscles, camera, etc.)
        # - A summary of tool actions/events (optional, but helpful)
        # - Instructions to generate a friendly, concise message

        summary_of_changes = summarize_changes(state, current_state)
        # Or, pass the actual tool results/events if you want more detail

        # Build the new prompt
        system_message = SystemMessage(content=RESPONSE_PROMPT + f"\nHighlighted muscles: {highlighted_muscles_str}\nAnimation: {animation_str}\nCamera: {camera_str}\nChanges: {summary_of_changes}")
        user_message = HumanMessage(content=messages[-1]["content"])
        tool_info_message = HumanMessage(content=f"Tool actions taken: {model_changes}")

        # Compose the prompt for the LLM
        final_prompt = [system_message] + conversation_history + [user_message, tool_info_message]

        # Call the LLM to generate the final message
        final_response = llm.invoke(final_prompt)
        final_message = final_response.content
    else:
        # If no tool calls, use the regular response content
        final_message = initial_assessment or "I'll help you with that right away."

    # Add the response to messages
    new_messages = messages.copy()
    new_messages.append({"role": "assistant", "content": final_message})
    current_state["messages"] = new_messages

    print(f"Final state updates: {current_state}")
    print(f"Final message count in state: {len(current_state.get('messages', []))}")

    current_state["current_agent"] = "model_agent"
    return Command(goto=END, update=current_state) 