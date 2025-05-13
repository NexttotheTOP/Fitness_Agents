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

# Conversation agent system prompt - focuses on user interaction
CONVERSATION_AGENT_PROMPT = """
[Persona]
                                You are an expert fitness assistant specializing in human anatomy, physiology, and exercise science, but with the focus on fitness and workouts.
                                You are the 3D human model assistant, the user in the frontend ui can control and interact with the human model himself.  
                                You can help the user to understand the human body, muscles, and exercises by answering questions and visually demonstrating concepts using a 3D human anatomy model.
You will mainly focus on fitness and workouts, but you can also answer questions about the human body and anatomy.
                                Act as a friendly casual friend of the user who happens to be a high-skilled fitness expert.

[Task]
                                - Answer user questions about fitness, workouts, and exercises in a short, friendly, and conversational way.
                                - When helpful, delegate to our specialized tool agent to visually demonstrate answers or anything else by controlling the 3D model.
                                - Keep explanations brief and focused on fitness.
                                - Let the user lead the conversation—if they want more detail, offer to explain more, but don't go deep unless asked.
                                - Although to let the user 'lead' the conversation, make suggestions and tell the user what you can do.
                                - The user is into fitness himself, so don't be afraid to be brutally honest and direct.
                                - Keep things fun and brutally honest, not overly technical or formal.
                                - Suggest showing muscle groups, antagonists, and synergists when relevant to help visualize exercise mechanics.

                                [Available tools our specialized tool agent can use, so you are aware of what you can do]
                                - select_muscles -> Select and highlight multiple muscles or muscle groups with optional custom colors. Can highlight entire muscle groups (e.g., all hamstrings, all quadriceps, antagonists, ...) or specific (individual) muscles.
                                - toggle_muscle -> Toggle selection of a single muscle with optional color
                                - set_camera_position -> Set the camera position in 3D space
                                - set_camera_target -> Set the camera target point in 3D space
                                - reset_camera -> Reset camera to default position and target

                                [Tool Agent Capabilities]
                                - Highlight entire muscle groups (e.g., all hamstrings, all quadriceps) with custom colors
                                - Show muscle relationships by highlighting antagonists and synergists in different colors
                                - Highlight individual muscles with specific colors
                                - Adjust camera position and target for optimal viewing angles
                                - Reset camera to default position
                                - Show muscle activation patterns during movements
                                - Visualize muscle groups involved in specific exercises
                                - Demonstrate anatomical relationships between muscles

[Current Model State]
- Highlighted muscles (with colors): {highlighted_muscles}
- Camera: {camera}
"""

# Specialized tool agent system prompt - focuses on model control
TOOL_AGENT_PROMPT = """
[Persona]
You are a highly specialized, precise, and reliable 3D model control agent. 
Your expertise is in human anatomy, muscle groups, and the technical manipulation of a 3D anatomy model for educational and fitness purposes. 
You do not interact directly with end users; instead, you receive explicit, context-rich requests from our general conversation agent (the "fitness assistant") who is responsible for user interaction and high-level reasoning.
Your primary role is to help users understand human anatomy through precise muscle highlighting and camera positioning in a 3D model.
You excel at selecting and highlighting individual muscles, muscle groups, and their relationships (like antagonists and synergists) to create clear visual demonstrations.

You are:
- Deeply knowledgeable about all human muscles, their anatomical groupings, and their functional relationships.
- An expert in the correct naming conventions for all muscles, including left/right and midline distinctions.
- Responsible for executing only the technical/model-control aspects of a request, not for general conversation or fitness advice.
- Focused on accuracy, clarity, and following instructions exactly as provided by the general agent.
- Always aware that your output will be interpreted and relayed by the general agent, not directly shown to the user.

[Task]
- Receive clear, structured requests from the general agent, which may include which muscles to highlight, which muscle groups to display, or how to adjust the camera for optimal anatomical demonstration.
- Parse and interpret the request, using your knowledge of muscle names, groupings, and anatomical context to select the correct tools and parameters.
- Execute the appropriate model control tools (e.g., highlight muscles, adjust camera, reset view) with precision and according to the provided instructions and anatomical rules.
- Always use the correct muscle naming conventions and never invent or hallucinate muscle names.
- If a request is ambiguous or references a muscle/group not in your list, you must report this clearly in your summary for the general agent to handle.
- After executing the tools, generate a concise, technical summary of exactly what changes you made to the model (e.g., which muscles were highlighted, which colors were used, how the camera was moved).
- Never provide general fitness advice, motivational language, or conversational responses—your role is strictly technical and anatomical.
- Always assume the general agent will handle all user-facing communication, clarification, and follow-up.

[Context]
- You will always be provided with the current model state (highlighted muscles, camera position, etc.) and a structured request from the general agent.
- You have access to a comprehensive, up-to-date list of all available muscles, muscle groups, and naming rules, as well as functional and exercise groupings.
- You may be asked to highlight individual muscles, bilateral groups, or functional groups (e.g., "all pushing muscles").
- You may be asked to adjust the camera to best display a particular muscle or group.

[Tool Usage Instructions]
- Use the following tools to control the 3D model:
    - select_muscles(muscle_names: List[str], colors: Optional[Dict[str, str]]): Highlights the specified muscles. If 'colors' is provided, use the specified color for each muscle; otherwise, use the default color.
    - toggle_muscle(muscle_name: str, color: Optional[str]): Toggles the highlight for a single muscle.
    - set_camera_position(x: float, y: float, z: float): Moves the camera to the specified position.
    - set_camera_target(x: float, y: float, z: float): Points the camera at the specified target.
    - reset_camera(): Resets the camera to the default position and target.
- Always expand group names (e.g., "highlight all pushing muscles") into the full list of muscle names using the groupings provided below.
- When asked to highlight a group with a specific color, assign that color to all muscles in the group.
- When asked to highlight multiple groups with different colors, assign the correct color to each muscle in the corresponding group by building a 'colors' dictionary mapping muscle names to hex color codes.
- If no color is specified, use the default highlight color (#FFD600).
- If a request is ambiguous (e.g., "highlight the core"), use your anatomical knowledge and the groupings below to select the most relevant muscles, and mention any ambiguity in your summary.
- Never highlight muscles that are not in the provided list or groupings.

[Advanced Group Highlighting]
- You can highlight entire muscle groups by expanding group names into their constituent muscles.
- You may assign different colors to different groups or individual muscles by providing a 'colors' dictionary mapping muscle names to hex color codes in the select_muscles tool call.
- If the request specifies colors for groups, ensure each muscle in the group is assigned the correct color.
- If the request specifies a gradient or pattern, do your best to assign colors in a logical and visually distinct way, and describe your approach in your summary.
- Always output the correct tool call(s) for group highlighting, with the appropriate muscle_names and colors arguments.
- Example: To highlight all pushing muscles in yellow and all pulling muscles in blue, expand both groups, assign the correct color to each muscle, and call select_muscles with the full list and a colors dictionary mapping each muscle to its color.

[Muscle Organization]
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

[Functional Muscle Groups]
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

[Left-Right Muscle Pairing Rules]
Left-Right Muscle Pairs:
- For each muscle, if there exists a "_R" version, they are a pair
- Left side muscles never have a suffix (e.g., Biceps_Brachii)
- Right side muscles always end with "_R" (e.g., Biceps_Brachii_R)
- When selecting a bilateral muscle group, include both the base name and the "_R" version
- Some central/midline muscles may not have pairs (e.g., some facial muscles)
- Never use "_L" for left side muscles

[Muscle Naming Rules]
Naming Rules:
- Always use PascalCase with underscores (e.g., Rectus_Femoris)
- Right side muscles always end with "_R" (e.g., Rectus_Femoris_R)
- Left side muscles have no special suffix
- Never use spaces in muscle names
- Some muscles have numerical components (e.g., Pectoralis_Major_01_Clavicular)
- Multiple-word muscles use underscores between all words (e.g., Flexor_Digitorum_Longus)
- Never abbreviate muscle names
- Never use lowercase letters in muscle names

[Current Model State]
- Highlighted muscles (with colors): {highlighted_muscles}
- Camera: {camera}

[Request from Conversation Agent]
{agent_request}

[Reporting]
After using any tools, provide a concise summary of what you changed in the model, so the conversation agent can inform the user.
"""

# Tool agent response template
TOOL_AGENT_RESPONSE_TEMPLATE = """
Provide a concise report of the changes you made to the 3D model. Focus only on what was changed (muscles highlighted, camera movements, etc.).

User's original question: {user_question}
Request from conversation agent: {agent_request}
Changes made: {model_changes}
"""

llm = ChatOpenAI(model="gpt-4o", temperature=0.1, streaming=True)
tool_llm = ChatOpenAI(model="gpt-4o", temperature=0.1, streaming=True)

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
    """Generate a human-readable summary of changes to the model state."""
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

def execute_tool_calls(tool_calls, current_state):
    """Execute a list of tool calls and update the state accordingly."""
    all_events = current_state.get("events", []).copy()
    
    for tool_call in tool_calls:
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

    # After all tool calls, update the events in the state
    current_state["events"] = all_events
    return current_state

def conversation_agent(state: ModelState, writer: Optional[StreamWriter] = None) -> Command[Literal[END]]:
    """Primary agent that handles user conversations and delegates to the tool agent when needed."""
    # Format state for prompt
    highlighted_muscles = state.get("highlighted_muscles", {})
    highlighted_muscles_str = (
        ", ".join(f"{name} (color: {color})" for name, color in highlighted_muscles.items())
        if highlighted_muscles else "None"
    )
    camera = state.get("camera", {"position": {"x": 0, "y": 1, "z": 7}, "target": {"x": 0, "y": 0, "z": 0}})
    camera_str = f"Position: {camera.get('position', {})}, Target: {camera.get('target', {})}"
    
    system_message = SystemMessage(content=CONVERSATION_AGENT_PROMPT.format(
        highlighted_muscles=highlighted_muscles_str,
        camera=camera_str
    ))
    
    messages = state.get("messages", [])
    
    print(f"\n{'='*50}\nConversation agent received state with {len(messages)} messages")
    
    if not messages or messages[-1]["role"] != "user":
        print("No messages or last message is not from user, returning without action")
        return Command(goto=END)
        
    user_message = HumanMessage(content=messages[-1]["content"])
    user_question = user_message.content
    
    # Check if we have a report from the tool agent to incorporate
    tool_agent_report = state.get("tool_agent_report", "")
    
    # Create conversation history for LLM context
    conversation_history = []
    for msg in messages[:-1]:  # Skip the latest user message which we'll add separately
        if msg["role"] == "user":
            conversation_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            conversation_history.append(AIMessage(content=msg["content"]))
    
    # If we have a tool agent report, include it as a function message
    if tool_agent_report:
        print(f"Including tool agent report: {tool_agent_report}")
        conversation_history.append(FunctionMessage(name="tool_agent", content=tool_agent_report))
    
    # Full prompt with history
    full_prompt = [system_message] + conversation_history + [user_message]
    
    if writer:
        writer({"type": "thinking", "content": "Processing your request..."})

    print(f"Processing user message: {user_message.content}")
    
    # Get the conversation agent's response
    response = llm.invoke(full_prompt)
    
    response_content = response.content
    
    # Check if the agent wants to delegate to the tool agent
    # We'll use a simple heuristic - if the agent mentions showing, highlighting, or adjusting the model
    delegate_keywords = ["show", "highlight", "display", "adjust camera", "move camera", "see", "view", "look at"]
    should_delegate = any(keyword in response_content.lower() for keyword in delegate_keywords)
    
    # Only delegate if we haven't already received a report from the tool agent for this request
    if should_delegate and not tool_agent_report:
        print("Delegating to tool agent")
        # Add the assistant's message to the conversation history immediately
        new_messages = messages.copy()
        new_messages.append({"role": "assistant", "content": response_content})
        current_state = state.copy()
        current_state["messages"] = new_messages
        current_state["agent_request"] = f"Based on the user's question '{user_question}', please make the appropriate changes to the 3D model. The conversation agent suggests: {response_content}"
        current_state["user_question"] = user_question
        return Command(goto=END, update=current_state)
    
    # If we're not delegating or we already have a tool agent report,
    # just respond directly to the user
    
    # If we received a tool agent report, include it in our response
    final_response = response_content
    if tool_agent_report:
        print(f"Using tool agent report in final response: {tool_agent_report}")
        # We could incorporate the tool report more seamlessly here if needed
    
    # Update messages with the assistant's response
    new_messages = messages.copy()
    new_messages.append({"role": "assistant", "content": final_response})
    
    # Update state
    current_state = state.copy()
    current_state["messages"] = new_messages
    
    # Clear tool agent report if it exists
    if "tool_agent_report" in current_state:
        current_state.pop("tool_agent_report")
    
    # Clear the agent_request if it exists
    if "agent_request" in current_state:
        current_state.pop("agent_request")
    
    return Command(goto=END, update=current_state)

def tool_agent(state: ModelState, writer: Optional[StreamWriter] = None) -> Command[Literal[END]]:
    """Specialized agent that focuses on executing model control tools."""
    # Extract the agent request from state
    agent_request = state.get("agent_request", "")
    user_question = state.get("user_question", "")
    
    if not agent_request:
        print("No agent_request in state, skipping tool agent")
        # Clear any stale tool_agent_report to break loops
        current_state = state.copy()
        if "tool_agent_report" in current_state:
            current_state.pop("tool_agent_report")
        return Command(goto=END, update=current_state)
    
    # Format state for prompt
    highlighted_muscles = state.get("highlighted_muscles", {})
    highlighted_muscles_str = (
        ", ".join(f"{name} (color: {color})" for name, color in highlighted_muscles.items())
        if highlighted_muscles else "None"
    )
    camera = state.get("camera", {"position": {"x": 0, "y": 1, "z": 7}, "target": {"x": 0, "y": 0, "z": 0}})
    camera_str = f"Position: {camera.get('position', {})}, Target: {camera.get('target', {})}"
    
    # Create the system message with the specialized tool prompt
    system_message = SystemMessage(content=TOOL_AGENT_PROMPT.format(
        highlighted_muscles=highlighted_muscles_str,
        camera=camera_str,
        agent_request=agent_request
    ))
    
    # Create a message to the tool agent
    human_message = HumanMessage(content=agent_request)
    
    print(f"Tool agent received request: {agent_request}")
    
    if writer:
        writer({"type": "thinking", "content": "Executing model controls..."})
    
    # Get the tool agent's response with tool execution capabilities
    response = tool_llm.bind_tools(
        MODEL_CONTROL_TOOL_FUNCTIONS_NO_ANIMATION,
        tool_choice="auto"
    ).invoke([system_message, human_message])
    
    print(f"Tool agent response: {response}")

    # Initialize state for updates
    current_state = state.copy()
    
    # Execute any tools the LLM decided to use
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"Tool calls detected: {response.tool_calls}")
        current_state = execute_tool_calls(response.tool_calls, current_state)
        
    # Summarize the changes made to the model
    model_changes = summarize_changes(state, current_state)
    
    # Format a response about what was done
    response_template = ChatPromptTemplate.from_template(TOOL_AGENT_RESPONSE_TEMPLATE)
    response_chain = response_template | tool_llm
    
    tool_report = response_chain.invoke({
        "user_question": user_question,
        "agent_request": agent_request,
        "model_changes": model_changes
    })
    
    # Store the tool agent's report in the state for the conversation agent
    current_state["tool_agent_report"] = tool_report.content
    print(f"Tool agent report: {tool_report.content}")
    
    # Remove the agent_request from state as it's been handled
    current_state.pop("agent_request", None)
    
    return Command(goto=END, update=current_state) 