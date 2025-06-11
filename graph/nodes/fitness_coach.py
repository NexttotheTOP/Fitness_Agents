from typing import Dict, Any, AsyncGenerator
from datetime import datetime
import os
import re
import requests
import base64
from io import BytesIO
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from graph.workout_state import WorkoutState, UserProfile, AgentState, QueryType
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
import json
from dotenv import load_dotenv
# Add Anthropic imports
import anthropic
from anthropic import Anthropic
# Add stream writer for custom streaming
from langgraph.config import get_stream_writer
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()



class ProfileAgent:
    def __init__(self):
        # Standard text model for non-visual analysis
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # Initialize Anthropic client for vision analysis
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            print("Warning: ANTHROPIC_API_KEY not found in environment variables")
            # Fall back to OpenAI if no Anthropic key is available
            self.use_claude = False
            # Vision model for body composition analysis (fallback to OpenAI)
            self.vision_model = ChatOpenAI(
                model="gpt-o3",
                temperature=0.1,
                max_tokens=4096,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()]
            )
        else:
            self.use_claude = True
            self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)
            self.claude_model = "claude-3-sonnet-20240229"
            print("Using Claude for vision analysis")
        
        # Get the Supabase service key and URL from environment variables
        self.supabase_service_key = os.environ.get("SUPABASE_SERVICE_KEY")
        self.supabase_url = os.environ.get("SUPABASE_URL", "https://hjzszgaugorgqbbsuqgc.supabase.co")
        
        if not self.supabase_service_key:
            print("Warning: SUPABASE_SERVICE_KEY not found in environment variables")
    
    def _extract_timestamp(self, filename: str) -> str:
        """Extract timestamp from image filename"""
        # Example: front-1745841393048-494691126_700784015796392_635117349760825196_n.jpg
        timestamp_match = re.search(r'-([\d]+)-', filename)
        if timestamp_match:
            unix_timestamp = int(timestamp_match.group(1)) / 1000  # Convert to seconds
            try:
                # Convert Unix timestamp to readable date
                date_time = datetime.fromtimestamp(unix_timestamp)
                return date_time.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, OverflowError):
                # If conversion fails, return the raw timestamp
                return timestamp_match.group(1)
        return "Unknown"
    
    def _get_image_as_base64(self, image_url: str) -> str:
        """Download image and convert to base64"""
        try:
            print(f"Attempting to download image from: {image_url}")
            
            # Check if this is a Supabase storage URL
            is_supabase_url = "supabase" in image_url and "/body-images/" in image_url
            
            # Set up authentication headers
            headers = {}
            if is_supabase_url and self.supabase_service_key:
                # For private Supabase buckets, we need to use authentication
                headers = {
                    "Authorization": f"Bearer {self.supabase_service_key}"
                }
                print("Using authenticated Supabase access with Authorization header")
                
                # Convert from public to authenticated URL pattern if needed
                if "/object/public/" in image_url:
                    image_url = image_url.replace("/object/public/", "/object/authenticated/")
                    print(f"Converted to authenticated URL: {image_url}")
            
            # Make the request with proper headers
            try:
                response = requests.get(image_url, headers=headers, timeout=15)
                response.raise_for_status()
                print(f"Successfully accessed image: Status {response.status_code}, Content-Length: {len(response.content)} bytes")
            except Exception as e:
                print(f"Initial access failed: {str(e)}")
                
                # If first attempt failed and it's a Supabase URL, try alternative endpoint
                if is_supabase_url and self.supabase_service_key:
                    try:
                        # Try the direct object endpoint without 'public' or 'authenticated'
                        alt_url = image_url.replace("/object/public/", "/object/")
                        alt_url = alt_url.replace("/object/authenticated/", "/object/")
                        
                        print(f"Trying alternative URL: {alt_url}")
                        response = requests.get(alt_url, headers=headers, timeout=15)
                        response.raise_for_status()
                        print(f"Successfully accessed image with alternative URL: Status {response.status_code}, Content-Length: {len(response.content)} bytes")
                    except Exception as e:
                        print(f"Alternative URL access failed: {str(e)}")
                        return None
            
            # Check if we actually got image data
            if not response.content or len(response.content) < 100:
                print(f"Warning: Response may not contain valid image data (too small: {len(response.content)} bytes)")
                return None
                
            # Check if image size is too large for API (typically 20MB limit for OpenAI)
            content_size_mb = len(response.content) / (1024 * 1024)
            if content_size_mb > 19:  # 19MB to be safe
                print(f"Warning: Image is too large ({content_size_mb:.2f} MB). Resizing might be needed.")
                # We continue for now, but future enhancement could resize the image
            
            # Determine MIME type from content (more reliable)
            content_type = response.headers.get('Content-Type', '')
            
            if 'image/' in content_type:
                mime_type = content_type  # Use the server-provided MIME type
            else:
                # Try to detect MIME type from URL or fall back to defaults
                if image_url.lower().endswith('.png'):
                    mime_type = "image/png"
                elif image_url.lower().endswith('.gif'):
                    mime_type = "image/gif"
                elif image_url.lower().endswith('.jpeg') or image_url.lower().endswith('.jpg'):
                    mime_type = "image/jpeg"
                elif image_url.lower().endswith('.webp'):
                    mime_type = "image/webp"
                elif "png" in image_url.lower():
                    mime_type = "image/png"
                else:
                    mime_type = "image/jpeg"  # Default to JPEG
            
            # Convert image to base64
            image_data = base64.b64encode(response.content).decode('utf-8')
            
            if self.use_claude:
                # For Claude, we need just the base64 string and media type, not a data URL
                print(f"Successfully encoded image as base64 ({mime_type}, {len(image_data)} chars)")
                return {"data": image_data, "media_type": mime_type}
            else:
                # For OpenAI, we need a data URL
                data_url = f"data:{mime_type};base64,{image_data}"
                data_sample = image_data[:20] + "..." if len(image_data) > 20 else image_data
                print(f"Successfully encoded image as base64 ({mime_type}, {len(image_data)} chars)")
                print(f"Data URL format: data:{mime_type};base64,{data_sample}...")
                
                # Verify the data URL format to ensure it's correct for the API
                if not data_url.startswith("data:image/"):
                    print(f"Warning: Data URL format seems incorrect: {data_url[:30]}...")
                    
                return data_url
        except Exception as e:
            print(f"Error in _get_image_as_base64 for {image_url}: {str(e)}")
            return None
    
    def _format_body_photos(self, user_profile: dict) -> list:
        """Format body photos for vision model input"""
        formatted_images = []
        image_timestamps = {}
        
        # Check if we have image paths in the right format
        image_paths = user_profile.get("imagePaths", {})
        
        # Set the base URLs for Supabase storage - try authenticated endpoints for private buckets
        storage_base_url = f"{self.supabase_url}/storage/v1/object/authenticated/body-images/"
        storage_alt_url = f"{self.supabase_url}/storage/v1/object/body-images/"
        
        # Track failure information for better error handling
        failed_urls = []
        
        # Helper function to process a single image
        def process_single_image(view_type, path):
            timestamp = self._extract_timestamp(os.path.basename(path))
            
            # Try authenticated URLs since we know the bucket is private
            urls_to_try = [
                # Primary authenticated endpoint
                f"{storage_base_url}{path}",
                # Alternative endpoint (just /object/ without authenticated)
                f"{storage_alt_url}{path}",
                # If the path seems to be a complete URL already
                path if path.startswith('http') else None
            ]
            
            # Filter out None values
            urls_to_try = [url for url in urls_to_try if url]
            
            # Try each URL format
            base64_image = None
            for url in urls_to_try:
                base64_image = self._get_image_as_base64(url)
                if base64_image:
                    print(f"Successfully retrieved image for {view_type} view")
                    break
            
            if base64_image:
                if self.use_claude:
                    # Format for Claude
                    formatted_images.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": base64_image["media_type"],
                            "data": base64_image["data"]
                        }
                    })
                else:
                    # Format for OpenAI
                    formatted_images.append({
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image,
                            "detail": "high"
                        }
                    })
                
                # Store timestamp 
                if view_type not in image_timestamps:
                    image_timestamps[view_type] = []
                image_timestamps[view_type].append({"path": path, "timestamp": timestamp})
                return True
            else:
                failed_urls.append(f"{view_type}: {path}")
                return False
        
        # Process front view images
        for front_path in image_paths.get("front", []):
            process_single_image("front", front_path)
        
        # Process side view images
        for side_path in image_paths.get("side", []):
            process_single_image("side", side_path)
        
        # Process back view images
        for back_path in image_paths.get("back", []):
            process_single_image("back", back_path)
        
        # If no images were successfully processed, log detailed information
        if not formatted_images:
            print("No images were successfully processed. Skipping body analysis.")
            print(f"Failed URLs: {failed_urls}")
            
            # Add diagnostic info to user_profile for debugging
            user_profile["image_access_errors"] = {
                "failed_urls": failed_urls,
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Print information about the successfully formatted images
            print(f"Successfully processed {len(formatted_images)} images:")
            for i, img in enumerate(formatted_images):
                if self.use_claude:
                    # Print for Claude format
                    print(f"  Image {i+1}: type={img['type']}, media_type={img['source']['media_type']}")
                else:
                    # Print for OpenAI format
                    is_base64 = isinstance(img['image_url']['url'], str) and img['image_url']['url'].startswith('data:')
                    url_preview = "base64 data" if is_base64 else img['image_url']['url']
                    print(f"  Image {i+1}: type={img['type']}, format={url_preview[:30]}...")
        
        # Store timestamps in user profile for later use
        user_profile["image_timestamps"] = image_timestamps
        
        return formatted_images
    
    def format_user_profile_for_prompt(self, user_profile: dict) -> str:
        """Format the user profile into a readable markdown summary for LLM context, omitting all photo-related fields."""
        # Fields to skip (internal/technical and photo-related)
        skip_fields = {"user_id", "thread_id", "imagePaths", "image_timestamps"}
        lines = []
        for key, value in user_profile.items():
            if key in skip_fields:
                continue
            # Format lists
            if isinstance(value, list):
                value_str = ", ".join(str(v) for v in value) if value else "None"
            else:
                value_str = str(value) if value is not None else "None"
            # Prettify key
            pretty_key = key.replace("_", " ").title()
            lines.append(f"- **{pretty_key}**: {value_str}")
        return "\n".join(lines)
    
    async def _analyze_body_composition(self, user_profile):
        """Analyze body composition using a vision model and user input - now streams tokens"""
        print("Starting body composition analysis...")
        
        # Exit early if no imagePaths found
        image_paths = user_profile.get("imagePaths", {})
        if not image_paths or not any(image_paths.values()):
            print("No body photos found in profile, cannot analyze body composition")
            #yield "No body photos found for analysis."
            return  # Just return without value
            
        # Format the images for API call
        formatted_images = self._format_body_photos(user_profile)
        
        if not formatted_images:
            print("No valid body photos could be processed, skipping analysis")
            #yield "Unable to process body photos for analysis."
            return  # Just return without value
            
        print(f"Formatted {len(formatted_images)} images for analysis")
        
        # Create analysis prompt based on user information
        prompt_text = f"""[Persona]
You are a high-skilled professional fitness coach and nutritionist analyzing body composition photos of clients. Your expertise lies in providing detailed, evidence-based assessments that help clients understand their current physical state and make informed decisions about their fitness journey.

[Task]
Your task is to provide a comprehensive analysis of the provided body photos of a client, which show the person from different angles. You are given a list of images of the client, and you need to analyze each image carefully and provide your findings.

User details:
- Age: {user_profile.get('age', 'unknown')}
- Gender: {user_profile.get('gender', 'unknown')}
- Height: {user_profile.get('height', 'unknown')}
- Weight: {user_profile.get('weight', 'unknown')}
- Fitness goals: {user_profile.get('fitness_goals', 'unknown')}

[Instructions]
1. Begin your analysis with the section header "## Body Composition Analysis"

2. For each photo:
   - First, explicitly state what is visible in the image (e.g., "In the front view, I can clearly see...")
   - Describe the lighting conditions and image quality
   - Provide a detailed, objective description of what you observe
   - Explain how these observations inform your assessment

3. Then provide detailed measurements and explain your methodology:
   - **Approximate Body Fat: 15-20%** (use this format)
   - **BMI: 24.3** (if you can calculate it)
   - Any other relevant measurements
   For each measurement, explain:
   - How you arrived at this estimate
   - What visual indicators led you to this conclusion
   - The confidence level in your assessment

4. Provide an in-depth analysis of:
   - Current muscle mass regions:
     * Describe the development in each major muscle group
     * Explain what visual cues indicate muscle development
     * Compare left and right sides for symmetry
   - Posture observations:
     * Detail any postural patterns you observe
     * Explain how these might affect training
     * Note any potential compensation patterns
   - Apparent imbalances:
     * Describe any asymmetries in detail
     * Explain potential causes
     * Discuss implications for training

5. Provide a comprehensive assessment and recommendations:
   - Summarize your key findings
   - Explain how these findings relate to the client's stated goals
   - Provide detailed, actionable recommendations
   - Be brutally honest while maintaining professionalism
   - Explain the reasoning behind each recommendation

6. Formatting guidelines:
   - Use "## Body Composition Analysis" as the ONLY level 2 (H2) heading in your response
   - Use level 3 headings (###) for all subsections such as "### Measurements" or "### Muscle Assessment"
   - NEVER use level 2 headings (##) for any subsection - only for the main section title
   - Format all measurements and calculations in bold (e.g., **Body Fat: 18%**)
   - Use standard markdown for lists and formatting
   - Separate major sections with a blank line
   - Always end your response with "---" (three dashes) as a separator, followed by two blank lines before the next section title
   - Output in markdown format

Warning: Do not hallucinate. You must:
1. Start by explicitly stating what you can and cannot see in each photo
2. Only make assessments based on what is clearly visible
3. Explain your reasoning for each observation and measurement
4. If something is unclear, state that explicitly
5. If you cannot make a confident assessment, explain why
"""
        
        # Accumulated analysis text
        analysis = ""
        
        if self.use_claude:
            # For Claude API
            # Create content array for Claude
            content = [
                {"type": "text", "text": prompt_text}
            ]
            
            # Add each formatted image to the content array
            for image in formatted_images:
                content.append(image)
            
            # Add additional instructions at the end
            content.append({
                "type": "text", 
                "text": "Based on all images shown, provide a complete analysis with percentage estimates. Be professional and brutally honest in your assessment, while remaining encouraging."
            })
            
            # Print the structure of content for debugging
            debug_content = []
            for item in content:
                if item["type"] == "text":
                    debug_content.append({"type": "text", "text": item["text"][:50] + "..."})
                else:
                    debug_content.append({"type": "image", "source": {"type": "base64", "media_type": item["source"]["media_type"]}})
            
            print(f"Content has {len(content)} items: {len([i for i in content if i['type'] == 'text'])} text and {len([i for i in content if i['type'] == 'image'])} images")
            print(f"Message content structure: {json.dumps(debug_content, indent=2)}")
            
            try:
                # Format the message for Claude's Messages API
                messages = [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
                
                print(f"Sending request to Claude vision model with {len(messages)} messages")
                
                # Stream the response from Claude
                stream = self.anthropic_client.messages.create(
                    model=self.claude_model,
                    max_tokens=1024,
                    messages=messages,
                    stream=True
                )
                for chunk in stream:
                    print("CHUNK:", chunk)
                    # Skip tuples or non-content objects
                    if isinstance(chunk, tuple):
                        continue
                    if hasattr(chunk, "type") and chunk.type == "content_block_delta":
                        if hasattr(chunk, "delta") and hasattr(chunk.delta, "type") and chunk.delta.type == "text_delta":
                            token = chunk.delta.text
                            if token:
                                analysis += token
                                yield token
                
                # Print a sample of the response for debugging
                print(f"Analysis response received, {len(analysis)} chars")
                print(f"Sample of analysis: {analysis[:150]}...")
                
                # Don't return analysis - just end the generator
                return
            except Exception as e:
                print(f"Error during body composition analysis with Claude: {str(e)}")
                # Print more detailed error information
                import traceback
                print(f"Detailed error: {traceback.format_exc()}")
                #yield f"Error analyzing body photos: {str(e)}"
                return
        else:
            # For OpenAI API (existing code)
            # Create a content array for the message
            content = [
                {"type": "text", "text": prompt_text}
            ]
            
            # Add each formatted image to the content array
            for image in formatted_images:
                content.append(image)  # Each image is already in {"type": "image_url", "image_url": {"url": ...}} format
            
            # Add additional instructions at the end
            content.append({
                "type": "text", 
                "text": "Based on all images shown, provide a complete analysis with percentage estimates. Be professional and brutally honest in your assessment, while remaining encouraging."
            })
            
            # Make sure the content format is correct for a multimodal message
            print(f"Content has {len(content)} items: {len([i for i in content if i['type'] == 'text'])} text and {len([i for i in content if i['type'] == 'image_url'])} images")
            
            # Print the structure of content for debugging (without the actual image data)
            debug_content = []
            for item in content:
                if item['type'] == 'text':
                    debug_content.append({'type': 'text', 'text': item['text'][:50] + '...'})
                else:
                    url_preview = "data:image..." if isinstance(item['image_url']['url'], str) and item['image_url']['url'].startswith('data:') else str(item['image_url']['url'])[:30] + '...'
                    debug_content.append({'type': 'image_url', 'image_url': {'url': url_preview}})
            
            print(f"Message content structure: {json.dumps(debug_content, indent=2)}")
            
            try:
                # Use the vision model initialized in __init__
                messages = [HumanMessage(content=content)]
                print(f"Sending request to vision model with {len(messages)} messages")
                
                # Stream the response from OpenAI
                async for chunk in self.vision_model.astream(messages):
                    token = chunk.content
                    if token:
                        analysis += token
                        yield token
                
                # Print a sample of the response for debugging
                print(f"Analysis response received, {len(analysis)} chars")
                print(f"Sample of analysis: {analysis[:150]}...")
                
                # Don't return analysis - just end the generator
                return
            except Exception as e:
                print(f"Error during body analysis streaming: {str(e)}")
                # Print more detailed error information
                import traceback
                print(f"Detailed error: {traceback.format_exc()}")
                #yield f"Error analyzing body photos: {str(e)}"
                return

    async def stream(self, state: WorkoutState) -> AsyncGenerator[str, None]:
        """Stream the profile analysis"""
        # Get stream writer for emitting events
        writer = get_stream_writer()
        
        # Emit start step
        writer({"type": "step", "content": "Starting profile analysis..."})
        
        if not state["user_profile"]:
            state["user_profile"] = UserProfile().dict()
        
        # Check if we have body photos for analysis
        has_body_photos = False
        if isinstance(state["user_profile"], dict):
            has_body_photos = "imagePaths" in state["user_profile"] and any(state["user_profile"].get("imagePaths", {}).values())
        
        # Check if we have previous profile data
        previous_profile_assessment = None
        previous_body_analysis = None
        if state.get("previous_sections"):
            previous_profile_assessment = state["previous_sections"].get("profile_assessment", "")
            previous_body_analysis = state["previous_sections"].get("body_analysis", "")
        
        # Emit progress event
        writer({"type": "step", "content": "Analyzing user profile data..."})
        
        # Check if body analysis is already in the state (from streaming endpoint)
        body_analysis = state.get("body_analysis")
        
        if has_body_photos and not body_analysis:
            # If we have body photos but no analysis yet, perform and stream it
            # Emit body analysis step
            writer({"type": "step", "content": "Analyzing body composition from photos..."})
            
            # Stream the body composition analysis token by token
            body_analysis_content = ""
            async for token in self._analyze_body_composition(state["user_profile"]):
                if token:
                    body_analysis_content += token
                    # Stream each token
                    writer({"type": "profile", "content": token})
                    yield token
            
            # Store the complete body analysis in the state
            if body_analysis_content and len(body_analysis_content) > 50:
                state["body_analysis"] = body_analysis_content
                body_analysis = body_analysis_content
                
                # Signal completion
                writer({"type": "step", "content": "Body analysis complete. Generating profile assessment..."})
        elif body_analysis and has_body_photos:
            # Body analysis was already performed in the endpoint
            writer({"type": "step", "content": "Using previously generated body analysis for profile assessment..."})
            # We don't need to yield it again since it was already streamed from the endpoint
        
        # Prepare prompts with body analysis and previous data
        profile_prompt = f"""[Persona]  
You are an elite-veteran fitness profile analyst and sports-nutrition strategist. Your hallmark is rigorous, transparent reasoning: you show clients not just *what* you see, but *why* it matters and *how* you arrived there, all in casual natural language.

[Context]  
Two system messages will follow:  

• **CURRENT CLIENT PROFILE** – JSON (age, gender, height, weight, goals, activity level, lifestyle notes, etc.)  
• **CURRENT BODY ANALYSIS** – concise markdown summary from the Body-Composition Agent  

Assume missing fields are unavailable. Do **not** quote these messages verbatim.

[Mission]  
Produce a comprehensive, professional report that:  
1. Paints a clear lists tyle user profile overview for future reference.
2. Explains, step-by-step, how that status aligns—or clashes—with stated goals.  
3. Sets realistic timelines and tightly focused priorities for the next coaching block.  
4. Makes your reasoning explicit: whenever you draw a conclusion, briefly cite the specific data or visual cue that led you there.  
If a detail is uncertain, label it **Indeterminate**—never guess.

[Report Layout] — one H2 + six H3 sections  

## Profile Assessment               ← only H2  

### Current Fitness Snapshot  
1–2 medium paragraphs covering body-comp metrics, vitals (if provided), and functional capacity. After each numeric value, add a parenthetical *confidence tag*—**High**, **Moderate**, or **Low**. Bold every number; mark missing ones **N/A**.

### Movement & Posture  
Bullets summarising key observations: spinal alignment, limb symmetry, mobility restrictions. For each item, append “(Reason: …)” to show the visual or metric that triggered the call-out. Tag < 75 %-certain items as **Possible** or **Indeterminate**.

### Goal Alignment & Readiness  
For every stated goal, create a two-column mini-table: **Supports Progress** | **Potential Roadblock**. Under each column give one factor, then add a one-line rationale.

### Realistic Timeframes  
For each primary goal, give a **Best-Case ETA** and **Likely ETA**, plus the reasoning (e.g., “based on current weekly body-fat change of ~0.4 %”). If prediction is unsafe, write **Indeterminate**.

### Key Focus Areas & Action Steps  
List every relevant focus area—cover all that matter, no artificial limit. Start each bullet with a relevant icon (e.g., ✅ or ⚠️) 
• *Why it matters* – one sentence tied to physiology or behaviour.  
• *Immediate tactic* – a concrete action that can start this week.  
• *Metric to watch* – the single number or cue that proves progress.

[Style Guide]  
• Professional, direct and honest.  
• Reveal your reasoning in casual natural language 
• Bold all measurements.  
• One blank line between major sections.  
• Use markdown tables wherever tabular data is indicated; align columns with pipes (`|`).
• Explain all your reasoning in a natural, conversational tone—use brief parenthetical clarifiers where helpful, but keep it client-friendly.  
• Mix concise bullets with occasional fuller sentences; stay informative without padding.  
• Speak directly to the client—use **you / your** instead of third-person wording.
• Output in markdown format and end with three dashes (`---`) on its own line—nothing after.  

---
        """

        user_profile = state["user_profile"]

        writer({"type": "step", "content": "Generating profile assessment..."})

        # Initialize message list for profile assessment
        messages = []

        messages.append(SystemMessage(content=profile_prompt))
        messages.append(SystemMessage(content=f"The user's Profile:\n\n{user_profile}"))
        if body_analysis:
            messages.append(SystemMessage(content=f"The user's Body Composition Analysis:\n\n{body_analysis}"))

        if previous_profile_assessment:
            messages.append(SystemMessage(content=f"Previous Profile Assessment:\n\n{previous_profile_assessment}"))

        if previous_body_analysis:
            messages.append(SystemMessage(content=f"Previous Body Composition Analysis:\n{previous_body_analysis}"))
            
        
        # Stream the profile assessment using LLM
        writer({"type": "profile", "content": "\n"})
        yield "\n"
        async for chunk in self.llm.astream(messages):
            if chunk.content:
                # Emit profile content for custom stream
                writer({"type": "profile", "content": chunk.content})
                yield chunk.content
                
        # If we have a body type from the analysis, store it in user_profile
        if body_analysis and "body_type" not in state["user_profile"] or not state["user_profile"]["body_type"]:
            # Try to extract body_type from the body analysis
            if "body type:" in body_analysis.lower():
                body_type_part = body_analysis.lower().split("body type:")[1].strip()
                body_type = body_type_part.split("\n")[0].strip()
                state["user_profile"]["body_type"] = body_type
            elif "somatotype:" in body_analysis.lower():
                body_type_part = body_analysis.lower().split("somatotype:")[1].strip()
                body_type = body_type_part.split("\n")[0].strip()
                state["user_profile"]["body_type"] = body_type

        # Signal completion
        writer({"type": "step", "content": "Profile assessment completed."})


    def __call__(self, state: WorkoutState) -> WorkoutState:
        """Update or validate the user profile, preserving the original dictionary structure"""
        if not state["user_profile"]:
            state["user_profile"] = UserProfile().dict()
        
        # Check if we have body photos for analysis
        has_body_photos = False
        if isinstance(state["user_profile"], dict):
            has_body_photos = "imagePaths" in state["user_profile"] and any(state["user_profile"].get("imagePaths", {}).values())
        
        # Get previous data if available
        previous_profile_assessment = None
        previous_body_analysis = None
        if state.get("previous_sections"):
            previous_profile_assessment = state["previous_sections"].get("profile_assessment", "")
            previous_body_analysis = state["previous_sections"].get("body_analysis", "")
        
        # Check if we have body analysis data in the state
        body_analysis = state.get("body_analysis")
    
        # Create profile analysis with or without body analysis
        profile_prompt = f"""[Persona]  
You are an elite-veteran fitness profile analyst and sports-nutrition strategist. Your hallmark is rigorous, transparent reasoning: you show clients not just *what* you see, but *why* it matters and *how* you arrived there, all in casual natural language.

[Context]  
Two system messages will follow:  

• **CURRENT CLIENT PROFILE** – JSON (age, gender, height, weight, goals, activity level, lifestyle notes, etc.)  
• **CURRENT BODY ANALYSIS** – concise markdown summary from the Body-Composition Agent  

Assume missing fields are unavailable. Do **not** quote these messages verbatim.

[Mission]  
Produce a comprehensive, professional report that:  
1. Paints a clear lists tyle user profile overview for future reference.
2. Explains, step-by-step, how that status aligns—or clashes—with stated goals.  
3. Sets realistic timelines and tightly focused priorities for the next coaching block.  
4. Makes your reasoning explicit: whenever you draw a conclusion, briefly cite the specific data or visual cue that led you there.  
If a detail is uncertain, label it **Indeterminate**—never guess.

[Report Layout] — one H2 + six H3 sections  

## Profile Assessment               ← only H2  

### Current Fitness Snapshot  
1–2 medium paragraphs covering body-comp metrics, vitals (if provided), and functional capacity. After each numeric value, add a parenthetical *confidence tag*—**High**, **Moderate**, or **Low**. Bold every number; mark missing ones **N/A**.

### Movement & Posture  
Bullets summarising key observations: spinal alignment, limb symmetry, mobility restrictions. For each item, append “(Reason: …)” to show the visual or metric that triggered the call-out. Tag < 75 %-certain items as **Possible** or **Indeterminate**.

### Goal Alignment & Readiness  
For every stated goal, create a two-column mini-table: **Supports Progress** | **Potential Roadblock**. Under each column give one factor, then add a one-line rationale.

### Realistic Timeframes  
For each primary goal, give a **Best-Case ETA** and **Likely ETA**, plus the reasoning (e.g., “based on current weekly body-fat change of ~0.4 %”). If prediction is unsafe, write **Indeterminate**.

### Key Focus Areas & Action Steps  
List every relevant focus area—cover all that matter, no artificial limit. Start each bullet with a relevant icon (e.g., ✅ or ⚠️) 
• *Why it matters* – one sentence tied to physiology or behaviour.  
• *Immediate tactic* – a concrete action that can start this week.  
• *Metric to watch* – the single number or cue that proves progress.

[Style Guide]  
• Professional, direct and honest.  
• Reveal your reasoning in casual natural language 
• Bold all measurements.  
• One blank line between major sections.  
• Use markdown tables wherever tabular data is indicated; align columns with pipes (`|`).
• Explain all your reasoning in a natural, conversational tone—use brief parenthetical clarifiers where helpful, but keep it client-friendly.  
• Mix concise bullets with occasional fuller sentences; stay informative without padding.  
• Speak directly to the client—use **you / your** instead of third-person wording.  
• Output in markdown format and end with three dashes (`---`) on its own line—nothing after.
        """

        user_profile = state["user_profile"]


        # Initialize message list for profile assessment
        messages = []
        print(f" ==================== previous_profile_assessment: {previous_profile_assessment}")
        print(f" ==================== previous_body_analysis: {previous_body_analysis}")
        print(f" ==================== state: {state}")
        print(f" ==================== has complete previous overview: {state.get('previous_complete_response', '')}")
        print(f" ==================== previous_sections: {state.get('previous_sections', '')}")

        formatted_profile = self.format_user_profile_for_prompt(user_profile)
        messages.append(SystemMessage(content=profile_prompt))
        messages.append(HumanMessage(content=f"My Profile:\n\n{formatted_profile}"))
        if body_analysis:
            messages.append(HumanMessage(content=f"My Body Composition Analysis:\n\n{body_analysis}"))

        if previous_profile_assessment:
            messages.append(SystemMessage(content=f"Previous Profile Assessment:\n\n{previous_profile_assessment}"))

        if previous_body_analysis:
            messages.append(SystemMessage(content=f"Previous Body Composition Analysis:\n{previous_body_analysis}"))
        
        # Use synchronous invoke for the __call__ method
        response = self.llm.invoke(messages)
        
        # Store the response in user_profile for downstream agents
        state["user_profile_data"] = state["user_profile"]
        state["structured_user_profile"] = state["user_profile"]  # Keep original data
        
        # Send structured response
        state["user_profile"] = response.content
        
        return state

class DietaryAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
    
    async def stream(self, state: WorkoutState) -> AsyncGenerator[str, None]:
        """Stream dietary recommendations"""
        # Get stream writer for emitting events
        writer = get_stream_writer()
        
        # Emit start step
        writer({"type": "step", "content": "Starting dietary plan generation..."})
        
        # Check if we have previous dietary plan and profile
        previous_dietary_plan = None
        previous_user_profile = None
        
        if state.get("previous_sections"):
            previous_dietary_plan = state["previous_sections"].get("dietary_plan", "")
        
        # Get previous structured profile if available
        if state.get("structured_user_profile") and state.get("previous_complete_response"):
            previous_user_profile = json.dumps(state.get("structured_user_profile"))
        
        user_profile = state["user_profile"]
        body_analysis = state.get("body_analysis")

        
        diet_prompt = f"""ROLE
You are a world-class sports-nutrition strategist who turns body-composition data and training goals into practical, appetising meal strategies. Your trademark: crystal-clear reasoning in everyday language.

⸻

WORKING FILES

Immediately after this instruction you will receive up to three system messages (do not quote them verbatim):
	1.	CURRENT PROFILE ASSESSMENT – markdown overview of age, lifestyle, activity level, goals, etc.
	2.	CURRENT BODY ANALYSIS – concise markdown digest from the Body-Composition Agent.
	3.	PREVIOUS DIETARY PLAN – markdown summary of the last plan you produced (may be absent).

⸻

MANDATE
	1.	Diagnose – Extract calorie and macro needs from today’s body metrics, goals and activity level.
	2.	Design – Provide a menu of balanced meal options that hit those targets and respect lifestyle constraints.
	3.	Evolve – If a prior plan exists, highlight key adjustments that promote progression.
	4.	Educate – Explain why each component matters for performance, recovery and adherence.

⸻

OUTPUT LAYOUT

Produce a markdown document with one H2 and multiple H3 sections in the order below.

## Dietary Plan ← single H2

(Always start the file with two blank lines, then this line.)

### Daily Targets
	•	Total Calories: XXXX kcal
	•	Macros: Protein = XXX g, Carbs = XXX g, Fat = XXX g
	•	Method: briefly state formula or evidence (e.g., “Mifflin-St Jeor with 10 % deficit for fat-loss phase”).

### Food Guidance (≈ 50-150 words)
Write a concise narrative  that:  
• Lists **foods to prioritise** and why (e.g., oily fish → EPA/DHA for anti-inflammation).  
• Lists **foods / habits to limit or avoid** and why (e.g., excess refined sugars → glycaemic volatility).  
• Highlights smart swaps, sourcing tips and portion-control cues to keep the plan practical.  
*Do not use bullets here—compose a flowing paragraph or two that reads like a mini-article.*

### Meal Library

Break the day into sub-sections. Each option = bold name, then one-sentence description, then italic macros:

Breakfast – provide ≥8 options
	•	Greek-Yoghurt Parfait — layered yoghurt, berries, granola. Macros: ~30 g P, 45 g C, 8 g F

Lunch – ≥8 options

…

Dinner – ≥8 options

…

Snacks – 5–7 options

…

### Hydration & Electrolytes

Bullet key fluid and electrolyte goals (e.g., “3 L water daily; add 500 mg sodium on double-session days”).

### Nutritional Insights
	•	Goal Support: how the plan aligns with hypertrophy / fat loss / endurance.
	•	Adaptation Rules: how to scale portions on light vs. heavy training days.
	•	Recovery Nutrients: spotlight protein timing, omega-3s, antioxidants.
	•	Prep & Adherence Tips: batch-cook hacks, travel strategies, flavour swaps.

### Plan Evolution (omit if no previous plan)

Table with two columns → | Previous Plan | Current Adjustment | — one row per major change.

⸻

STYLE & FORMAT RULES
	•	Only one H2 (“## Dietary Plan”); all others are H3 (“### …”) or lower.
	•	Bold every food item; italicise macros in the exact pattern _Macros: 30 g P, 45 g C, 15 g F_.
	•	Insert horizontal rules (---) between major sections.
	•	Use up-to-date metric or imperial units consistent with profile (default to metric if unclear).
	•	End the document with a final ---, then two blank lines.

⸻

SAFETY & DATA INTEGRITY
	•	Never invent client data; flag missing items as Indeterminate.
	•	Avoid medical claims; stay evidence-based and practical.
	•	Encourage but never shame; acknowledge real-world adherence challenges.

⸻
"""

        messages = []

        messages.append(SystemMessage(content=diet_prompt))
        messages.append(SystemMessage(content=f"The user's Profile Assessment:\n\n{user_profile}"))
        if body_analysis:
            messages.append(SystemMessage(content=f"The user's Body Composition Analysis:\n\n{body_analysis}"))

        if previous_dietary_plan:
            messages.append(SystemMessage(content=f"Previous Generated Dietary Plan:\n\n{previous_dietary_plan}"))
        
        state["dietary_state"].is_streaming = True
        state["dietary_state"].last_update = datetime.now().isoformat()
        
        # Emit progress update
        writer({"type": "step", "content": "Analyzing nutritional needs and preferences..."})
        
        # Stream the dietary recommendations
        async for chunk in self.llm.astream(messages):
            if chunk.content:
                # Emit dietary content for custom stream
                writer({"type": "dietary", "content": chunk.content})
                yield chunk.content
        
        state["dietary_state"].is_streaming = False
        
        # Signal completion
        writer({"type": "step", "content": "Dietary plan completed."})

    def __call__(self, state: WorkoutState) -> WorkoutState:
        """Generate a dietary plan based on user profile"""
        # Check if we have previous dietary plan and profile
        previous_dietary_plan = None
        previous_user_profile = None
        
        if state.get("previous_sections"):
            previous_dietary_plan = state["previous_sections"].get("dietary_plan", "")
        
        # Get previous structured profile if available
        if state.get("structured_user_profile") and state.get("previous_complete_response"):
            previous_user_profile = json.dumps(state.get("structured_user_profile"))
        
        user_profile = state["user_profile"]
        body_analysis = state.get("body_analysis")

        
        diet_prompt = f"""ROLE
You are a world-class sports-nutrition strategist who turns body-composition data and training goals into practical, appetising meal strategies. Your trademark: crystal-clear reasoning in everyday language.

⸻

WORKING FILES

Immediately after this instruction you will receive up to three system messages (do not quote them verbatim):
	1.	CURRENT PROFILE ASSESSMENT – markdown overview of age, lifestyle, activity level, goals, etc.
	2.	CURRENT BODY ANALYSIS – concise markdown digest from the Body-Composition Agent.
	3.	PREVIOUS DIETARY PLAN – markdown summary of the last plan you produced (may be absent).

⸻

MANDATE
	1.	Diagnose – Extract calorie and macro needs from today’s body metrics, goals and activity level.
	2.	Design – Provide a menu of balanced meal options that hit those targets and respect lifestyle constraints.
	3.	Evolve – If a prior plan exists, highlight key adjustments that promote progression.
	4.	Educate – Explain why each component matters for performance, recovery and adherence.

⸻

OUTPUT LAYOUT

Produce a markdown document with one H2 and multiple H3 sections in the order below.

## Dietary Plan ← single H2

(Always start the file with two blank lines, then this line.)

### Daily Targets
	•	Total Calories: XXXX kcal
	•	Macros: Protein = XXX g, Carbs = XXX g, Fat = XXX g
	•	Method: briefly state formula or evidence (e.g., “Mifflin-St Jeor with 10 % deficit for fat-loss phase”).

### Food Guidance (≈ 50-150 words)
Write a concise narrative  that:  
• Lists **foods to prioritise** and why (e.g., oily fish → EPA/DHA for anti-inflammation).  
• Lists **foods / habits to limit or avoid** and why (e.g., excess refined sugars → glycaemic volatility).  
• Highlights smart swaps, sourcing tips and portion-control cues to keep the plan practical.  
*Do not use bullets here—compose a flowing paragraph or two that reads like a mini-article.*

### Meal Library

Break the day into sub-sections. Each option = bold name, then one-sentence description, then italic macros:

Breakfast – provide ≥8 options
	•	Greek-Yoghurt Parfait — layered yoghurt, berries, granola. Macros: ~30 g P, 45 g C, 8 g F

Lunch – ≥8 options

…

Dinner – ≥8 options

…

Snacks – 5–7 options

…

### Hydration & Electrolytes

Bullet key fluid and electrolyte goals (e.g., “3 L water daily; add 500 mg sodium on double-session days”).

### Nutritional Insights
	•	Goal Support: how the plan aligns with hypertrophy / fat loss / endurance.
	•	Adaptation Rules: how to scale portions on light vs. heavy training days.
	•	Recovery Nutrients: spotlight protein timing, omega-3s, antioxidants.
	•	Prep & Adherence Tips: batch-cook hacks, travel strategies, flavour swaps.

### Plan Evolution (omit if no previous plan)

Table with two columns → | Previous Plan | Current Adjustment | — one row per major change.

⸻

STYLE & FORMAT RULES
	•	Only one H2 (“## Dietary Plan”); all others are H3 (“### …”) or lower.
	•	Bold every food item; italicise macros in the exact pattern _Macros: 30 g P, 45 g C, 15 g F_.
	•	Insert horizontal rules (---) between major sections.
	•	Use up-to-date metric or imperial units consistent with profile (default to metric if unclear).
	•	End the document with a final ---, then two blank lines.

⸻

SAFETY & DATA INTEGRITY
	•	Never invent client data; flag missing items as Indeterminate.
	•	Avoid medical claims; stay evidence-based and practical.
	•	Encourage but never shame; acknowledge real-world adherence challenges.

⸻
"""

        messages = []

        messages.append(SystemMessage(content=diet_prompt))
        messages.append(HumanMessage(content=f"My Profile Assessment:\n\n{user_profile}"))
        if body_analysis:
            messages.append(HumanMessage(content=f"My Body Composition Analysis:\n\n{body_analysis}"))

        if previous_dietary_plan:
            messages.append(SystemMessage(content=f"Previous Generated Dietary Plan:\n\n{previous_dietary_plan}"))

        # print("\n\n==========================================")
        # print(f"previous_dietary_plan: {previous_dietary_plan}")
        # print(f"previous_sections: {state['previous_sections']}")
        # #print(f"previous_sections content raw: {state['previous_sections'].get('dietary_plan')}")
        # print(f"previous_complete_response: {state.get('previous_complete_response')}")
        # print("==========================================\n\n")
        
        response = self.llm.invoke(messages)
        state["dietary_state"].content = response.content
        state["dietary_state"].last_update = datetime.now().isoformat()
        return state

class FitnessAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
    
    async def stream(self, state: WorkoutState) -> AsyncGenerator[str, None]:
        """Stream fitness recommendations"""
        # Get stream writer for emitting events
        writer = get_stream_writer()
        
        # Emit start step
        writer({"type": "step", "content": "Starting fitness plan generation..."})
        
        # Check if we have previous fitness plan and profile
        previous_fitness_plan = None
        previous_user_profile = None
        user_profile = state["user_profile"]
        body_analysis = state.get("body_analysis")
        
        if state.get("previous_sections"):
            previous_fitness_plan = state["previous_sections"].get("fitness_plan", "")
        
        # Get previous structured profile if available
        if state.get("structured_user_profile") and state.get("previous_complete_response"):
            previous_user_profile = json.dumps(state.get("structured_user_profile"))
        
        fitness_prompt = f"""[ROLE & EXPERTISE]
You are an elite Personal Fitness Strategist and guide with 20+ years coaching individuals as personalized as possible. Your role is to analyze their data and create a personalised strategic framework that speaks directly to them.

You are responsible for creating personalised intelligent architecture tailored specifically to their profile, body, lifestyle, and goals.

Your expertise spans:
- Periodization theory and training systems design
- Movement quality assessment and corrective strategies
- Physiological adaptation mechanisms and recovery science
- Performance psychology and behavior change
- Evidence-based training methodologies

[INPUT]
You will receive three analytical documents about this specific user:

- Their profile assessment
- Their body composition analysis
- Their dietary guide 

[TASK] 
Synthesize their data to create a personalised strategic roadmap that addresses their specific needs, challenges, and opportunities.
Ecosystem Integration

Important side note: This user has access to dedicated workout creation systems that will handle specific programming (exercise selection, sets/reps, periodization schedules). 
Your task is to provide the strategic foundation and theoretical framework tailored specifically to them.

Deliver a comprehensive personalized strategic analysis as a single markdown document.

[Focus Areas & Deliverables]
Personalized Strategic Analysis (Not Generic Prescription)
Focus on:

✅ Training methodologies specifically suited to their profile
✅ Periodization approaches that fit their lifestyle and goals
✅ Movement pattern priorities based on their assessment and movements to avoid etc
✅ Energy system strategies aligned with their objectives
✅ Recovery approaches that work with their schedule and stress levels
✅ Long-term development tailored to their training age and aspirations

Avoid:

❌ Generic workout schedules or training days
❌ Exact sets, reps, or load prescriptions
❌ Cookie-cutter exercise instructions
❌ One-size-fits-all programming calendars

[Personalized Movement & Corrective Strategy]
Analyze their specific needs:

Movement pattern deficiencies identified in their assessment
Mobility/stability imbalances specific to their body
Injury risk factors based on their history and lifestyle
Warm-up strategies that address their particular needs

[Their Physiological Adaptation Framework]
Address their specific profile:

Primary adaptation targets based on their goals and current state
Optimal stimulus-recovery ratios for their recovery capacity
Metabolic considerations based on their dietary analysis
Hormonal optimization strategies for their demographic
Sleep and stress management tailored to their lifestyle

Their Monitoring & Feedback Systems
Establish personalized tracking:

Key performance indicators relevant to their goals
Subjective wellness monitoring that fits their routine
Objective measurements appropriate for their training level
Decision trees for adjustments based on their response patterns
Personal red flags based on their risk factors

[STYLE]
Use "you," "your," and "yours" throughout
Reference specific data points from their assessments
Connect recommendations directly to their stated goals
Address their specific limitations and challenges
Acknowledge their strengths and build upon them.

[OUTPUT FORMATTING]
Begin your response with 2 blank lines, then on a new line the heading "## Fitness Plan".

- Use "## Fitness Plan" as the ONLY level 2 (H2) heading in your response
- Use level 3 headings (###) for all subsections.
- NEVER use level 2 headings (##) for any subsection - only for the main section title.
- Use horizontal rules (---) to separate major sections
- Include proper spacing between sections for readability

Always end your response with "---" (three dashes) as a separator, followed by two blank lines before the next section title.

[Data Synthesis Requirements]
Identify patterns specific to their three data sources
Highlight priority areas based on their individual assessment
Address any contradictions in their data with personalized solutions
Create clear hierarchy of focuses for their situation

Output Goal: Create a personalised fitness strategy that speaks directly to this individual—using their data to craft a personalised guide for their fitness journey like they have had never before.
        """

        messages = []

        messages.append(SystemMessage(content=fitness_prompt))
        messages.append(SystemMessage(content=f"The user's Profile Assessment:\n\n{user_profile}"))
        if body_analysis:
            messages.append(SystemMessage(content=f"The user's Body Composition Analysis:\n\n{body_analysis}"))

        if previous_fitness_plan:
            messages.append(SystemMessage(content=f"Previous Generated Fitness Plan:\n\n{previous_fitness_plan}"))

        state["fitness_state"].is_streaming = True
        state["fitness_state"].last_update = datetime.now().isoformat()
        
        # Emit progress update
        writer({"type": "step", "content": "Analyzing fitness goals and creating personalized workout plan..."})
        
        # Stream the fitness recommendations
        async for chunk in self.llm.astream(messages):
            if chunk.content:
                # Emit fitness content for custom stream
                writer({"type": "fitness", "content": chunk.content})
                yield chunk.content
        
        state["fitness_state"].is_streaming = False
        
        # Signal completion
        writer({"type": "step", "content": "Fitness plan completed."})

    def __call__(self, state: WorkoutState) -> WorkoutState:
        """Generate a fitness plan based on user profile"""
        # Check if we have previous fitness plan and profile
        previous_fitness_plan = None
        previous_user_profile = None
        user_profile = state["user_profile"]
        body_analysis = state.get("body_analysis")
        dietary_analysis = state.get("dietary_state")
        
        if state.get("previous_sections"):
            previous_fitness_plan = state["previous_sections"].get("fitness_plan", "")
        
        # Get previous structured profile if available
        if state.get("structured_user_profile") and state.get("previous_complete_response"):
            previous_user_profile = json.dumps(state.get("structured_user_profile"))
        
        fitness_prompt = f"""[ROLE & EXPERTISE]
You are an elite Personal Fitness Strategist and guide with 20+ years coaching individuals as personalized as possible. Your role is to analyze their data and create a personalised strategic framework that speaks directly to them.

You are responsible for creating personalised intelligent architecture tailored specifically to their profile, body, lifestyle, and goals.

Your expertise spans:
- Periodization theory and training systems design
- Movement quality assessment and corrective strategies
- Physiological adaptation mechanisms and recovery science
- Performance psychology and behavior change
- Evidence-based training methodologies

[INPUT]
You will receive three analytical documents about this specific user:

- Their profile assessment
- Their body composition analysis
- Their dietary guide 

[TASK] 
Synthesize their data to create a personalised strategic roadmap that addresses their specific needs, challenges, and opportunities.
Ecosystem Integration

Important side note: This user has access to dedicated workout creation systems that will handle specific programming (exercise selection, sets/reps, periodization schedules). 
Your task is to provide the strategic foundation and theoretical framework tailored specifically to them.

Deliver a comprehensive personalized strategic analysis as a single markdown document.

[Focus Areas & Deliverables]
Personalized Strategic Analysis (Not Generic Prescription)
Focus on:

✅ Training methodologies specifically suited to their profile
✅ Periodization approaches that fit their lifestyle and goals
✅ Movement pattern priorities based on their assessment and movements to avoid etc
✅ Energy system strategies aligned with their objectives
✅ Recovery approaches that work with their schedule and stress levels
✅ Long-term development tailored to their training age and aspirations

Avoid:

❌ Generic workout schedules or training days
❌ Exact sets, reps, or load prescriptions
❌ Cookie-cutter exercise instructions
❌ One-size-fits-all programming calendars

[Personalized Movement & Corrective Strategy]
Analyze their specific needs:

Movement pattern deficiencies identified in their assessment
Mobility/stability imbalances specific to their body
Injury risk factors based on their history and lifestyle
Warm-up strategies that address their particular needs

[Their Physiological Adaptation Framework]
Address their specific profile:

Primary adaptation targets based on their goals and current state
Optimal stimulus-recovery ratios for their recovery capacity
Metabolic considerations based on their dietary analysis
Hormonal optimization strategies for their demographic
Sleep and stress management tailored to their lifestyle

Their Monitoring & Feedback Systems
Establish personalized tracking:

Key performance indicators relevant to their goals
Subjective wellness monitoring that fits their routine
Objective measurements appropriate for their training level
Decision trees for adjustments based on their response patterns
Personal red flags based on their risk factors

[STYLE]
Use "you," "your," and "yours" throughout
Reference specific data points from their assessments
Connect recommendations directly to their stated goals
Address their specific limitations and challenges
Acknowledge their strengths and build upon them.

[OUTPUT FORMATTING]
Begin your response with 2 blank lines, then on a new line the heading "## Fitness Plan".

- Use "## Fitness Plan" as the ONLY level 2 (H2) heading in your response
- Use level 3 headings (###) for all subsections.
- NEVER use level 2 headings (##) for any subsection - only for the main section title.
- Use horizontal rules (---) to separate major sections
- Include proper spacing between sections for readability

Always end your response with "---" (three dashes) as a separator, followed by two blank lines before the next section title.

[Data Synthesis Requirements]
Identify patterns specific to their three data sources
Highlight priority areas based on their individual assessment
Address any contradictions in their data with personalized solutions
Create clear hierarchy of focuses for their situation

Output Goal: Create a personalised fitness strategy that speaks directly to this individual—using their data to craft a personalised guide for their fitness journey like they have had never before.
        """

        messages = []

        messages.append(SystemMessage(content=fitness_prompt))
        messages.append(HumanMessage(content=f"My Profile Assessment:\n\n{user_profile}"))
        if body_analysis:
            messages.append(HumanMessage(content=f"My Body Composition Analysis:\n\n{body_analysis}"))
        if dietary_analysis:
            messages.append(HumanMessage(content=f"My Dietary Analysis:\n\n{dietary_analysis}"))

        if previous_fitness_plan:
            messages.append(SystemMessage(content=f"Previous Generated Fitness Plan:\n\n{previous_fitness_plan}"))

        response = self.llm.invoke(messages)
        state["fitness_state"].content = response.content
        state["fitness_state"].last_update = datetime.now().isoformat()
        return state

class QueryAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
    
    async def stream(self, state: WorkoutState) -> AsyncGenerator[str, None]:
        """Stream response to query"""
        # Get stream writer for emitting events
        #writer = get_stream_writer()
        
        if not state["current_query"]:
            return
        
        # Emit start of query processing
        #writer({"type": "step", "content": f"Processing query: {state['current_query']}"})
        
        # Get previous sections if available
        #previous_sections = state.get("previous_sections", {})
        
        # Simple query handling without routing logic
        system_message = SystemMessage(
            content=(
                "You are an elite, highly experienced fitness and nutrition coach. "
                "Below is the user's complete fitness profile overview, including all relevant assessments, plans, and progress tracking."
                "Use this as the authoritative context for answering any questions. "
                "Be specific, actionable, and always reference the user's actual data and goals. "
                "If the question is about progress, compare relevant sections. "
                "If you need to clarify, ask a follow-up question."
                "\n\n"
                f"{state.get('complete_response', '')}"
            )
        )
        
        human_message = HumanMessage(
            content=state["current_query"]
        )

        messages = [system_message, human_message]
        # Emit analyzing step
        #writer({"type": "step", "content": "Analyzing query against your personalized plans..."})
        
        # Stream the response
        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield chunk.content
        
        # Signal completion
        #writer({"type": "step", "content": "Query processing completed."})

    def __call__(self, state: WorkoutState) -> WorkoutState:
        """Handle user queries about their fitness and dietary plans"""
        if not state["current_query"]:
            return state
        
        # Get previous sections if available
        previous_sections = state.get("previous_sections", {})
        
        # Simple query handling without routing logic
        query_prompt = f"""
        User Query: {state["current_query"]}
        
        Context:
        Dietary Plan: {state["dietary_state"].content}
        Fitness Plan: {state["fitness_state"].content}
        User Profile: {state["user_profile"]}
        
        {f'''Previous Data (for reference):
        Previous Profile Assessment: {previous_sections.get("profile_assessment", "")}
        Previous Dietary Plan: {previous_sections.get("dietary_plan", "")}
        Previous Fitness Plan: {previous_sections.get("fitness_plan", "")}
        ''' if previous_sections else ''}
        
        Provide a detailed, personalized response to the user's question.
        Ensure the answer is:
        - Specific to their health and fitness plan
        - Actionable
        - Based on the generated plans
        - Considers all relevant aspects of their fitness and dietary plans
        
        {f'''If the user's question relates to changes over time or progress:
        - Compare their previous and current plans
        - Highlight improvements or changes
        - Explain how their approach has evolved based on their progress
        ''' if previous_sections else ''}
        """
        
        response = self.llm.invoke(query_prompt)
        state["conversation_history"].append(
            {"role": "user", "content": state["current_query"]},
        )
        state["conversation_history"].append(
            {"role": "assistant", "content": response.content}
        )
        return state

class HeadCoachAgent:
    """Coordinates the fitness coaching process and produces the final structured output."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
    
    async def stream(self, state: WorkoutState) -> AsyncGenerator[str, None]:
        """Stream comprehensive coaching plan"""
        # Get stream writer for emitting events
        writer = get_stream_writer()
        
        # If complete_response is already generated, stream it directly
        if state.get("complete_response"):
            # Split by sections for more natural streaming
            sections = state["complete_response"].split("\n\n")
            for section in sections:
                writer({"type": "response", "content": section + "\n\n"})
                yield section + "\n\n"
            return
        
        # If not already generated, create the structured output with all components
        writer({"type": "response", "content": "# Your Personalized Fitness Plan\n\n"})
        yield "# Your Personalized Fitness Plan\n\n"
        
        # Add profile assessment (no need to add header, it's in the content)
        profile_content = state["user_profile"] if isinstance(state["user_profile"], str) else json.dumps(state["user_profile"], indent=2)
        writer({"type": "profile", "content": profile_content})
        yield profile_content
        yield "\n\n"
        
        # Include body analysis if available (no need to add header, it's in the content)
        if state.get("body_analysis") and len(state.get("body_analysis", "")) > 50:
            writer({"type": "profile", "content": state["body_analysis"]})
            yield state["body_analysis"]
            yield "\n\n"
        
        # Add dietary plan (no need to add the title since it's in the content)
        writer({"type": "dietary", "content": state["dietary_state"].content})
        yield state["dietary_state"].content
        yield "\n\n"
        
        # Add fitness plan (no need to add the title since it's in the content)
        writer({"type": "fitness", "content": state["fitness_state"].content})
        yield state["fitness_state"].content
        yield "\n\n"

        # Add progress comparison if available
        if state.get("progress_comparison"):
            writer({"type": "progress", "content": state["progress_comparison"]})
            yield state["progress_comparison"]
            yield "\n\n"
        
        # Final message
        final_message = "---\n\nYour personalized fitness and dietary plans have been created. You can now ask specific questions about your plans."
        writer({"type": "response", "content": final_message})
        yield final_message

    async def compare_responses(self, previous_overview, current_overview, user_profile):
        """Compare previous and current fitness overviews to identify progress, streaming token by token."""
        if not previous_overview or not current_overview:
            yield "\n\n---\n\n"
            yield "## Progress Tracking\n\n"
            yield "This is your first fitness assessment. Future assessments will include progress tracking."
            return
        

        comparison_prompt = f"""Role & Expertise
You are a world-class Fitness Progress Analyst and Performance Coach with 15+ years specializing in longitudinal client assessment and transformation tracking. Your expertise encompasses:

Quantitative Analysis: Biometric trends, performance metrics, and data-driven progress evaluation
Qualitative Assessment: Lifestyle adaptation, behavioral change patterns, and adherence analysis
Contextual Interpretation: Expected vs. actual progress rates, individual variation factors, and realistic timeline assessment
Motivational Psychology: Progress framing, momentum building, and sustainable behavior reinforcement
Strategic Adjustment: Identifying optimization opportunities and course correction needs

Your role is to provide comprehensive, personalized progress analysis that both celebrates achievements and provides actionable insights for continued success.
Task Objective
Conduct a detailed comparative analysis between the client's previous and current assessments, delivering a personalized progress report that:

Quantifies improvements across all measurable domains
Contextualizes progress within realistic expectations
Identifies strategic opportunities for optimization
Maintains motivation while providing honest, actionable feedback
Guides decision-making for future programming adjustments

Context Integration Protocol
You will receive three comprehensive data sets:
Data Sources:

PREVIOUS USER OVERVIEW → Complete historical snapshot: profile, body analysis, dietary plan, fitness strategy
CURRENT USER OVERVIEW → Complete current snapshot: updated profile, body analysis, dietary plan, fitness strategy
CURRENT USER PROFILE → JSON metadata: demographics, goals, activity level, lifestyle factors

Analysis Framework:

Temporal Comparison: Identify changes, trends, and progression patterns
Contextual Assessment: Evaluate progress against individual baseline and realistic expectations
Holistic Integration: Consider interactions between fitness, nutrition, lifestyle, and goal achievement
Predictive Insight: Assess trajectory and recommend strategic adjustments

Comprehensive Analysis Domains
Required Comparison Categories:

Anthropometric & Body Composition Changes

Weight fluctuations and trend analysis
Body fat percentage and muscle mass shifts
Circumference measurements and visual progress
Health markers and vital statistics


Performance & Fitness Metrics

Strength progression across movement patterns
Cardiovascular endurance improvements
Flexibility, mobility, and movement quality changes
Workout consistency and training adherence


Nutritional Progress & Dietary Adherence

Macro and micronutrient compliance trends
Eating pattern consistency and behavior changes
Energy level and dietary satisfaction improvements
Supplement adherence and effectiveness


Lifestyle Integration & Behavioral Changes

Sleep quality and recovery metrics
Stress management and lifestyle balance
Activity level outside structured exercise
Habit formation and routine establishment


Goal Achievement & Strategic Alignment

Progress toward stated objectives
Timeline adherence and expectation management
Strategy effectiveness and optimization needs
Motivation levels and psychological well-being



Output Structure & Formatting
Document Framework:
[2 blank lines]
## Progress Tracking

### Executive Progress Summary
### Body Composition & Physical Changes  
### Performance & Training Progression
### Nutritional Adherence & Dietary Evolution
### Lifestyle Integration & Behavioral Wins
### Strategic Insights & Forward Recommendations
---
Content Architecture Requirements:
For Each Analysis Section:

Quantified Improvements: Specific metrics with percentage/absolute changes
Contextual Assessment: Whether progress aligns with expected timelines
Individual Relevance: How changes relate to their specific goals and circumstances
Strategic Implications: What this means for future programming decisions

Formatting Standards:

Bold ALL quantified improvements: Weight decreased by 2.3kg (5.1 lbs)
Bold significant behavioral changes: Increased workout consistency from 2 to 5 days per week
Use H3 (###) for ALL section headings - never use H2 (##) except for main title
Use H4 (####) for subsections when additional hierarchy is needed
End with separator: Three dashes (---) on final line

Analysis Excellence Standards
Quantification Requirements:

Calculate percentage changes for all measurable metrics
Provide absolute numbers alongside relative changes
Include timeframe context (e.g., "over 8 weeks," "since last assessment")
Compare against typical progress ranges when relevant

Motivational Framework:

Celebrate wins: Acknowledge ALL positive changes, however small
Provide context: Explain why certain progress rates are realistic/impressive
Address challenges constructively: Frame setbacks as learning opportunities
Maintain momentum: End each section with forward-looking encouragement

Scientific Rigor:

Reference normal/expected progress ranges where applicable
Explain physiological reasons behind certain changes or plateaus
Distinguish between meaningful changes and normal fluctuation
Provide evidence-based context for recommendations

Communication Excellence
Tone & Style:

Professional yet personal: Expert analysis delivered with genuine care
Encouraging but honest: Celebrate progress while acknowledging areas for improvement
Action-oriented: Every insight should guide future decisions
Individually focused: Everything framed around their specific journey

Language Guidelines:

Use "you" and "your" throughout for direct personal connection
Avoid generic fitness language - make everything specific to their situation
Explain technical terms when used, but keep language accessible
Balance optimism with realistic assessment

Data Presentation:

Lead with improvements and positive changes
Present challenges as opportunities for optimization
Use specific numbers and percentages whenever possible
Provide comparison context (e.g., "above average," "typical for your demographic")

Quality Assurance Standards
Pre-Delivery Checklist:

 All quantifiable changes are calculated and presented clearly
 Progress is contextualized against realistic expectations
 Both improvements and challenges are addressed honestly
 Recommendations are specific and actionable
 Formatting follows exact specifications (H2 only for main title, H3 for sections)
 Response begins with exactly 2 blank lines before "## Progress Tracking"
 Document ends with "---" separator
 Tone remains motivational while maintaining honesty

Success Metrics:
The analysis should enable the client to:

Understand their progress clearly and objectively
Feel motivated by recognizing their achievements
Identify opportunities for continued improvement
Make informed decisions about future strategies
Maintain realistic expectations about their journey

Strategic Integration Notes
Connection to Ecosystem:

Reference how progress impacts future strategic planning
Suggest when updated assessments or strategy modifications may be beneficial
Connect progress patterns to workout programming effectiveness
Identify when specialist consultation might optimize results

Predictive Elements:

Assess trajectory toward long-term goals
Identify potential obstacles or plateaus ahead
Recommend strategic adjustments based on current progress patterns
Highlight metrics that warrant closer monitoring


Final Output Goal: Deliver a comprehensive, personalized progress analysis that accurately quantifies achievements, contextualizes progress within realistic expectations, and provides strategic insights that fuel continued success and motivation.
        """

        messages = []
        messages.append(SystemMessage(content=comparison_prompt))
        messages.append(HumanMessage(content=f"My Previous Overview:\n\n{previous_overview}"))
        messages.append(HumanMessage(content=f"My Current Overview:\n\n{current_overview}"))

        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield chunk.content

    
    def __call__(self, state: WorkoutState) -> WorkoutState:
        """Produce the final structured output with all components"""
        # Ensure we preserve the original structured data
        if "original_user_profile" in state:
            user_profile_data = state["original_user_profile"]
        elif "user_profile_data" in state:
            user_profile_data = state["user_profile_data"]
        else:
            user_profile_data = {}
            
        if isinstance(user_profile_data, dict) and "image_timestamps" in user_profile_data:
            image_timestamps = user_profile_data["image_timestamps"]
        else:
            image_timestamps = {}
        
        # Start building the complete response
        complete_response = ""
        
        # First, generate progress comparison if we have previous data
        progress_comparison = None
        try:
            # Use the complete previous response for comparison - this is more reliable than parsed sections
            if "previous_complete_response" in state and state["previous_complete_response"]:
                print("\n\n==========================================")
                print(f"Generating progress comparison")
                print(f"Previous overview length: {len(state['previous_complete_response'])}")
                print(f"State keys: {list(state.keys())}")
                
                # Build current overview for comparison
                current_overview = ""
                
                if state.get("body_analysis"):
                    current_overview += state["body_analysis"] + "\n\n"

                
                current_overview += state["user_profile"] if isinstance(state["user_profile"], str) else json.dumps(state["user_profile"], indent=2) + "\n\n"
                
                
                # Add dietary plan
                current_overview += state["dietary_state"].content + "\n\n"
                
                # Add fitness plan
                current_overview += state["fitness_state"].content + "\n\n"
                
                print(f"Current overview length: {len(current_overview)}")
                print("==========================================\n\n")
                
                # Generate progress comparison using complete previous response
                progress_comparison = self.compare_responses(
                    state["previous_complete_response"],
                    current_overview,
                    user_profile_data
                )
                
                print(f"\n\n==========================================")
                print(f"Completed progress comparison. Result length: {len(progress_comparison)}")
                print(f"==========================================\n\n")
            
            # Add progress comparison at the beginning if available
            if progress_comparison:
                print("\n\n==========================================")
                print(f"Adding progress comparison to the beginning (length: {len(progress_comparison)})")
                print("==========================================\n\n")
                complete_response += progress_comparison
                complete_response += "\n\n---\n\n"
                state["progress_comparison"] = progress_comparison
            else:
                # No previous response available
                print("\n\n==========================================")
                print("No previous response data found for comparison")
                print("==========================================\n\n")
                state["progress_comparison"] = "This is your first fitness assessment. Future assessments will include progress tracking."
                complete_response += "\n\n---\n\n"
                complete_response += "## Progress Tracking\n\n"
                complete_response += state["progress_comparison"]
                complete_response += "\n\n---\n\n"
        except Exception as e:
            print("\n\n==========================================")
            print(f"Error generating progress comparison: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            print("==========================================\n\n")
            # Continue without the comparison
            complete_response += "# Your Personalized Fitness Plan\n\n"
        
        # Now add the rest of the sections (no need to add headers, they're in the content)
        complete_response += state["user_profile"] if isinstance(state["user_profile"], str) else json.dumps(state["user_profile"], indent=2)
        
        # Include body analysis if available
        if state.get("body_analysis") and len(state.get("body_analysis", "")) > 50:
            complete_response += "\n\n"
            complete_response += state["body_analysis"]
        
        # Add dietary plan (no need to add the title since it's in the content)
        complete_response += "\n\n"
        complete_response += state["dietary_state"].content
        
        # Add fitness plan (no need to add the title since it's in the content)
        complete_response += "\n\n"
        complete_response += state["fitness_state"].content
        
        # Final message
        complete_response += "\n\n---\n\n"
        complete_response += "Your personalized fitness and dietary plans have been created. You can now ask specific questions about your plans."
        
        # Store the complete response and original user profile data
        state["complete_response"] = complete_response
        state["user_profile_data"] = user_profile_data
        
        return state 