"""
Supabase-based memory store for fitness coach profile overviews.
"""
from typing import Dict, Any, List, Optional, Tuple
import os
import json
from datetime import datetime
import re
from supabase import create_client, Client
from langgraph.checkpoint.memory import MemorySaver as MemoryCheckpointer

# Global Supabase client
_supabase_client = None

# Initialize Supabase client
def get_supabase_client():
    """Create and return a Supabase client."""
    global _supabase_client
    
    if _supabase_client is not None:
        return _supabase_client
        
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables must be set")
    
    _supabase_client = create_client(supabase_url, supabase_key)
    return _supabase_client

def get_postgres_checkpointer():
    """Create an in-memory checkpointer for LangGraph instead of PostgreSQL."""
    try:
        print("Setting up in-memory checkpointer for LangGraph...")
        # Create in-memory checkpointer
        checkpointer = MemoryCheckpointer()
        print("In-memory checkpointer created successfully")
        return checkpointer
    except Exception as e:
        print(f"Error setting up in-memory checkpointer: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise


def get_previous_profile_overviews(user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Get previous profile overviews for a user, ordered by most recent first."""
    try:
        print(f"\n\n==========================================")
        print(f"Attempting to retrieve previous profile overviews for user {user_id}")
        
        supabase = get_supabase_client()
        print(f"Successfully connected to Supabase for retrieval")
        
        # Perform the query
        result = supabase.table("fitness_profile_generations") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("timestamp", desc=True) \
            .limit(limit) \
            .execute()
        
        print(f"Query executed. Found {len(result.data)} results for user {user_id}")
        if result.data:
            for i, item in enumerate(result.data):
                print(f"Result {i+1}: thread_id={item.get('id')}, timestamp={item.get('timestamp')}")
        else:
            print(f"No results found for user {user_id} in table fitness_profile_generations")
        
        print(f"==========================================\n\n")
        return result.data
    except Exception as e:
        print(f"\n\n==========================================")
        print(f"Error retrieving profile overviews: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        print(f"==========================================\n\n")
        # Return empty list on error instead of raising
        return []

def get_most_recent_profile_overview(user_id: str) -> Optional[Dict[str, Any]]:
    responses = get_previous_profile_overviews(user_id, limit=1)
    result = responses[0] if responses else None
    
    if result:
        print(f"\n\n==========================================")
        print(f"Found most recent profile overview for user {user_id}")
        print(f"Thread ID: {result.get('id')}")
        print(f"Timestamp: {result.get('timestamp')}")
        print(f"Response length: {len(result.get('content', ''))}")
        print(f"==========================================\n\n")
    else:
        print(f"\n\n==========================================")
        print(f"No most recent profile overview found for user {user_id}")
        print(f"==========================================\n\n")
    
    return result

def parse_profile_overview(overview_text: str) -> Dict[str, str]:
    """Parse a complete profile overview into sections.
    
    Args:
        overview_text: The complete profile overview text
        
    Returns:
        A dictionary containing each section of the profile overview
    """
    if not overview_text:
        return {}
    
    print("\n\n==========================================")
    print(f"Parsing overview text with length: {len(overview_text)}")
    
    # Define the sections we're looking for
    sections = {
        "profile_assessment": "",
        "body_analysis": "",
        "dietary_plan": "",
        "fitness_plan": "",
        "progress_comparison": ""
    }
    
    # First, split the content by H2 headings (##)
    # The regex looks for ## followed by text, capturing the heading text
    # and everything until the next ## or end of string
    section_pattern = r'##\s*(.*?)(?=\n##|\Z)'
    
    # Find all sections with their h2 headings
    matches = re.finditer(section_pattern, overview_text, re.DOTALL)
    
    for match in matches:
        # Get the full match content
        section_content = match.group(0)
        
        # Extract the heading from the content
        heading_match = re.match(r'##\s*(.*?)(?:\n|\r\n)', section_content)
        if not heading_match:
            # Try to handle the case where the line starts with '---' followed by '##'
            alt_heading_match = re.match(r'-{3,}\s*##\s*(.*?)(?:\n|\r\n)', section_content)
            if alt_heading_match:
                heading = alt_heading_match.group(1).strip()
                print(f"Found section with heading (after dashes): '{heading}'")
                # Remove the leading dashes and whitespace from the section_content for content extraction
                section_content_clean = re.sub(r'^-{3,}\s*', '', section_content, 1)
                content = re.sub(r'^##\s*.*?(?:\n|\r\n)', '', section_content_clean, 1).strip()
            else:
                print(f"Found a section but couldn't extract heading: {section_content[:50]}...")
                continue
        else:
            heading = heading_match.group(1).strip()
            print(f"Found section with heading: '{heading}'")
            # Extract the content by removing the heading line
            content = re.sub(r'^##\s*.*?(?:\n|\r\n)', '', section_content, 1).strip()
        
        # Match the heading to our expected sections
        if re.search(r'profile\s*assessment', heading, re.IGNORECASE):
            sections["profile_assessment"] = content
            print(f"Matched Profile Assessment section, length: {len(content)}")
        
        elif re.search(r'body\s*composition\s*analysis', heading, re.IGNORECASE):
            sections["body_analysis"] = content
            print(f"Matched Body Composition Analysis section, length: {len(content)}")
        
        elif re.search(r'dietary\s*plan', heading, re.IGNORECASE):
            sections["dietary_plan"] = content
            print(f"Matched Dietary Plan section, length: {len(content)}")
        
        elif re.search(r'fitness\s*plan', heading, re.IGNORECASE):
            sections["fitness_plan"] = content
            print(f"Matched Fitness Plan section, length: {len(content)}")
        
        elif re.search(r'progress\s*(comparison|tracking)', heading, re.IGNORECASE):
            sections["progress_comparison"] = content
            print(f"Matched Progress Tracking/Comparison section, length: {len(content)}")
        
        else:
            print(f"Unknown section heading: '{heading}', content length: {len(content)}")
    
    # Print summary of found sections
    print(f"\nParsed sections summary:")
    for section_name, content in sections.items():
        if content:
            preview = content[:50].replace('\n', ' ') + '...' if len(content) > 50 else content
            print(f"- {section_name}: {len(content)} chars, preview: {preview}")
        else:
            print(f"- {section_name}: Not found")
    
    print("==========================================\n\n")
    
    return sections

def get_structured_previous_overview(user_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, str]]]:
    """Get the most recent profile overview and parse it into structured sections.
    
    Args:
        user_id: Unique identifier for the user
        
    Returns:
        A tuple containing:
        1. The raw overview data from the database (or None if not found)
        2. A dictionary with parsed sections (or None if parsing failed)
    """
    try:
        # Get most recent overview
        previous_data = get_most_recent_profile_overview(user_id)
        
        if not previous_data or "content" not in previous_data:
            print(f"No previous overview found for user {user_id}")
            return None, None
        
        # Parse the overview text into sections
        previous_overview_text = previous_data.get("content", "")
        parsed_sections = parse_profile_overview(previous_overview_text)
        
        print("\n\n==========================================")
        print(f"Retrieved and parsed previous overview for user {user_id}")
        print(f"Found {len(parsed_sections)} sections")
        for section, content in parsed_sections.items():
            if content:
                print(f"- {section}: {len(content)} chars")
        print("==========================================\n\n")
        
        return previous_data, parsed_sections
    except Exception as e:
        print("\n\n==========================================")
        print(f"Error retrieving structured previous overview: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        print("==========================================\n\n")
        return None, None
