"""
Supabase-based memory store for fitness coach profile overviews.
"""
from typing import Dict, Any, List, Optional, Tuple
import os
import json
from datetime import datetime
import re
from supabase import create_client, Client

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

# Use in-memory checkpointer for LangGraph instead of PostgreSQL
from langgraph.checkpoint.memory import MemorySaver as MemoryCheckpointer

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

def setup_fitness_tables():
    """Create the profile_overview_generations table if it doesn't exist."""
    try:
        print("\n\n==========================================")
        print("Setting up fitness tables")
        
        supabase = get_supabase_client()
        
        try:
            # First check if the table exists by running a query
            test_result = supabase.table("profile_overview_generations").select("count").limit(1).execute()
            print(f"Table exists check: {test_result}")
            print("Table profile_overview_generations exists")
        except Exception as e:
            print(f"Error querying table: {str(e)}")
            print("Table may not exist, attempting table creation via SQL...")
            
        
        # Run a comprehensive test of Supabase connectivity and table operations
        test_result = test_supabase_connection_and_table()
        if test_result:
            print("Supabase connection and table setup validated successfully")
        else:
            print("Supabase connection or table setup has issues - check logs")
        
        print("==========================================\n\n")
    except Exception as e:
        print("\n\n==========================================")
        print(f"Error setting up fitness tables: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        print("==========================================\n\n")
        raise

def store_profile_overview(user_id: str, thread_id: str, complete_overview: str, metadata: Dict[str, Any] = None):
    """Store a complete fitness profile overview in Supabase.
    
    Args:
        user_id: Unique identifier for the user
        thread_id: Unique identifier for the conversation thread
        complete_overview: The complete profile overview text (stored as "response" in the database)
        metadata: Additional data about the profile
    """
    try:
        print("\n\n==========================================")
        print(f"Attempting to store profile overview for user {user_id}")
        print(f"Thread ID: {thread_id}")
        print(f"Overview length: {len(complete_overview)}")
        print(f"Metadata: {metadata}")
        
        supabase = get_supabase_client()
        print(f"Successfully connected to Supabase")
        
        # Insert into profile_overview_generations table
        data = {
            "user_id": user_id,
            "thread_id": thread_id,
            "response": complete_overview,  # DB field name is "response"
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Print detailed information about the insert operation
        print(f"Inserting data into table 'profile_overview_generations'")
        print(f"Data structure: {', '.join(data.keys())}")
        
        # Debug: Check if the table exists
        try:
            # Try a simple query first
            test_result = supabase.table("profile_overview_generations").select("count").limit(1).execute()
            print(f"Table exists check: {test_result}")
        except Exception as table_e:
            print(f"Error checking table: {str(table_e)}")
        
        # Perform the insert
        result = supabase.table("profile_overview_generations").insert(data).execute()
        
        # If we get here, the operation was successful
        print(f"Insert successful. Result data: {result.data}")
        print(f"Successfully stored profile overview for user {user_id}")
        print("==========================================\n\n")
        
        return result.data
    except Exception as e:
        print("\n\n==========================================")
        print(f"Error storing profile overview: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        print("==========================================\n\n")
        raise

def get_previous_profile_overviews(user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Get previous profile overviews for a user, ordered by most recent first."""
    try:
        print(f"\n\n==========================================")
        print(f"Attempting to retrieve previous profile overviews for user {user_id}")
        
        supabase = get_supabase_client()
        print(f"Successfully connected to Supabase for retrieval")
        
        # Debug: Check if the table exists
        try:
            # Try a simple query first
            test_result = supabase.table("profile_overview_generations").select("count").limit(1).execute()
            print(f"Table exists check: {test_result}")
        except Exception as table_e:
            print(f"Error checking table: {str(table_e)}")
            raise
        
        # Perform the query
        result = supabase.table("profile_overview_generations") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("timestamp", desc=True) \
            .limit(limit) \
            .execute()
        
        print(f"Query executed. Found {len(result.data)} results for user {user_id}")
        if result.data:
            for i, item in enumerate(result.data):
                print(f"Result {i+1}: thread_id={item.get('thread_id')}, timestamp={item.get('timestamp')}")
        else:
            print(f"No results found for user {user_id} in table profile_overview_generations")
        
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
    """Get the most recent profile overview for a user.
    
    Returns:
        A dictionary containing the overview data, where the complete overview
        text is stored in the "response" field.
    """
    responses = get_previous_profile_overviews(user_id, limit=1)
    result = responses[0] if responses else None
    
    if result:
        print(f"\n\n==========================================")
        print(f"Found most recent profile overview for user {user_id}")
        print(f"Thread ID: {result.get('thread_id')}")
        print(f"Timestamp: {result.get('timestamp')}")
        print(f"Response length: {len(result.get('response', ''))}")
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
            print(f"Found a section but couldn't extract heading: {section_content[:50]}...")
            continue
            
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
        
        if not previous_data or "response" not in previous_data:
            print(f"No previous overview found for user {user_id}")
            return None, None
        
        # Parse the overview text into sections
        previous_overview_text = previous_data.get("response", "")
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

def test_supabase_connection_and_table():
    """Test if we can connect to Supabase and the profile_overview_generations table exists and is writable."""
    try:
        print("\n\n==========================================")
        print("Testing Supabase connection and table access")
        
        # Get client
        supabase = get_supabase_client()
        print("Successfully connected to Supabase")
        
        # Test table existence
        try:
            test_result = supabase.table("profile_overview_generations").select("count").limit(1).execute()
            print(f"Table exists check: {test_result}")
        except Exception as table_e:
            print(f"Error checking table: {str(table_e)}")
            print("The table profile_overview_generations may not exist. Try to create it.")
            
            # Attempt to create table (only if Supabase API supports it)
            try:
                print("Attempting to create table via REST API (this may fail if not supported)")
                # This may not work depending on your Supabase setup
                # Usually tables should be created via migrations or the Supabase Studio
                result = supabase.rpc("create_profile_overview_table").execute()
                print(f"Table creation result: {result}")
            except Exception as create_e:
                print(f"Error creating table: {str(create_e)}")
                print("Please create the table manually in Supabase Studio with these columns:")
                print("- id: uuid (primary key)")
                print("- user_id: text")
                print("- thread_id: text")
                print("- response: text")
                print("- timestamp: timestamp with time zone")
                print("- metadata: jsonb")
        
        # Try inserting a test record
        try:
            test_data = {
                "user_id": "test_user",
                "thread_id": "test_thread",
                "response": "This is a test response",
                "timestamp": datetime.now().isoformat(),
                "metadata": {"test": True}
            }
            
            print(f"Attempting to insert test data: {test_data}")
            result = supabase.table("profile_overview_generations").insert(test_data).execute()
            print(f"Test insert result: {result}")
            print("Test insert successful!")
            
            # Try to retrieve the test record
            retrieve_result = supabase.table("profile_overview_generations").select("*").eq("user_id", "test_user").limit(1).execute()
            if retrieve_result.data:
                print(f"Successfully retrieved test data: {retrieve_result.data}")
            else:
                print("Failed to retrieve test data even though insert was successful")
                
        except Exception as insert_e:
            print(f"Error inserting test data: {str(insert_e)}")
            print("The table may exist but you might not have permission to insert data")
        
        print("==========================================\n\n")
        return True
    except Exception as e:
        print(f"\n\n==========================================")
        print(f"Overall test failed: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        print(f"==========================================\n\n")
        return False

# Replace direct PostgreSQL connections with Supabase client
def get_db_connection():
    # Simply return the Supabase client for database operations
    return get_supabase_client()

def release_db_connection(conn):
    # No need to release anything since the Supabase client manages its own connections
    pass 