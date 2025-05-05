"""
Supabase-based memory store for conversation history in the ask endpoint.
"""
from typing import Dict, Any, List, Optional
import os
import json
from datetime import datetime
import logging

# Reuse the Supabase client setup from memory_store
from graph.memory_store import get_supabase_client

def store_conversation(user_id: str, thread_id: str, conversation_history: List[Dict[str, Any]]) -> bool:
    """Store a conversation history in Supabase.
    
    Args:
        user_id: Unique identifier for the user
        thread_id: Unique identifier for the conversation thread
        conversation_history: List of message objects with role, content, and timestamp
        
    Returns:
        Boolean indicating success
    """
    try:
        supabase = get_supabase_client()
        
        # Check if conversation exists already
        existing = supabase.table("conversation_history") \
            .select("id") \
            .eq("thread_id", thread_id) \
            .execute()
            
        data = {
            "user_id": user_id,
            "thread_id": thread_id,
            "messages": conversation_history,
            "updated_at": datetime.now().isoformat()
        }
        
        if existing.data:
            # Update existing conversation
            result = supabase.table("conversation_history") \
                .update(data) \
                .eq("thread_id", thread_id) \
                .execute()
        else:
            # Insert new conversation
            data["created_at"] = datetime.now().isoformat()
            result = supabase.table("conversation_history") \
                .insert(data) \
                .execute()
                
        logging.info(f"Successfully stored conversation for thread {thread_id}")
        return True
        
    except Exception as e:
        logging.error(f"Error storing conversation: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def get_conversation_history(thread_id: str) -> Optional[List[Dict[str, Any]]]:
    """Get conversation history for a thread.
    
    Args:
        thread_id: Unique identifier for the conversation thread
        
    Returns:
        List of message objects or None if not found
    """
    try:
        print(f"\n\n==== RETRIEVING CONVERSATION HISTORY ====")
        print(f"Thread ID: {thread_id}")
        
        supabase = get_supabase_client()
        
        # Debug the query
        print(f"Querying conversation_history table for thread_id: {thread_id}")
        
        result = supabase.table("conversation_history") \
            .select("messages") \
            .eq("thread_id", thread_id) \
            .execute()
        
        print(f"Query result: {result}")
        print(f"Data: {result.data}")
        
        if result.data and len(result.data) > 0:
            messages = result.data[0].get("messages", [])
            print(f"Found {len(messages)} messages for thread {thread_id}")
            return messages
        
        print(f"No conversation found for thread {thread_id}")
        return []
        
    except Exception as e:
        print(f"\n\n==== ERROR RETRIEVING CONVERSATION ====")
        print(f"Error retrieving conversation history: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        print(f"====================================\n\n")
        return []
