"""
Shared state between modules to avoid circular imports.
"""

# Dictionary to track Socket.IO sessions by thread_id
thread_to_sid = {}

# Socket.IO instance reference
sio = None

# Function to register the Socket.IO instance
def register_socketio(socketio_instance):
    global sio
    sio = socketio_instance
    print(f"[shared_state] Registered Socket.IO instance")

# Function to register a Socket.IO session ID for a thread
def register_sid(thread_id, sid):
    thread_to_sid[thread_id] = sid
    print(f"[shared_state] Registered Socket.IO session {sid} for thread {thread_id}")
    print(f"[shared_state] Active thread_to_sid mappings: {list(thread_to_sid.keys())}")
    
# Function to get a Socket.IO session ID for a thread
def get_sid(thread_id):
    return thread_to_sid.get(thread_id)

# Function to emit an event directly through Socket.IO
async def emit_event(event, thread_id):
    """Emit an event via Socket.IO to the client with the given thread_id."""
    try:
        if not sio:
            print(f"[shared_state] Error: No Socket.IO instance registered")
            return False
            
        # Get the Socket.IO session ID for this thread
        sid = get_sid(thread_id)
        if not sid:
            print(f"[shared_state] No SID found for thread {thread_id}")
            return False
            
        # Emit the event
        await sio.emit('model_event', event, to=sid)
        print(f"[shared_state] Emitted event {event['type']} to {sid}")
        return True
    except Exception as e:
        print(f"[shared_state] Error emitting event: {e}")
        import traceback
        traceback.print_exc()
        return False 