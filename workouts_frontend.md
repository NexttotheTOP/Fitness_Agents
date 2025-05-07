# Workout Generation API Documentation

## Endpoint: Create New Workout
`POST /workout/create`

Creates a new workout based on natural language query and user profile.

### Request Body
```typescript
interface WorkoutNLQRequest {
    user_id: string;      // Required: User identifier
    prompt: string;       // Required: Natural language description of desired workout
    thread_id?: string;   // Optional: For conversation continuity
}
```

Example:
```json
{
    "user_id": "b7bfcdfd-67e6-42e6-8ed0-d465e0e179c2",
    "prompt": "Create an intermediate leg workout focusing on strength and endurance",
    "thread_id": "12349"
}
```

### Response
Returns a state object containing the created workouts:

```typescript
interface WorkoutResponse {
    user_id: string;
    thread_id: string;
    created_workouts: Array<{
        name: string;
        description: string;
        difficulty_level?: string;      // beginner/intermediate/advanced
        estimated_duration?: string;    // e.g. "45 minutes"
        target_muscle_groups?: string[];
        equipment_required?: string[];
        exercises: Array<{
            name: string;
            sets: number;
            reps?: number;              // For strength exercises
            duration?: string;          // For cardio/endurance exercises
            notes?: string;
            details: {
                description: string;
                category: string;       // strength/cardio/flexibility/etc
                muscle_groups: string[];
                difficulty: string;
                equipment_needed: string;
            }
        }>
    }>;
}
```

Example Response:
```json
{
    "user_id": "b7bfcdfd-67e6-42e6-8ed0-d465e0e179c2",
    "thread_id": "12349",
    "created_workouts": [
        {
            "name": "Strength and Endurance Leg Workout A",
            "description": "This workout focuses on building leg strength and endurance through compound lifts...",
            "difficulty_level": "intermediate",
            "estimated_duration": "45 minutes",
            "target_muscle_groups": ["Quadriceps", "Hamstrings", "Glutes"],
            "equipment_required": ["Barbell", "Dumbbells"],
            "exercises": [
                {
                    "name": "Barbell Squats",
                    "sets": 4,
                    "reps": 8,
                    "notes": "Keep your back straight and core engaged...",
                    "details": {
                        "description": "A compound exercise targeting the quadriceps, hamstrings, and glutes.",
                        "category": "strength",
                        "muscle_groups": ["Quadriceps", "Hamstrings", "Glutes"],
                        "difficulty": "intermediate",
                        "equipment_needed": "Barbell"
                    }
                }
                // ... more exercises
            ]
        }
        // ... potentially more workout variations
    ]
}
```

### Error Responses
- `400 Bad Request`: Invalid request body
- `500 Internal Server Error`: Server-side error with error message

### Notes
1. The backend automatically fetches the user's profile data using the `user_id`
2. No need to send profile data from frontend - it's managed by backend
3. The response includes 2-3 workout variations by default
4. Each workout includes detailed exercise information with form guidance
5. Exercise can be either rep-based (strength) or duration-based (cardio)

### Example Usage (React/TypeScript)
```typescript
async function createWorkout(userId: string, prompt: string) {
    try {
        const response = await fetch('http://your-api-url/workout/create', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_id: userId,
                prompt: prompt
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to create workout');
        }
        
        const workouts = await response.json();
        return workouts;
    } catch (error) {
        console.error('Error creating workout:', error);
        throw error;
    }
}
```


