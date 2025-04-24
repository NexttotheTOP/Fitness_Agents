#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Testing Fitness Coach API...${NC}"

# Create profile and get thread_id
echo -e "${GREEN}Creating fitness profile...${NC}"
RESPONSE=$(curl -s -X POST http://localhost:8000/fitness/profile \
-H "Content-Type: application/json" \
-H "Accept: text/event-stream" \
-d '{
    "age": 30,
    "gender": "male",
    "height": 175.0,
    "weight": 75.0,
    "activity_level": "moderate",
    "fitness_goals": ["weight loss", "muscle gain"],
    "dietary_preferences": ["high protein", "low carb"],
    "health_restrictions": ["no peanuts"]
}')

# Extract thread_id from response (you might need to adjust this based on actual response format)
THREAD_ID=$(echo $RESPONSE | grep -o '"thread_id":"[^"]*' | cut -d'"' -f4)

echo "Thread ID: $THREAD_ID"

# Wait for profile creation to complete
sleep 2

# Check session state
echo -e "${GREEN}Checking session state...${NC}"
curl -s -X GET http://localhost:8000/fitness/session/$THREAD_ID | json_pp

# Test dietary query
echo -e "${GREEN}Testing dietary query...${NC}"
curl -X POST http://localhost:8000/fitness/query \
-H "Content-Type: application/json" \
-H "Accept: text/event-stream" \
-d "{
    \"thread_id\": \"$THREAD_ID\",
    \"query\": \"Can you suggest a high-protein breakfast meal plan?\"
}"

sleep 2

# Test fitness query
echo -e "${GREEN}Testing fitness query...${NC}"
curl -X POST http://localhost:8000/fitness/query \
-H "Content-Type: application/json" \
-H "Accept: text/event-stream" \
-d "{
    \"thread_id\": \"$THREAD_ID\",
    \"query\": \"What exercises should I do for leg day?\"
}"

sleep 2

# Test general query
echo -e "${GREEN}Testing general query...${NC}"
curl -X POST http://localhost:8000/fitness/query \
-H "Content-Type: application/json" \
-H "Accept: text/event-stream" \
-d "{
    \"thread_id\": \"$THREAD_ID\",
    \"query\": \"How can I track my overall progress?\"
}"

sleep 2

# Test workout variation
echo -e "${GREEN}Testing workout variation...${NC}"
curl -X POST http://localhost:8000/workout/variation \
-H "Content-Type: application/json" \
-d '{
    "requestData": {
        "name": "Basic Strength Workout",
        "description": "A fundamental strength training routine",
        "exercises": [
            {
                "name": "Squats",
                "sets": 3,
                "reps": 10,
                "notes": "Keep proper form",
                "details": {
                    "description": "Basic barbell squat",
                    "category": "strength",
                    "muscle_groups": ["legs", "core"],
                    "difficulty": "intermediate",
                    "equipment_needed": "barbell"
                }
            }
        ]
    }
}' | json_pp

echo -e "${BLUE}Testing complete!${NC}" 