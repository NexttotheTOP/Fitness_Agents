#!/bin/bash

# Update Fitness Knowledge Pipeline
# This script runs the complete pipeline to update the fitness knowledge base

echo "=== Fitness Knowledge Update Pipeline ==="
echo "Starting at $(date)"

# Create directories
mkdir -p fitness_data
mkdir -p ./.chroma

# Step 1: Collect YouTube videos and transcripts
echo -e "\n=== Step 1: Collecting YouTube videos and transcripts ==="
python youtube_fitness_collector.py --limit 10

# Check if the collection was successful
if [ $? -ne 0 ]; then
    echo "Error collecting YouTube videos. Exiting."
    exit 1
fi

# Step 2: Convert transcripts to vector database
echo -e "\n=== Step 2: Converting transcripts to vector database ==="
python youtube_to_vector.py

# Check if the conversion was successful
if [ $? -ne 0 ]; then
    echo "Error converting transcripts to vector database. Exiting."
    exit 1
fi

# Step 3: Verify RAG system
echo -e "\n=== Step 3: Verifying RAG system ==="
python -c "
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
try:
    db = Chroma(collection_name='fitness-coach-chroma', persist_directory='./.chroma', embedding_function=OpenAIEmbeddings())
    count = db._collection.count()
    print(f'Successfully connected to vector database. Found {count} entries.')
    exit(0)
except Exception as e:
    print(f'Error connecting to vector database: {e}')
    exit(1)
"

# Check if the verification was successful
if [ $? -ne 0 ]; then
    echo "Error verifying RAG system. Exiting."
    exit 1
fi

echo -e "\n=== Pipeline completed successfully at $(date) ==="
echo "To start the fitness coach API, run: python main.py"
echo "The API will be available at http://localhost:8000" 