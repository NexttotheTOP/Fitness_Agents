# Fitness Coach Professional with YouTube Knowledge

This project creates a professional fitness coach agent powered by YouTube content from fitness experts. The system uses a RAG (Retrieval Augmented Generation) approach to provide accurate, up-to-date fitness advice.


   ```

## Data Collection

The system collects fitness knowledge from YouTube in three steps:

### 1. Collect Videos and Transcripts

Use the `youtube_fitness_collector.py` script to collect videos from YouTube channels, playlists, or search results:

```bash
# Using default creators (AthleanX, Jeff Nippard, Renaissance Periodization)
python youtube_fitness_collector.py

# Specify custom channels
python youtube_fitness_collector.py --channels "https://www.youtube.com/c/athleanx" "https://www.youtube.com/c/JeffNippard"

# Specify custom playlists
python youtube_fitness_collector.py --playlists "https://www.youtube.com/playlist?list=your_playlist_id"

# Search for specific topics
python youtube_fitness_collector.py --search "weight training beginners" "proper squat form"

# Set a custom limit per source
python youtube_fitness_collector.py --limit 30
```

This will create a `fitness_data` directory with:
- `fitness_videos.json`: Metadata for collected videos
- `fitness_transcripts.json`: Transcripts from the videos

### 2. Convert Transcripts to Vector Database

Use the `youtube_to_vector.py` script to convert the collected transcripts into a vector database:

```bash
# Using default settings
python youtube_to_vector.py

# Specify custom input and output
python youtube_to_vector.py --input fitness_data/fitness_transcripts.json --collection my-fitness-db --persist_dir ./my_chroma_db
```

### 3. Run the Fitness Coach API

Start the API server:

```bash
python main.py
```

The server will run at `http://localhost:8000`.

## API Endpoints

### Create a User Profile
```http
POST /fitness/profile
```
Payload example:
```json
{
  "age": 30,
  "gender": "male",
  "height": "180cm",
  "weight": "80kg",
  "activity_level": "moderate",
  "fitness_goals": ["build muscle", "lose fat"],
  "dietary_preferences": ["high protein"],
  "health_restrictions": []
}
```

### Query the Fitness Coach
```http
POST /fitness/query
```
Payload example:
```json
{
  "thread_id": "your_thread_id_here",
  "query": "Can you suggest a good chest workout for beginners?"
}
```

### Direct Knowledge Base Query
```http
POST /fitness/rag-query
```
Payload example:
```json
{
  "query": "What does Jeff Nippard recommend for bicep training?"
}
```

### Get Session State
```http
GET /fitness/session/{thread_id}
```

### Check RAG System Status
```http
GET /fitness/rag-status
```

## Features

- **YouTube Knowledge**: Leverages content from top fitness experts on YouTube
- **Personalized Advice**: Tailors recommendations based on user profiles and goals
- **Professional Guidance**: Provides science-based workout and nutrition advice
- **Conversation Memory**: Maintains context throughout the coaching session
- **Vector Database**: Efficiently stores and retrieves relevant fitness knowledge

## Data Sources

The system uses content from these expert sources:
- AthleanX (Jeff Cavaliere)
- Jeff Nippard
- Renaissance Periodization

## Customization

You can customize the fitness coach by:
1. Adding more YouTube channels and videos to the knowledge base
2. Adjusting the embedding and retrieval parameters in `main.py`
3. Modifying the RAG integration in the query handling

## Notes

- The YouTube collection process respects rate limits and may take time for larger collections
- Make sure your OpenAI API account has sufficient credits for embedding and generation
- Consider using a more robust database for production environments