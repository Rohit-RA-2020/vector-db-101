from typing import List, Dict, Any
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import uvicorn

app = FastAPI()

# Add CORS middleware to allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VectorSearchService:
    def __init__(self, 
                 qdrant_url: str, 
                 qdrant_api_key: str, 
                 collection_name: str, 
                 model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        self.client = QdrantClient(qdrant_url, api_key=qdrant_api_key)
        self.collection_name = collection_name

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode(query)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        
        return [
            {
                "score": hit.score,
                "payload": hit.payload,
                "id": hit.id
            }
            for hit in results
        ]

# Initialize vector search service
vector_search = VectorSearchService(
    qdrant_url="YOUR_QDRANT_URL",
    qdrant_api_key="YOUR_API_KEY",
    collection_name="json_vector_collection"
)

@app.get("/search")
async def search_vectors(query: str = Query(...), limit: int = 5):
    results = vector_search.search(query, limit)
    
    # Return only the URL of the first result if available
    if results and results[0].get('payload', {}).get('url'):
        return {"url": results[0]['payload']['url']}
    
    return {"url": None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)