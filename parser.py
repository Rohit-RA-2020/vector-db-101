import os
import json
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import logging

class JSONVectorStore:
    def __init__(self, 
                 qdrant_url: str, 
                 qdrant_api_key: str, 
                 collection_name: str, 
                 model_name: str = 'sentence-transformers/all-mpnet-base-v2',
                 batch_size: int = 100):
        """
        Initialize Qdrant vector store for JSON files.
        
        Args:
            qdrant_url: URL of the Qdrant instance
            qdrant_api_key: API key for Qdrant
            collection_name: Name of the collection to create/use
            model_name: Sentence transformer model to use for embeddings
            batch_size: Number of points to upsert in each batch
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Initialize model and Qdrant client
        self.model = SentenceTransformer(model_name)
        self.client = QdrantClient(qdrant_url, api_key=qdrant_api_key)
        self.collection_name = collection_name
        self.batch_size = batch_size
        
    def initialize_collection(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            if not any(c.name == self.collection_name for c in collections):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.model.get_sentence_embedding_dimension(),
                        distance=Distance.COSINE
                    )
                )
                self.logger.info(f"Created new collection: {self.collection_name}")
            else:
                self.logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            self.logger.error(f"Failed to initialize collection: {str(e)}")
            raise

    def process_json_directory(self, directory_path: str) -> None:
        """
        Process all JSON files in a directory and store their vector embeddings.
        
        Args:
            directory_path: Path to the directory containing JSON files
        """
        # Validate directory
        if not os.path.isdir(directory_path):
            raise ValueError(f"Invalid directory path: {directory_path}")
        
        # Collect all JSON files
        json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        
        # Total stats
        total_processed = 0
        total_errors = 0
        
        # Process files
        for file_name in json_files:
            file_path = os.path.join(directory_path, file_name)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    
                    # Handle both single object and list of objects
                    if isinstance(data, dict):
                        data = [data]
                    
                    # Prepare batches for processing
                    texts = []
                    payloads = []
                    
                    for item in data:
                        if 'text' in item and 'url' in item:
                            texts.append(item['text'])
                            payloads.append({
                                'url': item['url'],
                                'file_name': file_name,
                                'start': item.get('start'),
                                'end': item.get('end')
                            })
                    
                    # Process in batches
                    for i in range(0, len(texts), self.batch_size):
                        batch_texts = texts[i:i+self.batch_size]
                        batch_payloads = payloads[i:i+self.batch_size]
                        
                        # Generate embeddings for this batch
                        try:
                            embeddings = self.model.encode(batch_texts)
                            
                            # Prepare points for Qdrant
                            points = [
                                PointStruct(
                                    id=total_processed + idx + 1, 
                                    vector=embedding.tolist(), 
                                    payload=payload
                                )
                                for idx, (embedding, payload) in enumerate(zip(embeddings, batch_payloads))
                            ]
                            
                            # Upsert batch to Qdrant
                            self.client.upsert(
                                collection_name=self.collection_name,
                                wait=True,
                                points=points
                            )
                            
                            total_processed += len(points)
                            self.logger.info(f"Processed batch of {len(points)} points from {file_name}")
                            
                            # Add a small delay to prevent overwhelming the server
                            # time.sleep(0.5)
                            
                        except Exception as batch_error:
                            total_errors += len(batch_texts)
                            self.logger.error(f"Error processing batch from {file_name}: {str(batch_error)}")
                            # Continue with next batch
                            continue
                    
            except json.JSONDecodeError:
                self.logger.error(f"Invalid JSON in file: {file_name}")
            except Exception as e:
                self.logger.error(f"Error processing {file_name}: {str(e)}")
        
        self.logger.info(f"Finished processing. Total points: {total_processed}, Errors: {total_errors}")

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar texts in the collection.
        
        Args:
            query: Text to search for
            limit: Maximum number of results to return
            
        Returns:
            List of search results with scores and payloads
        """
        try:
            # Generate query embedding
            query_embedding = self.model.encode(query)
            
            # Search in Qdrant
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
            
        except Exception as e:
            self.logger.error(f"Failed to perform search: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":

    vector_store = JSONVectorStore(
        qdrant_url="YOUR_QDRANT_URL",
        qdrant_api_key="YOUR_API_KEY",
        collection_name="json_vector_collection",
        batch_size=75
    )
    
    vector_store.initialize_collection()
    
    vector_store.process_json_directory("transcripts")
    
    # Example search
    # results = vector_store.search("What is gradient descent")
    # print("Search results:", json.dumps(results, indent=2))