from sentence_transformers import SentenceTransformer
import time

# Define the sentences
# sentences = ["The Indian Premier League (IPL) franchises have finalized their player retentions ahead of the 2025 season. Each team was permitted to retain up to six players, utilizing both direct retentions and the Right to Match (RTM) option."]

# # Initialize the model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# # Start the timer
# start_time = time.time()

# # Encode the sentences
# embeddings = model.encode(sentences)

# # Calculate the time taken
# end_time = time.time()
# encoding_time = end_time - start_time

# # Print the embeddings and the time taken
# print(embeddings[0])
# print("<-------------------------------->")
# print(f"Time taken for encoding: {encoding_time:.6f} seconds")



from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

client = QdrantClient("YOUR_QDRANT_URL", api_key="YOUR_API_KEY")

# client.create_collection(
#     collection_name="test_collection",
#     vectors_config=VectorParams(size=model.get_sentence_embedding_dimension(), distance=Distance.DOT)
# )

# operation_info = client.upsert(
#     collection_name="test_collection",
#     wait=True,
#     points=[
#         PointStruct(id=1, vector=embeddings[0], payload={"sentence": "This is an example sentence"}),
#         PointStruct(id=2, vector=embeddings[1], payload={"sentence": "India is my country"})
#     ],
# )

# print(operation_info)

query_sentence = ["Hunger Levels"]

query_embeddings = model.encode(query_sentence)

search_result = client.query_points(
    collection_name="db-ht-program",
    query=query_embeddings,
).points

for result in search_result:
    print(f"Sentence: {result.payload['content']}, Score: {result.score}")
