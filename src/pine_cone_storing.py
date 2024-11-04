from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()

def store_embeddings_in_pinecone(embeddings, metadata):
    # Initialize Pinecone with gRPC
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    # Name of the index
    index_name = 'multimodal-embeddings'

    # Get the list of all indexes
    indexes_info = pc.list_indexes()

    # Check if `indexes_info` has an attribute 'indexes' or directly accessable
    if hasattr(indexes_info, 'indexes'):
        index_exists = any(index.name == index_name for index in indexes_info.indexes)
    else:
        index_exists = any(index['name'] == index_name for index in indexes_info)

    # If the index exists, delete it
    if index_exists:
        pc.delete_index(index_name)
        print(f"Index '{index_name}' deleted.")


    # Create the index after deleting or if it doesn't exist
    pc.create_index(
        name=index_name,
        dimension=1536,  # Dimension of embeddings
        metric="cosine",  # Metric for similarity search
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

    # Connect to the index
    index = pc.Index(host=os.environ["PINECONE_HOST"])

    vectors = []
    for i, embedding in enumerate(embeddings):
        vector_id = f"vec_{i}"

        # Ensure embedding is a NumPy array and flatten it
        if isinstance(embedding, list):
            embedding = np.array(embedding)  # Convert list to NumPy array

        # Store embedding and metadata
        vectors.append((vector_id, embedding.flatten().tolist(), metadata[i]))

    # Upsert vectors into Pinecone (replace `index` with the actual Pinecone index object)
    upsert_response = index.upsert(
        vectors=vectors,
        namespace="example-namespace"  # Use namespaces to organize data if needed
    )
    print(f"Upsert response: {upsert_response}")