from src.storing_processing import *
from pinecone.grpc import PineconeGRPC as Pinecone
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from PIL import Image
from openai import OpenAI

# Function to display the results for all types
def display_results(result):
    # Display text result
    if "text" in result:
        text_result = result["text"]  # Assuming this is already a dictionary
        st.write(f"**Text Result** - Score: {text_result['score']}")
        st.write(f"Content: {text_result['metadata']['content']}")
        st.write(" ")

    # Display image result
    if "image" in result:
        image_result = result["image"]  # Assuming this is already a dictionary
        st.write(f"**Image Result** - Score: {image_result['score']}")
        image_path = image_result['metadata']['path']
        if os.path.exists(image_path):
            img = Image.open(image_path)
            st.image(img, caption=f"Image - {image_path}", use_column_width=True)
        st.write(" ")

    # Display video result
    if "video" in result:
        video_result = result["video"]  # Assuming this is already a dictionary
        st.write(f"**Video Result** - Score: {video_result['score']}")
        video_path = video_result['metadata']['path']
        if os.path.exists(video_path):
            st.video(video_path)
        st.write(" ")

    # Display audio result
    if "audio" in result:
        audio_result = result["audio"]  # Assuming this is already a dictionary
        st.write(f"**Audio Result** - Score: {audio_result['score']}")
        st.write(f"Transcription: {audio_result['metadata']['transcription']}")
        audio_path = audio_result['metadata']['path']
        if os.path.exists(audio_path):
            st.audio(audio_path)
        st.write(" ")



# Function to query Pinecone with embeddings
def query_pinecone_for_all_types(query_embedding, top_k=10):
    # Initialize Pinecone with gRPC
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    
    index = pc.Index(host="https://multimodal-embeddings-eogurgi.svc.aped-4627-b74a.pinecone.io")
    
    # Ensure embedding is a NumPy array before flattening
    query_vector = np.array(query_embedding).flatten().tolist()  # Flatten the query embedding if necessary
    query_response = index.query(
        vector=query_vector,
        top_k=top_k,
        namespace="example-namespace",  # Use your namespace
        include_metadata=True  # Retrieve metadata with results
    )
    return query_response


# Function to process the query results and return the top 1 match for each type
def process_query_results(query_response):
    top_results = {"text": None, "image": None, "video": None, "audio": None}
    for result in query_response['matches']:
        result_type = result['metadata']['type']
        if top_results[result_type] is None:
            top_results[result_type] = result
        # If you want only the top 1 per type, you can break the loop early when all types are found.
        if all(v is not None for v in top_results.values()):
            break
    return top_results


# Function to generate embeddings for a text query and query Pinecone
def query_text_and_return_all_types(text):
    query_embedding = get_embedding(text)
    query_response = query_pinecone_for_all_types(query_embedding)
    results = process_query_results(query_response)
    return (results)


# Function to generate embeddings for an image query and query Pinecone
def query_image_and_return_all_types(image_path):
    encoder = encode_image(image_path)
    description = summarize_image(encoder)
    query_embedding = get_embedding(description)
    query_response = query_pinecone_for_all_types(query_embedding)
    results = process_query_results(query_response)
    return (results)

# Function to generate embeddings for a video query and query Pinecone
def query_video_and_return_all_types(video_path, frame_rate=1):
    frames = extract_frames(video_path, frame_rate)
    embeddings = [get_embedding(summarize_image(encode_image(frame))) for frame in frames]
    if embeddings:
        avg_embedding = np.mean(embeddings, axis=0)
    else:
        avg_embedding = np.zeros((1536,))  # Assuming the embedding size is 1536
    query_response = query_pinecone_for_all_types(avg_embedding)
    results = process_query_results(query_response)
    return (results)

# Function to generate embeddings for an audio query and query Pinecone
def query_audio_and_return_all_types(audio_path):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    audio_file= open(audio_path, "rb")
    transcription = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    text = transcription.text
    query_embedding = get_embedding(text)
    query_response = query_pinecone_for_all_types(query_embedding)
    results = process_query_results(query_response)
    return (results)