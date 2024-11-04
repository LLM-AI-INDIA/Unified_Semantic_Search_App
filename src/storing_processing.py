
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import cv2
from PIL import Image
from io import BytesIO
import base64
import openai
from PIL import Image
from io import BytesIO
import base64
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def summarize_image(encoded_image):
    prompt = [
        SystemMessage(content="You are a bot that is good at analyzing images"),
        HumanMessage(content=[
            {
                "type": "text",
                "text": "Describe the contents of this image."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                },
            },
        ])
    ]
    llm = ChatOpenAI(model="gpt-4o",max_tokens=1024,api_key=os.environ["OPENAI_API_KEY"])
    ai_msg = llm.invoke(prompt)
    return (ai_msg.content)


def summarize_videos(encoded_image):
    prompt = [
        SystemMessage(content="You are a bot that is good at analyzing video frames, you analysis the frame and descibed that"),
        HumanMessage(content=[
            {
                "type": "text",
                "text": "Describe the contents of this video frame. Avoid using the phrase 'this image.' Instead, focus on describing what is happening or what is visible in the frame."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                },
            },
        ])
    ]
    llm = ChatOpenAI(model="gpt-4o",max_tokens=1024,api_key=os.environ["OPENAI_API_KEY"])
    ai_msg = llm.invoke(prompt)
    return (ai_msg.content)





# Function to get embeddings from OpenAI
def get_embedding(text):
    response = client.embeddings.create(
    model="text-embedding-ada-002",
    input=text,
    encoding_format="float")

    return response.data[0].embedding

# Function to describe and embed images
def process_image(image_path):
    encoder = encode_image(image_path)  # Handle file path or PIL Image object
    description = summarize_image(encoder)  # Assume this function describes the image
    embedding = get_embedding(description)
    return embedding, {"type": "image", "path": image_path}

# Modify the encode_image function to handle both image paths and PIL Image objects
def encode_image(image):
    # If the input is a PIL Image, convert it to bytes
    if isinstance(image, Image.Image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    # If the input is a file path, handle it as before
    elif isinstance(image, str):
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    else:
        raise TypeError("Expected a PIL Image or a file path as input.")

# Function to extract and process video frames, then generate a single averaged embedding
def process_video(video_path, frame_rate=1):
    embeddings = []
    frames = extract_frames(video_path, frame_rate)

    for frame in frames:
        encoder = encode_image(frame)  # Now it accepts PIL Image objects
        description = summarize_image(encoder)  # Assume this function describes the video frame
        embedding = get_embedding(description)
        embeddings.append(embedding)

    # Compute the mean of all frame embeddings to get a single video embedding
    if embeddings:
        mean_embedding = np.mean(embeddings, axis=0)
    else:
        mean_embedding = np.zeros(1536)  # Assuming 1536 is the embedding size

    return mean_embedding, {"type": "video", "path": video_path}

# Function to extract key frames from a video
def extract_frames(video_path, frame_rate=1):
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % (fps * frame_rate) == 0:  # Extract one frame every 'frame_rate' seconds
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert to PIL Image
            frames.append(img_pil)
        frame_count += 1
    cap.release()
    return frames

# Function to process text
def process_text(text):
    embedding = get_embedding(text)  # Embed the raw text using GPT-2
    return embedding, {"type": "text", "content": text}

# Function to process audio and convert to text using OpenAI's Whisper model
def process_audio(audio_path):
    audio_file= open(audio_path, "rb")
    transcription = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    text = transcription.text
    embedding = get_embedding(text)
    return embedding, {"type": "audio", "path": audio_path, "transcription": text}


def split_paths_by_type(save_paths):
    # Initialize lists for each type
    images = save_paths.get('image', [])
    videos = save_paths.get('video', [])
    audios = save_paths.get('audio', [])
    
    return (images,videos,audios)

def main_processing(image_paths,video_paths,text_data,audio_paths):
    # Combined data to store embeddings and metadata
    combined_embeddings = []
    combined_metadata = []

    # Loop through images
    for image_path in image_paths:
        embedding, metadata = process_image(image_path)
        combined_embeddings.append(embedding)
        combined_metadata.append(metadata)

    # Loop through videos
    for video_path in video_paths:
        video_embedding, video_metadata = process_video(video_path)
        combined_embeddings.append(video_embedding)  # Append the averaged video embedding
        combined_metadata.append(video_metadata)     # Append the video metadata

    # Loop through text data
    for text in text_data:
        embedding, metadata = process_text(text)
        combined_embeddings.append(embedding)
        combined_metadata.append(metadata)

    # Loop through audio files
    for audio_path in audio_paths:
        embedding, metadata = process_audio(audio_path)
        combined_embeddings.append(embedding)
        combined_metadata.append(metadata)
        
    return combined_embeddings,combined_metadata