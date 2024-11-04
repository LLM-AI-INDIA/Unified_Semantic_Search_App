import os
from PIL import Image

# Function to save uploaded file into a specific folder
def save_uploaded_file(uploaded_file, folder):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Define the file path
    file_path = os.path.join(folder, uploaded_file.name)
    
    # Save the uploaded file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

# Function to handle multiple file uploads and save them in respective folders
def handle_multiple_file_uploads(uploaded_files):
    saved_paths = {"image": [], "video": [], "audio": [], "text": []}
    text_content = {}

    for uploaded_file in uploaded_files:
        # Get the file type
        file_type = uploaded_file.type.split('/')[0]
        file_name = uploaded_file.name
        
        # Save the file based on its type
        if file_type == "image":
            folder = "uploads/img/"
            save_path = save_uploaded_file(uploaded_file, folder)
            saved_paths["image"].append(save_path)
        
        elif file_type == "video":
            folder = "uploads/video/"
            save_path = save_uploaded_file(uploaded_file, folder)
            saved_paths["video"].append(save_path)
        
        elif file_type == "audio":
            folder = "uploads/audio/"
            save_path = save_uploaded_file(uploaded_file, folder)
            saved_paths["audio"].append(save_path)
        
        elif uploaded_file.type == "text/plain":
            # Handle text files specifically
            folder = "uploads/text/"
            save_path = save_uploaded_file(uploaded_file, folder)
            saved_paths["text"].append(save_path)
            
            # Retrieve and store the content of the text file
            content = uploaded_file.read().decode("utf-8")

    return saved_paths, content



def split_text_to_list(text):
    # Split the input text by commas and remove leading/trailing whitespaces
    result_list = [item.strip() for item in text.split(",")]
    return result_list


# Function to save uploaded file into a specific folder
def save_uploaded_query_file(uploaded_file, folder):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Define the file path
    file_path = os.path.join(folder, uploaded_file.name)

    # Save the uploaded file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path