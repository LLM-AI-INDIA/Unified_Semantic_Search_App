import os
import streamlit as st
from dotenv import load_dotenv
from src.utilities import handle_multiple_file_uploads,split_text_to_list,save_uploaded_query_file
from src.storing_processing import split_paths_by_type,main_processing
from src.pine_cone_storing import store_embeddings_in_pinecone
from src.querying_db import query_text_and_return_all_types,display_results, query_audio_and_return_all_types,query_image_and_return_all_types,query_video_and_return_all_types
import numpy as np

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Main function for multi-modal retrieval-augmented generation (RAG)
def multi_rag():
    col1, col2, col3, col4 = st.columns((2, 3, 6, 1))
    m1, m2, m3 = st.columns((2, 5, 2))

    with col2:
        st.write('## ')
        st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Select</span></p>", unsafe_allow_html=True)
    with col3:
        vAR_selects = st.radio("",options=["Data Management with VectorDB", "Unified Semantic Search"],horizontal=True)
    if vAR_selects == "Data Management with VectorDB":
        with col2:
            st.write('## ')
            st.write('### ')
            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload Data Files</span><br><span style='font-size:18px;'>(Text, Image, Audio, and Video)</span></p>", unsafe_allow_html=True)
        with col3:
            st.write('### ')
            vAR_file = st.file_uploader(" ",type=["png", "jpg", "jpeg", "mp4", "wav", "txt"],accept_multiple_files=True)
        if vAR_file:
            vAR_paths, vAR_text = handle_multiple_file_uploads(vAR_file)
            with col3:
                st.write('## ')
                if st.button("Embed and Store Data in VectorDB"):
                    vAR_text_list = split_text_to_list(vAR_text)
                    images,videos,audios = split_paths_by_type(vAR_paths)
                    combined_embeddings,combined_metadata = main_processing(images,videos,vAR_text_list,audios)
                    store_embeddings_in_pinecone(combined_embeddings,combined_metadata)
                    with col3:
                        st.write('# ')
                        st.success("VectorDB crated and uploaded successfully!")
                    with m2:
                        # Combine embeddings and metadata together
                        st.write("### Combined Embeddings and Metadata")

                        # Iterate over the embeddings and metadata
                        for i, (embedding, metadata) in enumerate(zip(combined_embeddings, combined_metadata)):
                            st.write(f"#### Item {i+1}")
                            
                            # Display metadata
                            st.write("**Metadata:**", metadata)
                            
                            # Show a summary of embeddings (e.g., first 5 values)
                            st.write("**Embedding (first 5 values):**", np.array(embedding)[:5])
                            
                            st.write("---")
    
    elif vAR_selects == "Unified Semantic Search":
        with col2:
            st.write('### ')
            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Search Data Type</span></p>", unsafe_allow_html=True)
        with col3:
            vAR_selects = st.selectbox("", options=["Select", "Text", "Image", "Video", "Audio"])

        # Text Query
        if vAR_selects == "Text":
            with col2:
                st.write('### ')
                st.write('# ')
                st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload Text</span></p>", unsafe_allow_html=True)
            with col3:
                vAR_text = st.text_area("")
                if vAR_text:
                    if st.button("Submit"):
                        result = query_text_and_return_all_types(vAR_text)
                        with m2:
                            st.write('# ')    
                            display_results(result)

        # Image Query
        elif vAR_selects == "Image":
            with col2:
                st.write('# ')
                st.write('### ')
                st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload Images</span></p>", unsafe_allow_html=True)
            with col3:
                vAR_files = st.file_uploader(" ", type=["png", "jpeg"])
                if vAR_files:
                    if st.button("Submit"):
                        image_folder = "query_datafile/img"
                        file_path = save_uploaded_query_file(vAR_files, image_folder)
                        result = query_image_and_return_all_types(file_path)
                        with m2:
                            st.write('# ')    
                            display_results(result)

        # Video Query
        elif vAR_selects == "Video":
            with col2:
                st.write('# ')
                st.write('### ')
                st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload Videos</span></p>", unsafe_allow_html=True)
            with col3:
                vAR_files = st.file_uploader(" ", type=["mp4"])
                if vAR_files:
                    if st.button("Submit"):
                        video_folder = "query_datafile/video"
                        file_path = save_uploaded_query_file(vAR_files, video_folder)
                        result = query_video_and_return_all_types(file_path)
                        with m2:
                            st.write('# ')    
                            display_results(result)

        # Audio Query
        elif vAR_selects == "Audio":
            with col2:
                st.write('# ')
                st.write('### ')
                st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload Audios</span></p>", unsafe_allow_html=True)
            with col3:
                vAR_files = st.file_uploader(" ", type=["wav"])
                if vAR_files:
                    if st.button("Submit"):
                        audio_folder = "query_datafile/audio"
                        file_path = save_uploaded_query_file(vAR_files, audio_folder)
                        result = query_audio_and_return_all_types(file_path)
                        with m2:
                            st.write('# ')    
                            display_results(result)             