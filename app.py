import streamlit as st
import os
from openai import OpenAI
from PIL import Image
import os
from src.multimodel_rag import multi_rag

# Set the page configuration
st.set_page_config(layout="wide")

# Add custom CSS to set the zoom level to 90%
st.markdown(
    """
    <style>
        body {
            zoom: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Adding (css)stye to application
with open('style/final.css') as f:
    st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)

# Adding company logo
# imcol1, imcol2, imcol3, imcol4 = st.columns((4,3,5,2))
imcol1, imcol2, imcol3, imcol4 = st.columns((6,5,5,2))
with imcol2:
    st.write("## ")
    st.image('image/llmatscale-logo.png')


st.markdown("<p style='text-align: center; color: black; font-size:25px;'><span style='font-weight: bold'></span>Unified Semantic Search for Enterprise</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: blue;margin-top: -10px ;font-size:20px;'><span style='font-weight: bold'></span>Search and Explore Data Across Text, Image, Audio and Video Files</p>", unsafe_allow_html=True)
st.markdown("<hr style=height:2.5px;margin-top:0px;width:100%;background-color:gray;>",unsafe_allow_html=True)

    
#---------Side bar-------#
with st.sidebar:
    st.markdown("<p style='text-align: center; color: white; font-size:25px;'><span style='font-weight: bold; font-family: century-gothic';></span>Solutions Scope</p>", unsafe_allow_html=True)
    vAR_AI_application = st.selectbox("",['Select Application','Unified Semantic Search App'],key='application')
    
    # selected = st.selectbox("",['User',"Logout"],key='text')
    vAR_LLM_model = st.selectbox("",['LLM Models',"gpt-3.5-turbo-16k-0613","gpt-4-0314","gpt-3.5-turbo-16k","gpt-3.5-turbo-1106","gpt-4-0613","gpt-4-0314"],key='text_llmmodel')
    vAR_LLM_framework = st.selectbox("",['LLM Framework',"Langchain"],key='text_framework')

    vAR_Library = st.selectbox("",
                    ["Library Used","Streamlit","Image","Pandas","openAI"],key='text1')
    vAR_Gcp_cloud = st.selectbox("",
                    ["GCP Services Used","VM Instance","Computer Engine","Cloud Storage"],key='text2')
    st.markdown("#### ")
    href = """<form action="#">
            <input type="submit" value="Clear/Reset"/>
            </form>"""
    st.sidebar.markdown(href, unsafe_allow_html=True)
    st.markdown("# ")
    st.markdown("<p style='text-align: center; color: White; font-size:20px;'>Build & Deployed on<span style='font-weight: bold'></span></p>", unsafe_allow_html=True)
    s1,s2=st.columns((2,2))
    with s1:
        st.markdown("### ")
        st.image('image/002.png')
    with s2:    
        st.markdown("### ")
        st.image("image/oie_png.png")


if vAR_AI_application == 'Unified Semantic Search App':
    multi_rag()