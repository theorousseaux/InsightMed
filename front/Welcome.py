import streamlit as st
import os
import sys
import streamlit as st
import pandas as pd
from utils import fetch_pdf_paths
import time

sys.path.append(os.getcwd())

st.set_page_config(
    page_title="InsightMed",
    page_icon=":microscope:",
)

url = "http://localhost:8000/"  # URL of the FastAPI server

# Initialiser les variables d'état
if "cache_bust" not in st.session_state:
    st.session_state.cache_bust = time.time()
if "upload_success" not in st.session_state:
    st.session_state.upload_success = False
if "pdf_paths" not in st.session_state:
    st.session_state.pdf_paths = fetch_pdf_paths(
        url, cache_bust=st.session_state.cache_bust
    )

st.title(":microscope: Welcome to **InsightMed**")

st.markdown("## How to use this application")

st.markdown(
    "This application is a simple interface to interact with the RAG model. You can choose between different functionalities:"
)

st.markdown(
    """
1. **Highlights trends**: You can see the trends of the articles.
2. **Resume generation**: if you want to generate a resume of the article, you can upload it and generate a resume.
3. **Chat with your article**: if you have a specific article you want to chat with, you can upload it and chat with it."""
)


with st.expander("Here are the pdfs available in the server:"):
    pdf_path_df = pd.DataFrame(st.session_state.pdf_paths, columns=["PDFs"])
    st.write(pdf_path_df["PDFs"].str.split(os.sep).str[-1])
