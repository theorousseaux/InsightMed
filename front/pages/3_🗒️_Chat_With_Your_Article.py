import streamlit as st
import requests
from front.utils import fetch_pdf_paths
import os
import pandas as pd
import time

url = "http://localhost:8000/"  # URL of the FastAPI server
pdf_names = [name.split(os.sep)[-1] for name in st.session_state.pdf_paths]

#### sidebar ####
st.sidebar.title("Documents available")

pdf_path_df = pd.DataFrame(pdf_names, columns=["PDFs"])
st.sidebar.write(pdf_path_df["PDFs"])
pdf_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])


def upload_pdf():
    if pdf_file:
        response = requests.post(url + "upload_pdf", files={"file": pdf_file})
        if response.status_code == 200:
            st.session_state.upload_success = True
            st.session_state.cache_bust = time.time()
            st.session_state.pdf_paths = fetch_pdf_paths(
                url, cache_bust=st.session_state.cache_bust
            )
        else:
            st.sidebar.error("An error occurred while uploading the file")


if pdf_file:
    upload_button = st.sidebar.button("Upload", on_click=upload_pdf)

# Afficher le message de succès s'il y a eu un upload réussi
if st.session_state.upload_success:
    st.sidebar.success("File uploaded successfully")
    # Réinitialiser le flag de succès pour le prochain rechargement
    st.session_state.upload_success = False


st.title(":spiral_note_pad: Chat with your article")

col1, col2 = st.columns(2)

with col1:
    query_container = st.container(border=True, height=350)
    with query_container:
        pdf_name = st.selectbox("Article", options=pdf_names)
        pdf_path = [path for path in st.session_state.pdf_paths if pdf_name in path][0]
        text_query = st.text_area(
            label="Retrieval Query",
            placeholder="What general conclusions are drawn about MET alterations and their impact on treatments?",
            height=150,
        )
        search_button = st.button("Search")

with col2:
    parameters_container = st.container(border=True, height=350)
    with parameters_container:
        top_k = st.number_input(label="Top K", value=5, min_value=1, max_value=50)
        st.write("Number of chunks to retrieve : ", top_k)
        search_type = st.selectbox("Search type", options=["mmr", "similarity"])
        st.selectbox("Generation model", options=["gpt-4o-mini", "llama3.1", "mistral"])

st.markdown("---")

if search_button:

    st.markdown("## Query")
    st.write(text_query)

    st.markdown("## Response")

    try:
        with st.spinner("Searching..."):

            body = {
                "chain_parameters": {
                    "search_type": search_type,
                    "top_k": top_k,
                    "pdf_path": pdf_path,
                },
                "query": text_query,
            }

            response = requests.post(url + "query_article", json=body)
            response.raise_for_status()  # This will raise an exception for HTTP errors

            result = response.json()

            st.write(result["answer"])

            with st.expander("View documents"):
                for doc in result["context"]:
                    st.markdown(doc["metadata"])
                    st.markdown(doc["page_content"])

    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while making the request: {str(e)}")
        if hasattr(e, "response") and e.response is not None:
            st.error(f"Response status code: {e.response.status_code}")
            st.error(f"Response text: {e.response.text}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
