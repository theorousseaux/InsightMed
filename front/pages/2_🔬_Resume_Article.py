import streamlit as st
import pandas as pd
import requests
import os
from front.utils import create_markdown_resume, fetch_pdf_paths
import time

url = "http://localhost:8000/"
pdf_names = [name.split(os.sep)[-1] for name in st.session_state.pdf_paths]

if "analysis" not in st.session_state:
    st.session_state.analysis = None

    #### sidebar ####
st.sidebar.title("Documents available")

pdf_path_df = pd.DataFrame(pdf_names, columns=["PDFs"])
st.sidebar.write(pdf_path_df[["PDFs"]])
pdf_file = st.sidebar.file_uploader(
    "Upload PDF", type=["pdf"], accept_multiple_files=True
)


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

#### main ####

st.markdown("# :microscope: Get relevant informations from an article")

# st.markdown("## General Questions")

prompts_df = pd.read_csv("prompts.csv")
# st.write(prompts_df)

st.markdown("## Choose the article you want to analyze")

pdf_name = st.selectbox("Article", options=pdf_names)
pdf_path = [path for path in st.session_state.pdf_paths if pdf_name in path][0]

analyse_button = st.button("Analyze")
answers = []

if analyse_button:

    try:
        with st.spinner("Analyzing..."):
            payload = {
                "chain_parameters": {
                    "search_type": "similarity",
                    "top_k": 5,
                    "pdf_path": pdf_path,
                },
                "prompts": prompts_df["prompts"].tolist(),
            }
            response = requests.post(url + "resume_article_from_prompts", json=payload)
            response.raise_for_status()

            result = response.json()
            result_df = pd.DataFrame(result)
        st.session_state.analysis = result_df

    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while making the request: {str(e)}")
        if hasattr(e, "response") and e.response is not None:
            st.error(f"Response status code: {e.response.status_code}")
            st.error(f"Response text: {e.response.text}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

if st.session_state.analysis is not None:
    df = st.session_state.analysis
    summary_txt = create_markdown_resume(df, title=pdf_name)

    resume = "\n".join(
        [
            response
            for response in df["answers"]
            if response != "Information not available."
        ]
    )
    # st.markdown(resume)

    for response in df["answers"]:
        if response != "Information not available.":
            st.markdown(response)
            st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Download as CSV",
            data=df.to_csv(index=False),
            file_name=pdf_name.split(".")[-2] + "_analysis.csv",
            mime="text/csv",
        )

    with col2:
        st.download_button(
            label="Download as Markdown",
            data=summary_txt,
            file_name=pdf_name.split(".")[-2] + "_analysis.md",
            mime="text/markdown",
        )
