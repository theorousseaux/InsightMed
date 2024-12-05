import streamlit as st
import requests
import pandas as pd


@st.cache_data
def fetch_pdf_paths(url: str, cache_bust: float = 0):
    """
    Fetch the list of PDF paths from the FastAPI server

    Args:
        url (str): The URL of the FastAPI server
        cache_bust (float): A value to bust the cache when needed

    Returns:
        list: The list of PDF paths
    """
    return requests.get(url + "list_pdfs").json()


def create_markdown_resume(answers_df: pd.DataFrame, title) -> str:
    """
    Create a markdown resume from the answers DataFrame

    Args:
        answers_df (pd.DataFrame): The DataFrame containing the answers

    Returns:
        str: The markdown resume
    """

    markdown = ""
    markdown += f"# {title}\n\n"
    for idx, row in answers_df.iterrows():
        markdown += f"### Question {idx + 1}\n\n"
        markdown += f"**Prompt:** {row['prompts']}\n\n"
        markdown += f"**Answer:** {row['answers']}\n\n"
        markdown += "---\n\n"
    return markdown
