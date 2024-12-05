from langchain_ollama import ChatOllama
from .config import openwebui_api_key, base_url, system_prompt
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from typing import Dict, List
from .vector_store import get_retriever

load_dotenv()


openai_api_key = os.getenv("OPENAI_API_KEY")
gpt4_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

generation_model = gpt4_model


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


def format_docs(docs: List[Document]) -> str:
    """
    Format the documents for display

    Args:
        docs (List[Document]): The documents to format

    Returns:
        str: The formatted documents
    """

    return "\n\n".join(doc.page_content for doc in docs)


def retrieve_and_format(
    query: str, retriever: VectorStoreRetriever
) -> Dict[str, str | List[Document]]:
    """
    Retrieve and format the best documents for a given query

    Args:
        query (str): The query to use
        retriever (VectorStoreRetriever): The retriever object to use

    Returns:
        Dict[str, str | List[Document]]: The context and chunks of the best documents
    """

    docs = retriever.invoke(query)
    return {"context": format_docs(docs), "chunks": docs}  # Store the original chunks


def get_rag_chain(
    search_type: str = "mmr",
    search_kwargs: dict = None,
    pdf_path: str = None,
    embedding_function: Embeddings = None,
):
    """
    Get a RAG chain object for a given search type, search arguments, and PDF file.
    If the vector store for the PDF file does not exist, it will be created using the given embedding function.

    Args:
        search_type (str): The search type to use
        search_kwargs (dict): The search arguments to pass to the retriever
        pdf_path (str): The path to the PDF file to use
        embedding_function (Embeddings): The embedding function to use. If the vector store does not exist, it will be created using this function, otherwise it will be loade and the embedding function should be the same as the one used to create the vector store.

    Returns:
        rag_chain: The RAG chain object
    """

    retriever = get_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
        pdf_path=pdf_path,
        embedding_function=embedding_function,
    )

    rag_chain = (
        {
            "context_and_chunks": lambda x: retrieve_and_format(x, retriever),
            "question": RunnablePassthrough(),
        }
        | RunnablePassthrough()
        | {
            "llm_response": (
                {
                    "context": lambda x: x["context_and_chunks"]["context"],
                    "input": lambda x: x["question"],
                }
                | prompt
                | generation_model
            ),
            "chunks": lambda x: x["context_and_chunks"]["chunks"],
            "question": lambda x: x["question"],
        }
    )

    return rag_chain
