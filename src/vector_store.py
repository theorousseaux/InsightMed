import faiss
import hashlib
import os
from uuid import uuid4
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import embedding_folder
import re
from langchain.schema import Document
import unicodedata


def clean_scientific_text(doc: Document) -> Document:
    """
    Nettoie le texte d'un document scientifique de manière générique.

    :param doc: Un objet Document de LangChain
    :return: Un nouvel objet Document avec le texte nettoyé
    """
    text = doc.page_content

    # Normaliser les caractères Unicode
    text = unicodedata.normalize("NFKD", text)

    # Supprimer les URLs
    text = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "",
        text,
    )

    # Supprimer les DOI, numéros de volume, pages, et autres métadonnées scientifiques
    text = re.sub(r"DOI:?\s*\d+[\./]\d+", "", text)  # Supprime les DOI
    text = re.sub(r"Vol\.\s*\d+", "", text)  # Supprime les numéros de volume
    text = re.sub(
        r"\b(?:[Pp]age|[Pp]ages)\s*\d+", "", text
    )  # Supprime les numéros de pages
    text = re.sub(r"©.*?(?:\d{4})?", "", text)  # Supprime les mentions de copyright

    # Supprimer les mentions de licence et réutilisation
    text = re.sub(r"Creative Commons.*?License", "", text)

    # Supprimer les informations de correspondance (emails et institutions)
    text = re.sub(r"[Cc]orrespondence.*?:.*", "", text)

    # Remplacer les nouvelles lignes et tabulations par des espaces
    text = re.sub(r"[\n\t\r]+", " ", text)

    # Supprimer les espaces multiples
    text = re.sub(r"\s+", " ", text)

    # Nettoyer les espaces au début et à la fin du texte
    text = text.strip()

    return Document(page_content=text, metadata=doc.metadata)


# Function to calculate the hash of the file content
def calculate_file_hash(file_path):
    with open(file_path, "rb") as f:
        file_content = f.read()
    return hashlib.sha256(file_content).hexdigest()


def load_or_create_vector_store(file_path: str, embedding_function: Embeddings):
    """
    Load or create a vector store for a given file path

    Args:
        file_path (str): The path to the file
        embedding_function (Embeddings): The embedding function to use

    Returns:
        vector_store: The vector store
    """

    file_hash = calculate_file_hash(file_path)
    embeddings_path = os.path.join(embedding_folder, file_hash)
    embeddings_path += (
        "_openai_embeddings"
        if type(embedding_function) == OpenAIEmbeddings
        else "_custom_embeddings"
    )

    if os.path.exists(embeddings_path):
        print("Loading existing vector store")
        vector_store = FAISS.load_local(
            embeddings_path, embedding_function, allow_dangerous_deserialization=True
        )
        return vector_store

    else:
        print("Creating new vector store")
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()

        pages_cleaned = [clean_scientific_text(page) for page in pages]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        all_splits = text_splitter.split_documents(pages_cleaned)

        uuids = [str(uuid4()) for _ in range(len(all_splits))]

        index = faiss.IndexFlatL2(len(embedding_function.embed_query("hello world")))

        vector_store = FAISS(
            embedding_function=embedding_function,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        vector_store.add_documents(documents=all_splits, ids=uuids)
        vector_store.save_local(embeddings_path)

        return vector_store


def get_retriever(
    search_type: str, search_kwargs: dict, pdf_path: str, embedding_function: Embeddings
):
    """
    Get a retriever object for a given search type, search arguments, and PDF file.
    If the vector store for the PDF file does not exist, it will be created using the given embedding function.

    Args:
        search_type (str): The search type to use
        search_kwargs (dict): The search arguments to pass to the retriever
        pdf_path (str): The path to the PDF file to use
        embedding_function (Embeddings): The embedding function to use. If the vector store does not exist, it will be created using this function, otherwise it will be loade and the embedding function should be the same as the one used to create the vector store.

    Returns:
        retriever: The retriever object
    """

    vector_store = load_or_create_vector_store(pdf_path, embedding_function)
    retriever = vector_store.as_retriever(
        search_type=search_type, search_kwargs=search_kwargs
    )
    return retriever
