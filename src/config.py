from dotenv import load_dotenv
import os

load_dotenv()

### Langchain API
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "test_rag"

### Local Embedding Model
openwebui_api_key = os.getenv("OPENWEBUI_API_KEY")
base_url = "http://149.202.125.247:8080/ollama"
embedding_model = "mxbai-embed-large:latest"
embedding_folder = os.path.join("src", "embeddings")
UPLOAD_DIRECTORY = os.path.join("src", "docs")


### RAG Model
root_doc_path = os.path.join("src", "docs")
pdf_paths = [os.path.join(root_doc_path, pdf) for pdf in os.listdir(root_doc_path)]
search_types = ["mmr", "similarity"]

system_prompt = """
You are an assistant for question-answering tasks.
You are an expert in biology and medicine. You are asked to answer questions
based solely on the retrieved context, avoiding all introductory or redundant phrases (e.g., "In this article," "Yes," "The article discusses").
Your answers should be concise, factual, and directly usable for a summary without requiring further editing.

If the context does not provide the necessary information, respond with "Information not available."
Use three sentences maximum and keep the answers concise. Use markdown to format the answers and make them more readable.

{context}
"""
