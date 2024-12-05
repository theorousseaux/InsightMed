from .chains import get_rag_chain
from .config import pdf_paths, search_types, UPLOAD_DIRECTORY
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from pydantic import BaseModel, field_validator, Field
from typing import Dict, Any, List
from langchain_openai import OpenAIEmbeddings
import os
import shutil
import pandas as pd

app = FastAPI()
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@app.get("/list_pdfs")
def list_pdfs() -> List[str]:
    """
    List all available PDFs
    """
    return pdf_paths


@app.post("/upload_pdf")
def upload_pdf(file: UploadFile = File(..., description="PDF file to upload")):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_size = len(file.file.read())
    file.file.seek(0)  # Reset file pointer to the beginning

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds the 20 MB limit")

    try:
        os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        pdf_paths.append(file_path)

        return {"message": f"File {file.filename} uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


class ChainParameters(BaseModel):
    search_type: str = Field(..., description="The search type to use", example="mmr")
    pdf_path: str = Field(
        ..., description="The path to the PDF file to search in", example="path/to/pdf"
    )
    top_k: int = Field(
        5, description="The number of chunks to retrieve", ge=1, examples=5
    )

    @field_validator("search_type")
    def validate_search_type(cls, v):
        if v not in search_types:
            raise ValueError(f"search_type must be one of {search_types}")
        return v

    @field_validator("pdf_path")
    def validate_collection_name(cls, v):
        if v not in pdf_paths:
            raise ValueError(f"PDF must be one of {pdf_paths}")
        return v


class DocumentResponse(BaseModel):
    page_content: str
    metadata: Dict[str, Any]


class QueryArticleResponse(BaseModel):
    question: str
    answer: str
    response_metadata: Dict[str, Any] = None
    context: List[DocumentResponse]


@app.post("/query_article")
def query_article(
    chain_parameters: ChainParameters,
    query: str = Body(
        ..., description="The query to search for", example="AI advancements"
    ),
) -> QueryArticleResponse:
    """
    Query a given article using the RAG model
    """

    rag_chain = get_rag_chain(
        search_type=chain_parameters.search_type,
        search_kwargs={"k": chain_parameters.top_k},
        pdf_path=chain_parameters.pdf_path,
        embedding_function=embeddings,
    )

    response = rag_chain.invoke(query)

    context = [
        DocumentResponse(page_content=doc.page_content, metadata=doc.metadata)
        for doc in response["chunks"]
    ]

    return QueryArticleResponse(
        question=response["question"],
        answer=response["llm_response"].content,
        response_metadata=response["llm_response"].response_metadata,
        context=context,
    )


@app.post("/resume_article_from_prompts")
def resume_article_from_prompts(
    chain_parameters: ChainParameters,
    prompts: List[str] = Body(
        ...,
        example=["AI advancements"],
        description="A list of prompts to resume the article from",
    ),
) -> Dict[str, List[str]]:
    """
    Resume an article from a list of prompts using the RAG model
    """

    rag_chain = get_rag_chain(
        search_type=chain_parameters.search_type,
        search_kwargs={"k": chain_parameters.top_k},
        pdf_path=chain_parameters.pdf_path,
        embedding_function=embeddings,
    )

    answers = []
    for prompt in prompts:
        response = rag_chain.invoke(prompt)
        answer = response["llm_response"].content
        answers.append(answer)

    response_df = pd.DataFrame({"prompts": prompts, "answers": answers})

    return response_df.to_dict(orient="list")
