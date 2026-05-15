"""
Smart chunking pakai LangChain.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def smart_chunk(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """Smart chunking - potong di tempat natural."""
    chunk_size = chunk_size or CHUNK_SIZE
    overlap = overlap or CHUNK_OVERLAP
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            "; ",
            ", ",
            " ",
            ""
        ],
        length_function=len,
    )
    
    return splitter.split_text(text)
    