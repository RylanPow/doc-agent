from google import genai
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os

load_dotenv()

EMBED_MODEL = "gemini-embedding-001"  # embedding model name
reader = PDFReader()
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def load_and_chunk_pdf(path: str):
    docs = reader.load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts
    )
    return response.embeddings
