from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
reader = PDFReader()

EMBED_MODEL = "models/text-embedding-004"
EMBED_DIM = 768 # has to match whats in vector_db.py

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200) # 200 characters so each chunk has last chunk's context

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks 


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    
    # Single text (query)
    if len(texts) == 1:
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=texts[0],
            task_type="retrieval_query"
        )
        return [result['embedding']]
    
    BATCH_SIZE = 100
    all_embeddings = []
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        try:
            result = genai.embed_content(
                model=EMBED_MODEL,
                content=batch,
                task_type="retrieval_document"
            )
            all_embeddings.extend(result['embeddings'])
        except Exception as e:
            print(f"Batch {i//BATCH_SIZE} failed, processing sequentially: {e}")
            for text in batch:
                single_result = genai.embed_content(
                    model=EMBED_MODEL,
                    content=text,
                    task_type="retrieval_document"
                )
                all_embeddings.append(single_result['embedding'])
        
    
    return all_embeddings