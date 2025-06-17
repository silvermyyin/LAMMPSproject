import faiss # type: ignore
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer # type: ignore

INDEX_PATH = 'rag/knowledge_base/faiss_index.bin'
DOCS_PATH = 'rag/knowledge_base/docs.pkl'
MODEL_NAME = 'all-MiniLM-L6-v2'

model = SentenceTransformer(MODEL_NAME)

with open(DOCS_PATH, 'rb') as f:
    docs = pickle.load(f)
index = faiss.read_index(INDEX_PATH)

def retrieve(query, topk=3):
    emb = model.encode([query]).astype('float32')
    D, I = index.search(emb, topk)
    return [docs[i]['content'] for i in I[0]] 