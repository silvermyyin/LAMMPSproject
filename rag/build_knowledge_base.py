import os
from pathlib import Path
import glob
import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer # type: ignore
import pickle

DOCS_DIR = 'data/RAGdocs'
INDEX_PATH = 'rag/knowledge_base/faiss_index.bin'
DOCS_PATH = 'rag/knowledge_base/docs.pkl'
MODEL_NAME = 'all-MiniLM-L6-v2'

model = SentenceTransformer(MODEL_NAME)

def load_docs():
    docs = []
    for file in glob.glob(f'{DOCS_DIR}/**/*', recursive=True):
        if os.path.isfile(file):
            with open(file, 'r', errors='ignore') as f:
                text = f.read()
                if text.strip():
                    docs.append({'path': file, 'content': text})
    return docs

def build_index(docs):
    texts = [d['content'] for d in docs]
    embeddings = model.encode(texts, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    return index, embeddings

def main():
    docs = load_docs()
    index, embeddings = build_index(docs)
    with open(DOCS_PATH, 'wb') as f:
        pickle.dump(docs, f)
    faiss.write_index(index, INDEX_PATH)
    print(f"Knowledge base built with {len(docs)} documents.")

if __name__ == '__main__':
    main() 