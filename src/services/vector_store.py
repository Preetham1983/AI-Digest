import faiss
import numpy as np
import ollama
import pickle
import os
from src.config import settings
from typing import List, Tuple
from src.services.logger import logger

INDEX_FILE = settings.DATA_DIR / "vector_index.faiss"
ID_MAP_FILE = settings.DATA_DIR / "vector_ids.pkl"

class VectorStore:
    def __init__(self):
        self.dimension = 4096 # Llama3 embedding size
        # Check if index exists
        if os.path.exists(INDEX_FILE) and os.path.exists(ID_MAP_FILE):
             self.load_index()
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.stored_ids = [] # map index -> item_id (url)
        
        self.id_set = set(self.stored_ids) # Fast lookup
        logger.info(f"VectorStore initialized with {self.index.ntotal} items.")

    def save_index(self):
        try:
            faiss.write_index(self.index, str(INDEX_FILE))
            with open(ID_MAP_FILE, "wb") as f:
                pickle.dump(self.stored_ids, f)
            logger.info("Vector index saved.")
        except Exception as e:
            logger.error(f"Failed to save vector index: {e}")

    def load_index(self):
        try:
            self.index = faiss.read_index(str(INDEX_FILE))
            with open(ID_MAP_FILE, "rb") as f:
                self.stored_ids = pickle.load(f)
            self.id_set = set(self.stored_ids)
        except Exception as e:
            logger.error(f"Failed to load vector index: {e}")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.stored_ids = []
            self.id_set = set()

    def has_id(self, id: str) -> bool:
        return id in self.id_set

    async def get_embedding(self, text: str) -> List[float]:
        try:
            # truncate text to avoid context limit errors
            resp = await ollama.AsyncClient(host=settings.OLLAMA_BASE_URL).embeddings(model=settings.OLLAMA_MODEL, prompt=text[:2000])
            return resp['embedding']
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return [0.0] * self.dimension

    async def add_item(self, id: str, text: str, embedding: List[float] = None):
        if embedding:
            emb = embedding
        else:
            emb = await self.get_embedding(text)
        vec = np.array([emb], dtype='float32')
        self.index.add(vec)
        self.stored_ids.append(id)
        self.id_set.add(id)
        # Auto-save every 10 items or just rely on pipeline to call save
        
    async def is_duplicate(self, text: str, threshold: float = 0.20, embedding: List[float] = None) -> bool:
        """
        Checks if a semantically similar item exists.
        Threshold: Lower means stricter/closer. 
        For L2 distance: 0 is identical. ~15000 is far for 4096 dim? 
        Actually for normalized embeddings, L2 = 2(1-cos). 
        If embeddings are NOT normalized, L2 is unbounded. 
        Ollama embeddings are usually normalized? Let's assume standard L2.
        We'll use a conservative threshold.
        """
        if self.index.ntotal == 0:
            return False
            
        if embedding:
            emb = embedding
        else:
            emb = await self.get_embedding(text)
        vec = np.array([emb], dtype='float32')
        
        # Search for 1 nearest neighbor
        D, I = self.index.search(vec, 1)
        
        distance = D[0][0]
        # logger.info(f"Nearest neighbor distance: {distance}")
        
        # Heuristic: For Llama3 embeddings (unnormalized usually), distance can be large.
        # Let's trust exact URL match primarily, and very close vector match.
        # Ideally we should strictly use Cosine Similarity (IndexFlatIP with normalized vectors).
        # But switching to L2 for simplicity as started.
        # If distance is very small, it's a duplicate.
        
        if distance < threshold: 
            return True
        return False
        
    async def search(self, text: str, k: int = 5) -> List[Tuple[str, float]]:
        emb = await self.get_embedding(text)
        vec = np.array([emb], dtype='float32')
        D, I = self.index.search(vec, k)
        
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx != -1 and idx < len(self.stored_ids):
                results.append((self.stored_ids[idx], float(dist)))
        return results

# Global instance
vector_store = VectorStore()
