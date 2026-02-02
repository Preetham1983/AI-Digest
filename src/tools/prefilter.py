from typing import List
import numpy as np
from src.models.items import IngestedItem
from src.services.vector_store import vector_store
from src.services.logger import logger
from src.config import settings

# Anchor concepts representing the ideal content for each persona
# ANCHORS = {
#     "GENAI": "Technical details about Large Language Models, AI agents, RAG, transformer architectures, and new model releases like Llama or GPT.",
#     "PRODUCT": "New software startup ideas, B2B SaaS opportunities, market gaps, and problems enabling new product development.",
#     "FINANCE": "Financial reports of tech companies, revenue data, funding rounds, IPOs, stock market analysis for AI companies.",
# }

# Broad keywords to filter obvious noise
# If it's tech news, these are decent signals.
KEYWORDS = {
    "ai", "llm", "gpt", "model", "agent", "rag", "transformer", 
    "neural", "ml", "deep learning", "generative", "inference",
    "python", "rust", "cuda", "gpu", "framework", "library",
    "tool", "launch", "release", "open source", "github",
    "api", "database", "search", "vector", "embedding",
    "startup", "product", "app", "web", "developer", "engineer"
}

class KeywordPrefilter:
    async def is_relevant(self, item: IngestedItem) -> bool:
        """
        Fast pre-filter to check if item has any relevant keywords.
        """
        text = (item.title + " " + (item.content or "")).lower()
        
        # Check for keywords
        if any(k in text for k in KEYWORDS):
            return True
        
        # Keep if score is high (HN > 100 points) regardless of keywords
        if item.raw_score and item.raw_score > 100:
            return True
            
        return False

# class SemanticPrefilter:
#     def __init__(self):
#         self.anchor_embeddings = {}
    
#     async def initialize(self):
#         """Pre-compute embeddings for anchors."""
#         logger.info("Initializing Semantic Prefilter Anchors...")
#         for key, text in ANCHORS.items():
#             emb = await vector_store.get_embedding(text)
#             self.anchor_embeddings[key] = np.array(emb, dtype='float32')
            
#     async def is_relevant(self, item: IngestedItem, threshold: float = 0.40) -> bool:
#         """
#         Checks if item is semantically similar to ANY of the active persona anchors.
#         Threshold: 0.4 is a safe baseline for Cosine Similarity (if vectors were normalized).
#         Since we are using L2-optimized vectors from vector_store (unnormalized?), we might need to normalize manually here for Cosine.
        
#         Let's do manual Cosine Similarity: A . B / (|A| * |B|)
#         """
#         if not self.anchor_embeddings:
#             await self.initialize()

#         text = (item.title + "\n" + (item.content or ""))[:1000]
        
#         if item.embedding:
#             item_emb = item.embedding
#         else:
#             item_emb = await vector_store.get_embedding(text)
#             item.embedding = item_emb # Cache it
            
#         item_vec = np.array(item_emb, dtype='float32')
        
#         # Normalize item vector
#         norm_i = np.linalg.norm(item_vec)
#         if norm_i == 0: return False
        
#         item_vec_norm = item_vec / norm_i

#         max_score = -1.0
        
#         # Check against enabled personas
#         cheks = []
#         if settings.PERSONA_GENAI_NEWS_ENABLED: cheks.append("GENAI")
#         if settings.PERSONA_PRODUCT_IDEAS_ENABLED: cheks.append("PRODUCT")
#         if settings.PERSONA_FINANCE_ENABLED: cheks.append("FINANCE")
        
#         for key in cheks:
#             anchor = self.anchor_embeddings.get(key)
#             if anchor is None: continue
            
#             # Normalize anchor (pre-calc ideally, but cheap enough here)
#             norm_a = np.linalg.norm(anchor)
#             if norm_a == 0: continue
            
#             anchor_norm = anchor / norm_a
            
#             # Cosine Similarity
#             score = np.dot(item_vec_norm, anchor_norm)
#             if score > max_score:
#                 max_score = score
        
#         # logger.info(f"Item '{item.title}' max relevance score: {max_score:.3f}")
        
#         if max_score > threshold:
#             return True
            
#         # Fallback: Keywords for safety (hybrid approach)
#         # If semantic fails but it has strong keywords, keep it?
#         # User said "better than keywords", so let's trust semantic primarily.
#         # But maybe keep "Legacy" keyword check as a 2nd pass if score is borderline (e.g. 0.35-0.4).
        
#         return False

# Global instance
prefilter = KeywordPrefilter()
