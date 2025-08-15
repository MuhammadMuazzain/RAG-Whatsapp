"""
RAG (Retrieval-Augmented Generation) Module
Handles vector search and LLM generation with Mistral via Ollama
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Generator, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
from functools import lru_cache
import time

try:
    from performance_config import (
        MODEL_CONFIG, EMBEDDING_CONFIG, CACHE_CONFIG, 
        SEARCH_CONFIG, get_optimized_config
    )
except ImportError:
    # Fallback to default settings if config not found
    MODEL_CONFIG = {"temperature": 0.7, "max_tokens": 500}
    EMBEDDING_CONFIG = {"model_name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"}
    CACHE_CONFIG = {"max_cache_size": 100}
    SEARCH_CONFIG = {"default_top_k": 3}
    get_optimized_config = lambda mode: MODEL_CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGEngine:
    """Handles retrieval from FAISS and generation with Mistral"""
    
    def __init__(self, 
                 embedding_model: str = None,
                 ollama_url: str = "http://localhost:11435",
                 performance_mode: str = "balanced"):
        """Initialize RAG engine with embedding model and Ollama endpoint"""
        logger.info(f"Initializing RAG engine in {performance_mode} mode")
        
        # Load configuration
        self.performance_mode = performance_mode
        self.model_config = get_optimized_config(performance_mode)
        
        # Load embedding model
        model_name = embedding_model or EMBEDDING_CONFIG.get("model_name", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Set device for embeddings
        if EMBEDDING_CONFIG.get("device") == "cuda":
            import torch
            if torch.cuda.is_available():
                self.embedding_model = self.embedding_model.to("cuda")
                logger.info("Using GPU for embeddings")
        
        self.ollama_url = ollama_url
        
        # Load FAISS index and metadata
        self.vector_store_path = Path("vector_store")
        self.index = None
        self.chunks = None
        self._load_index_and_metadata()
        
        # Cache for embeddings to speed up repeated queries
        self.embedding_cache = {}
        self.max_cache_size = CACHE_CONFIG.get("max_cache_size", 100)
        self.cache_timestamps = {}  # Track cache age
        
    def _load_index_and_metadata(self) -> None:
        """Load FAISS index and chunk metadata"""
        index_path = self.vector_store_path / "faiss.index"
        metadata_path = self.vector_store_path / "chunks.json"
        
        if not index_path.exists() or not metadata_path.exists():
            logger.error("FAISS index or metadata not found. Run embed.py first.")
            raise FileNotFoundError("Vector store not initialized. Process PDFs first.")
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
        
        # Load chunks
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            self.chunks = metadata.get("chunks", [])
        
        logger.info(f"Loaded {len(self.chunks)} text chunks")
    
    @lru_cache(maxsize=CACHE_CONFIG.get("embedding_cache_size", 128))
    def search_similar_chunks(self, query: str, top_k: int = None, min_score: float = 0.1) -> List[Tuple[str, float]]:
        """Search for similar chunks using FAISS with caching and filtering"""
        if top_k is None:
            top_k = SEARCH_CONFIG.get("default_top_k", 3)
        top_k = min(top_k, SEARCH_CONFIG.get("max_top_k", 10))
        
        logger.info(f"Searching for top {top_k} chunks for query: {query[:50]}...")
        
        # Check cache first
        cache_key = f"{query}_{top_k}"
        if cache_key in self.embedding_cache:
            logger.info("Using cached embeddings")
            query_embedding = self.embedding_cache[cache_key]
        else:
            # Embed query
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Add to cache (with size limit)
            if len(self.embedding_cache) < self.max_cache_size:
                self.embedding_cache[cache_key] = query_embedding
        
        # Search in FAISS - get more results for filtering
        distances, indices = self.index.search(query_embedding, min(top_k * 2, 10))
        
        # Get chunks and scores with filtering
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.chunks):
                # For cosine similarity, higher score is better (max 1.0)
                if score >= min_score:
                    results.append((self.chunks[idx], float(score)))
                    logger.info(f"Chunk score: {score:.3f}")
        
        # Limit to top_k after filtering
        results = results[:top_k]
        
        # If no results found with threshold, get at least one best match
        if not results and indices[0].size > 0:
            best_idx = indices[0][0]
            best_score = distances[0][0]
            if 0 <= best_idx < len(self.chunks):
                results = [(self.chunks[best_idx], float(best_score))]
                logger.warning(f"No chunks above threshold {min_score}, using best match with score {best_score:.3f}")
        
        logger.info(f"Found {len(results)} relevant chunks (filtered by score >= {min_score})")
        return tuple(results)  # Convert to tuple for caching
    
    def format_prompt(self, query: str, context_chunks: List[str], response_style: str = "moderate") -> str:
        """Format prompt for Mistral with context and query"""
        # Limit context based on response style
        if response_style == "brief":
            context_chunks = context_chunks[:1]  # Use only most relevant chunk
        elif response_style == "moderate":
            context_chunks = context_chunks[:2]  # Use top 2 chunks
        # detailed uses all chunks
        
        context = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])
        
        # Style-specific instructions - MUCH MORE STRICT
        style_instructions = {
            "brief": "Answer in EXACTLY 1-2 SHORT sentences only. Maximum 30 words total.",
            "moderate": "Answer in EXACTLY 2-3 SHORT sentences only. Maximum 50 words total. Be direct.",
            "detailed": "Answer in 4-5 sentences maximum. Include key details but stay under 100 words."
        }
        
        instruction = style_instructions.get(response_style, style_instructions["moderate"])
        
        prompt = f"""You are a medical information assistant. You MUST ONLY use the exact information provided.

CRITICAL RULES:
1. {instruction}
2. ONLY use information that is EXPLICITLY stated in the Context below.
3. If specific numbers/percentages are asked but NOT in the context, say "I don't have that specific data."
4. NEVER make up statistics, numbers, or percentages.
5. For Singapore or any specific location data - ONLY state if explicitly mentioned in context.
6. BE CONCISE. Keep responses short.

Context from documents:
{context}

User Question: {query}

Answer (ONLY from context, NO external knowledge):"""
        
        return prompt
    



    def query_stream(self, prompt: str, temperature: float = None, max_tokens: int = None) -> Generator[str, None, None]:
        """Stream response from Mistral via Ollama in real time."""
        logger.info("Streaming response with Mistral-7B")
        
        # Use configured values or defaults
        if temperature is None:
            temperature = self.model_config.get("temperature", 0.7)
        if max_tokens is None:
            max_tokens = self.model_config.get("max_tokens", 500)

        # Build options from config
        options = {
            "temperature": temperature,
            "top_p": self.model_config.get("top_p", 0.9),
            "num_predict": max_tokens,
            "num_ctx": self.model_config.get("num_ctx", 4096),
            "num_batch": self.model_config.get("num_batch", 512),
            "num_thread": self.model_config.get("num_thread", 8),
            "repeat_penalty": self.model_config.get("repeat_penalty", 1.1),
            "stop": self.model_config.get("stop", ["\n\n", "User:", "Human:"])
        }
        
        # Add GPU settings if available
        if self.model_config.get("num_gpu"):
            options["num_gpu"] = self.model_config["num_gpu"]
            options["main_gpu"] = self.model_config.get("main_gpu", 0)
        
        data = {
            "model": "mistral",
            "prompt": prompt,
            "stream": True,
            "options": options
        }

        try:
            with requests.post(
                f"{self.ollama_url}/api/generate",
                json=data,
                timeout=180,
                stream=True
            ) as response:
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line.decode("utf-8"))
                                if "response" in chunk:
                                    yield chunk["response"]  # Send each piece immediately
                            except Exception as e:
                                logger.error(f"Error decoding stream chunk: {e}")
                else:
                    logger.error(f"Ollama API error: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to Ollama. Make sure Ollama is running.")
        except Exception as e:
            logger.error(f"Error generating response: {e}")

    
    # def generate_response(self, prompt: str) -> str:
    #     """Generate response using Mistral via Ollama"""
    #     logger.info("Generating response with Mistral-7B")
        
    #     try:
    #         # Prepare request to Ollama
    #         # data = {
    #         #     "model": "mistral",
    #         #     "prompt": prompt,
    #         #     "stream": False,
    #         #     "options": {
    #         #         "temperature": 0.7,
    #         #         "top_p": 0.9,
    #         #         "max_tokens": 500
    #         #     }
    #         # }
            
    #         data = {
    #         "model": "mistral",
    #         "prompt": prompt,
    #         "stream": True,
    #         "options": {
    #             "temperature": 0.7,
    #             "top_p": 0.9,
    #             "num_predict": 500
    #         }
    #     }

    #         # Make request to Ollama

    #         response = requests.post(
    #             f"{self.ollama_url}/api/generate",
    #             json={**data, "stream": True},  # enable streaming in Ollama
    #             timeout=180,
    #             stream=True  # let requests yield chunks as they arrive
    #         )
    #         # response = requests.post(
    #         #     f"{self.ollama_url}/api/generate",
    #         #     json=data,
    #         #     timeout=180
    #         # )
            

    #         if response.status_code == 200:
    #             for line in response.iter_lines():
    #                 if line:
    #                     try:
    #                         chunk = json.loads(line.decode("utf-8"))
    #                         if "response" in chunk:
    #                             yield chunk["response"]  # send each piece
    #                     except Exception as e:
    #                         logger.error(f"Error decoding stream chunk: {e}")
    #         # if response.status_code == 200:
    #         #     result = response.json()
    #         #     generated_text = result.get("response", "")
    #         #     logger.info("Successfully generated response")
    #         #     return generated_text
    #         else:
    #             logger.error(f"Ollama API error: {response.status_code} - {response.text}")
    #             return "I'm sorry, I encountered an error generating a response."
                
    #     except requests.exceptions.ConnectionError:
    #         logger.error("Failed to connect to Ollama. Make sure Ollama is running.")
    #         return "Service temporarily unavailable. Please ensure Ollama is running."
    #     except Exception as e:
    #         logger.error(f"Error generating response: {e}")
    #         return "I encountered an error while processing your request."
    



    #Ollama saying I should change my query method:

    # to this:

    def query(self, user_query: str, top_k: int = None, response_style: str = "moderate") -> Dict[str, any]:
        """Main RAG pipeline: search + generate (non-streaming)"""
        logger.info(f"Processing query: {user_query}")
        
        start_time = time.time()
        
        # Search for relevant chunks
        search_results = self.search_similar_chunks(user_query, top_k)
        
        # Extract just the chunk texts
        context_chunks = [chunk for chunk, score in search_results]
        
        # Format prompt with style
        prompt = self.format_prompt(user_query, context_chunks, response_style)
        
        # Generate response with STRICT length control
        response = ""
        # Much shorter limits
        max_tokens = {"brief": 50, "moderate": 75, "detailed": 150}.get(response_style, 75)
        max_chars = {"brief": 150, "moderate": 250, "detailed": 500}.get(response_style, 250)
        
        for chunk in self.query_stream(prompt, max_tokens=max_tokens):
            response += chunk
            # Stop if we exceed character limit
            if len(response) > max_chars:
                # Find last complete sentence
                sentences = response.split('. ')
                if len(sentences) > 1:
                    response = '. '.join(sentences[:-1]) + '.'
                else:
                    response = response[:max_chars].rsplit(' ', 1)[0] + '...'
                break
        
        # Prepare result
        result = {
            "query": user_query,
            "response": response,
            "sources": [
                {"chunk": chunk, "score": score} 
                for chunk, score in search_results
            ],
            "processing_time": time.time() - start_time
        }
        
        logger.info(f"Query processed in {result['processing_time']:.2f} seconds")
        return result
    

    def query_with_stream(self, user_query: str, top_k: int = None, response_style: str = "moderate") -> Generator[str, None, None]:
        """RAG pipeline that yields streaming responses with SSE format"""
        logger.info(f"Processing streaming query: {user_query}")
        
        try:
            # Search for relevant chunks
            search_results = self.search_similar_chunks(user_query, top_k)
            
            # Extract just the chunk texts
            context_chunks = [chunk for chunk, score in search_results]
            
            # Format prompt with context and style
            prompt = self.format_prompt(user_query, context_chunks, response_style)
            
            # Stream the response
            for chunk in self.query_stream(prompt):
                if chunk:  # Only yield non-empty chunks
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
            
            # Send end signal
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming query: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"



    # def query(self, user_query: str, top_k: int = 3) -> Dict[str, any]:
    #     """Main RAG pipeline: search + generate"""
    #     logger.info(f"Processing query: {user_query}")
        
    #     # Search for relevant chunks
    #     search_results = self.search_similar_chunks(user_query, top_k)
        
    #     # Extract just the chunk texts
    #     context_chunks = [chunk for chunk, score in search_results]
        
    #     # Format prompt
    #     prompt = self.format_prompt(user_query, context_chunks)
        
    #     # Generate response
    #     # response = self.generate_response(prompt)
    #     response =self.query_stream(prompt)
        
    #     # Prepare result
    #     result = {
    #         "query": user_query,
    #         "response": response,
    #         "sources": [
    #             {"chunk": chunk, "score": score} 
    #             for chunk, score in search_results
    #         ]
    #     }
        
    #     return result
    
    def health_check(self) -> Dict[str, any]:
        """Check if all components are working"""
        status = {
            "embedding_model": "loaded" if self.embedding_model else "not loaded",
            "faiss_index": f"{self.index.ntotal} vectors" if self.index else "not loaded",
            "chunks": f"{len(self.chunks)} chunks" if self.chunks else "not loaded",
            "ollama": "unknown"
        }
        
        # Check Ollama
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                mistral_available = any("mistral" in model.get("name", "") for model in models)
                status["ollama"] = "connected" if mistral_available else "mistral not found"
                status["available_models"] = [m.get("name") for m in models]
            else:
                status["ollama"] = "error"
        except:
            status["ollama"] = "not connected"
        
        status["cache_size"] = len(self.embedding_cache)
        
        return status


    def clear_cache(self):
        """Clear the embedding cache"""
        self.embedding_cache.clear()
        self.search_similar_chunks.cache_clear()
        logger.info("Cache cleared")

    def warm_up(self):
        """Warm up the model with a dummy query"""
        logger.info("Warming up RAG engine...")
        try:
            # Warm up embeddings
            self.search_similar_chunks("test query", 1)
            
            # Pre-load model in Ollama if possible
            test_prompt = "Hello, this is a test."
            data = {
                "model": "mistral",
                "prompt": test_prompt,
                "stream": False,
                "options": {"num_predict": 1}
            }
            requests.post(f"{self.ollama_url}/api/generate", json=data, timeout=10)
            
            logger.info("Warm-up complete")
        except Exception as e:
            logger.error(f"Warm-up failed: {e}")
    
    def set_performance_mode(self, mode: str):
        """Change performance mode at runtime"""
        if mode in ["speed", "quality", "balanced"]:
            self.performance_mode = mode
            self.model_config = get_optimized_config(mode)
            logger.info(f"Performance mode changed to: {mode}")
        else:
            logger.error(f"Invalid performance mode: {mode}")

def test_rag():
    """Test function for RAG engine"""
    print("\n" + "="*50)
    print("TESTING RAG ENGINE")
    print("="*50)
    
    # Initialize RAG
    rag = RAGEngine()
    
    # Health check
    print("\nHealth Check:")
    health = rag.health_check()
    for component, status in health.items():
        print(f"  {component}: {status}")
    
    # Test query
    test_query = "What is the main topic of this document?"
    print(f"\nTest Query: {test_query}")
    
    result = rag.query(test_query)
    
    print(f"\nResponse: {result['response']}")
    print(f"\nSources used: {len(result['sources'])}")
    for i, source in enumerate(result['sources']):
        print(f"\n  Source {i+1} (score: {source['score']:.3f}):")
        print(f"  {source['chunk'][:100]}...")


if __name__ == "__main__":
    test_rag()