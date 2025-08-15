"""
Performance Configuration and Optimization Settings
"""

# Model Performance Settings
MODEL_CONFIG = {
    "temperature": 0.7,          # Lower = more focused, Higher = more creative
    "max_tokens": 500,           # Maximum response length
    "top_p": 0.9,               # Nucleus sampling threshold
    "num_ctx": 4096,            # Context window size
    "num_batch": 512,           # Batch size for prompt processing
    "num_thread": 8,            # Number of CPU threads
    "repeat_penalty": 1.1,      # Penalty for repetition
    "num_gpu": 1,               # Number of GPUs to use (if available)
    "main_gpu": 0,              # Main GPU for computation
    "low_vram": False,          # Enable if GPU has limited VRAM
    "f16_kv": True,            # Use 16-bit floats for key-value cache
    "logits_all": False,       # Return logits for all tokens
    "vocab_only": False,       # Only load vocabulary
    "use_mmap": True,          # Use memory-mapped files
    "use_mlock": False,        # Lock model in RAM
    "embedding_only": False,   # Only use embeddings
    "rope_frequency_base": 10000.0,  # RoPE base frequency
    "rope_frequency_scale": 1.0,     # RoPE frequency scaling
    "num_keep": 0,             # Number of tokens to keep from initial prompt
    "seed": -1,                # Random seed (-1 for random)
    "stop": ["\\n\\n", "User:", "Human:", "Assistant:"],  # Stop sequences
    "tfs_z": 1.0,              # Tail-free sampling parameter
    "typical_p": 1.0,          # Typical sampling parameter
    "mirostat": 0,             # Mirostat sampling (0=disabled, 1=v1, 2=v2)
    "mirostat_tau": 5.0,       # Mirostat target entropy
    "mirostat_eta": 0.1,       # Mirostat learning rate
}

# Embedding Model Settings
EMBEDDING_CONFIG = {
    "model_name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "device": "cpu",            # Use "cuda" if GPU available
    "normalize_embeddings": True,
    "batch_size": 32,          # Batch size for encoding
}

# FAISS Index Settings
FAISS_CONFIG = {
    "index_type": "Flat",       # Can be "Flat", "IVF", "HNSW"
    "nlist": 100,              # Number of clusters for IVF
    "nprobe": 10,              # Number of clusters to search
    "efConstruction": 200,     # HNSW construction parameter
    "efSearch": 50,            # HNSW search parameter
}

# Cache Settings
CACHE_CONFIG = {
    "max_cache_size": 100,      # Maximum number of cached queries
    "cache_ttl": 3600,         # Cache time-to-live in seconds
    "embedding_cache_size": 128,  # LRU cache size for embeddings
}

# Search Settings
SEARCH_CONFIG = {
    "default_top_k": 3,        # Default number of chunks to retrieve
    "max_top_k": 10,           # Maximum allowed top_k value
    "min_similarity": 0.3,     # Minimum similarity threshold
    "rerank": False,           # Enable reranking of results
}

# Preprocessing Settings
PREPROCESSING_CONFIG = {
    "chunk_size": 512,         # Size of text chunks
    "chunk_overlap": 50,       # Overlap between chunks
    "max_chunk_size": 1000,    # Maximum chunk size
    "min_chunk_size": 100,     # Minimum chunk size
}

# Server Settings
SERVER_CONFIG = {
    "enable_streaming": True,   # Enable streaming responses
    "stream_buffer_size": 1024, # Buffer size for streaming
    "request_timeout": 180,     # Request timeout in seconds
    "max_concurrent_requests": 10,  # Maximum concurrent requests
    "enable_cors": True,        # Enable CORS
    "cors_origins": ["*"],      # Allowed CORS origins
}

# Performance Tips
PERFORMANCE_TIPS = """
Performance Optimization Tips:

1. **GPU Acceleration**: 
   - Use GPU for embeddings: Set EMBEDDING_CONFIG["device"] = "cuda"
   - Enable GPU in Ollama: ollama run mistral --gpu-layers 35

2. **Model Quantization**:
   - Use quantized models for faster inference: ollama pull mistral:7b-q4_0
   - Trade-off: Slightly lower quality for much faster speed

3. **Caching**:
   - Enable query caching to avoid recomputation
   - Use Redis for distributed caching in production

4. **Batch Processing**:
   - Process multiple queries in batches when possible
   - Increase EMBEDDING_CONFIG["batch_size"] for better throughput

5. **Index Optimization**:
   - For large datasets (>1M vectors), use IVF or HNSW indices
   - Trade-off: Faster search but slightly lower recall

6. **Context Window**:
   - Reduce MODEL_CONFIG["num_ctx"] for faster processing
   - Trade-off: Less context available for generation

7. **Streaming**:
   - Enable streaming for better perceived performance
   - Users see results immediately as they're generated

8. **Preprocessing**:
   - Pre-compute embeddings during quiet periods
   - Use smaller embedding models for faster encoding

9. **Hardware**:
   - Use SSDs for faster model loading
   - Ensure sufficient RAM (8GB+ recommended)
   - Use NVIDIA GPUs with CUDA for best performance

10. **Monitoring**:
    - Track response times and optimize bottlenecks
    - Use profiling tools to identify slow operations
"""

def get_optimized_config(mode="balanced"):
    """
    Get optimized configuration based on mode.
    
    Modes:
    - "speed": Optimize for fastest response time
    - "quality": Optimize for best response quality
    - "balanced": Balance between speed and quality
    """
    config = MODEL_CONFIG.copy()
    
    if mode == "speed":
        config.update({
            "temperature": 0.5,
            "max_tokens": 200,
            "num_ctx": 2048,
            "num_batch": 256,
        })
    elif mode == "quality":
        config.update({
            "temperature": 0.8,
            "max_tokens": 1000,
            "num_ctx": 8192,
            "num_batch": 1024,
        })
    # balanced mode uses default settings
    
    return config

if __name__ == "__main__":
    print(PERFORMANCE_TIPS)
    print("\nCurrent configuration mode: balanced")
    print("\nTo change mode, use get_optimized_config('speed') or get_optimized_config('quality')")