"""
PDF Processing and Embedding Module
Extracts text from PDFs, chunks it, embeds using sentence-transformers, and stores in FAISS
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFEmbedder:
    """Handles PDF processing, chunking, embedding, and FAISS storage"""
    
    def __init__(self, model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"):
        """Initialize embedder with sentence transformer model"""
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.vector_store_path = Path("vector_store")
        self.vector_store_path.mkdir(exist_ok=True)
        
    # def extract_text_from_pdf(self, pdf_path: str) -> str:
    #     """Extract all text from PDF using PyMuPDF"""
    #     logger.info(f"Extracting text from PDF: {pdf_path}")
        
    #     if not os.path.exists(pdf_path):
    #         raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
    #     doc = fitz.open(pdf_path)
    #     text = ""
        
    #     for page_num in range(len(doc)):
    #         page = doc[page_num]
    #         text += page.get_text()
            
    #     doc.close()
        
    #     logger.info(f"Extracted {len(text)} characters from {len(doc)} pages")
    #     return text



    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract all text from PDF using PyMuPDF"""
        logger.info(f"Extracting text from PDF: {pdf_path}")
    
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
            logger.info(f"Extracted {len(text)} characters from {len(doc)} pages")  # ✅ still inside `with`

        return text

    
    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """Split text into chunks of approximately chunk_size words with overlap"""
        logger.info(f"Chunking text into ~{chunk_size} word chunks")
        
        # Split into words
        words = text.split()
        chunks = []
        
        # Create chunks with overlap
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk = " ".join(chunk_words)
            
            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """Embed text chunks using sentence transformer"""
        logger.info(f"Embedding {len(chunks)} chunks")
        
        # Embed all chunks
        embeddings = self.model.encode(
            chunks,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create FAISS index from embeddings"""
        logger.info("Creating FAISS index")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index
        index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        index.add(embeddings)
        
        logger.info(f"FAISS index created with {index.ntotal} vectors")
        return index
    
    def save_index_and_metadata(self, index: faiss.Index, chunks: List[str], 
                                source_file: str) -> None:
        """Save FAISS index and chunk metadata"""
        # Save FAISS index
        index_path = self.vector_store_path / "faiss.index"
        faiss.write_index(index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")
        
        # Save chunk metadata
        metadata = {
            "source_file": source_file,
            "num_chunks": len(chunks),
            "embedding_model": "multi-qa-MiniLM-L6-cos-v1",
            "chunks": chunks
        }
        
        metadata_path = self.vector_store_path / "chunks.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved metadata to {metadata_path}")
    
    def process_pdf(self, pdf_path: str) -> None:
        """Main method to process a PDF file"""
        logger.info(f"Starting PDF processing: {pdf_path}")
        
        try:
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            
            # Chunk text
            chunks = self.chunk_text(text)
            
            # Embed chunks
            embeddings = self.embed_chunks(chunks)
            
            # Create FAISS index
            index = self.create_faiss_index(embeddings)
            
            # Save everything
            self.save_index_and_metadata(index, chunks, os.path.basename(pdf_path))
            
            logger.info("PDF processing completed successfully!")
            
            # Print summary
            print("\n" + "="*50)
            print("PDF PROCESSING SUMMARY")
            print("="*50)
            print(f"Source PDF: {pdf_path}")
            print(f"Total chunks: {len(chunks)}")
            print(f"Embedding dimension: {self.embedding_dim}")
            print(f"Index saved to: vector_store/faiss.index")
            print(f"Metadata saved to: vector_store/chunks.json")
            print("="*50 + "\n")
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
    
    def add_to_existing_index(self, pdf_path: str) -> None:
        """Add new PDF to existing index"""
        logger.info(f"Adding PDF to existing index: {pdf_path}")
        
        # Load existing index and metadata
        index_path = self.vector_store_path / "faiss.index"
        metadata_path = self.vector_store_path / "chunks.json"
        
        if not index_path.exists() or not metadata_path.exists():
            logger.warning("No existing index found. Creating new index.")
            self.process_pdf(pdf_path)
            return
        
        # Load existing data
        index = faiss.read_index(str(index_path))
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        existing_chunks = metadata.get("chunks", [])
        
        # Process new PDF
        text = self.extract_text_from_pdf(pdf_path)
        new_chunks = self.chunk_text(text)
        new_embeddings = self.embed_chunks(new_chunks)
        
        # Normalize and add to index
        faiss.normalize_L2(new_embeddings)
        index.add(new_embeddings)
        
        # Update metadata
        all_chunks = existing_chunks + new_chunks
        metadata["chunks"] = all_chunks
        metadata["num_chunks"] = len(all_chunks)
        metadata["source_files"] = metadata.get("source_files", [metadata.get("source_file", "")])
        if os.path.basename(pdf_path) not in metadata["source_files"]:
            metadata["source_files"].append(os.path.basename(pdf_path))
        
        # Save updated index and metadata
        faiss.write_index(index, str(index_path))
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Successfully added {len(new_chunks)} chunks to existing index")
        print(f"\nAdded {len(new_chunks)} chunks. Total chunks now: {len(all_chunks)}")


def main():
    """CLI interface for PDF embedding"""
    if len(sys.argv) < 2:
        print("Usage: python embed.py <pdf_file> [--append]")
        print("  <pdf_file>: Path to PDF file to process")
        print("  --append: Add to existing index instead of creating new one")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    append_mode = "--append" in sys.argv
    
    # Create embedder
    embedder = PDFEmbedder()
    
    # Process PDF
    if append_mode:
        embedder.add_to_existing_index(pdf_path)
    else:
        embedder.process_pdf(pdf_path)


if __name__ == "__main__":
    main()