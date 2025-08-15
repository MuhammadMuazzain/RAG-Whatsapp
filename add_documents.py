#!/usr/bin/env python
"""
Utility script to add multiple documents to the RAG knowledge base
"""

import os
import sys
from pathlib import Path
from embed import PDFEmbedder
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def add_multiple_documents(pdf_files, reset_index=False):
    """
    Add multiple PDF documents to the knowledge base
    
    Args:
        pdf_files: List of PDF file paths
        reset_index: If True, create new index. If False, append to existing
    """
    embedder = PDFEmbedder()
    
    for i, pdf_path in enumerate(pdf_files):
        if not os.path.exists(pdf_path):
            logger.error(f"File not found: {pdf_path}")
            continue
            
        logger.info(f"Processing {i+1}/{len(pdf_files)}: {pdf_path}")
        
        if i == 0 and reset_index:
            # First file - create new index
            embedder.process_pdf(pdf_path)
        else:
            # Subsequent files - append to existing index
            embedder.add_to_existing_index(pdf_path)
    
    print("\n" + "="*60)
    print("✅ DOCUMENT PROCESSING COMPLETE")
    print("="*60)
    
    # Show summary
    metadata_path = Path("vector_store/chunks.json")
    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Total documents: {len(metadata.get('source_files', []))}")
        print(f"Total chunks: {metadata.get('num_chunks', 0)}")
        print("\nDocuments in knowledge base:")
        for doc in metadata.get('source_files', []):
            print(f"  - {doc}")
    print("="*60)

def main():
    """Main function for CLI usage"""
    if len(sys.argv) < 2:
        print("""
Usage: python add_documents.py <pdf1> [pdf2] [pdf3] ... [options]

Options:
  --reset    Create new index (removes existing knowledge)
  --append   Add to existing index (default behavior)

Examples:
  # Add single document to existing knowledge
  python add_documents.py medical_guide.pdf
  
  # Add multiple documents
  python add_documents.py doc1.pdf doc2.pdf doc3.pdf
  
  # Reset and start fresh with new documents
  python add_documents.py new_doc.pdf --reset
""")
        sys.exit(1)
    
    # Parse arguments
    pdf_files = []
    reset_index = False
    
    for arg in sys.argv[1:]:
        if arg == "--reset":
            reset_index = True
        elif arg == "--append":
            reset_index = False
        elif arg.endswith('.pdf'):
            pdf_files.append(arg)
    
    if not pdf_files:
        print("❌ Error: No PDF files specified")
        sys.exit(1)
    
    # Process documents
    add_multiple_documents(pdf_files, reset_index)

if __name__ == "__main__":
    main()