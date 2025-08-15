#!/usr/bin/env python
"""
Debug script to test RAG retrieval and see what's happening
"""

import json
from rag import RAGEngine
from conversation_manager import ConversationManager

def test_rag_retrieval():
    """Test RAG retrieval to see if it's finding relevant chunks"""
    
    print("\n" + "="*60)
    print("RAG RETRIEVAL DIAGNOSTIC TEST")
    print("="*60)
    
    # Initialize RAG engine
    rag = RAGEngine()
    
    # Test queries
    test_queries = [
        "What is vitiligo?",
        "symptoms of vitiligo",
        "treatment options",
        "Hello",
        "random unrelated query xyz123"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        print("-" * 40)
        
        # Search for chunks
        results = rag.search_similar_chunks(query, top_k=3)
        
        if results:
            print(f"Found {len(results)} chunks:")
            for i, (chunk, score) in enumerate(results, 1):
                print(f"\n  Chunk {i} (Score: {score:.3f}):")
                print(f"  {chunk[:150]}..." if len(chunk) > 150 else f"  {chunk}")
        else:
            print("❌ No chunks found!")
        
        # Test with actual query
        print("\n  🤖 Generating response...")
        result = rag.query(query, response_style="brief")
        print(f"  Response: {result['response'][:200]}...")

def test_conversation_manager():
    """Test conversation manager integration"""
    
    print("\n" + "="*60)
    print("CONVERSATION MANAGER TEST")
    print("="*60)
    
    manager = ConversationManager()
    
    test_messages = [
        "Hi",
        "What is vitiligo?",
        "Tell me about symptoms"
    ]
    
    for msg in test_messages:
        print(f"\n👤 User: {msg}")
        result = manager.process_message(msg)
        print(f"   Intent: {result['intent']}")
        print(f"   Use RAG: {result['use_rag']}")
        print(f"   Style: {result['response_style']}")
        if result['quick_response']:
            print(f"   Quick Response: {result['quick_response']}")

def check_vector_store():
    """Check vector store contents"""
    
    print("\n" + "="*60)
    print("VECTOR STORE DIAGNOSTIC")
    print("="*60)
    
    try:
        with open('vector_store/chunks.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"✅ Vector store loaded successfully")
        print(f"   Total chunks: {metadata.get('num_chunks', 0)}")
        print(f"   Source files: {metadata.get('source_files', metadata.get('source_file', 'Unknown'))}")
        print(f"   Embedding model: {metadata.get('embedding_model', 'Unknown')}")
        
        # Show sample chunks
        chunks = metadata.get('chunks', [])
        if chunks:
            print(f"\n   Sample chunks (first 3):")
            for i, chunk in enumerate(chunks[:3], 1):
                print(f"\n   Chunk {i}:")
                print(f"   {chunk[:100]}..." if len(chunk) > 100 else f"   {chunk}")
    
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")

if __name__ == "__main__":
    # Run diagnostics
    check_vector_store()
    test_rag_retrieval()
    test_conversation_manager()
    
    print("\n" + "="*60)
    print("✅ DIAGNOSTIC TESTS COMPLETE")
    print("="*60)