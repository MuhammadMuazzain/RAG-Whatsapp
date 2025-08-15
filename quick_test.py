#!/usr/bin/env python
"""
Quick test to verify RAG is using document content
"""

from rag import RAGEngine

def quick_test():
    print("\n" + "="*60)
    print("TESTING RAG DOCUMENT-BASED RESPONSES")
    print("="*60)
    
    rag = RAGEngine()
    
    # Test questions
    questions = [
        "What is vitiligo?",
        "What are the symptoms?",
        "How is it treated?"
    ]
    
    for q in questions:
        print(f"\n❓ Question: {q}")
        print("-" * 40)
        
        # Get chunks first to see what context is being used
        chunks = rag.search_similar_chunks(q, top_k=2)
        print(f"📚 Found {len(chunks)} relevant chunks:")
        for i, (chunk, score) in enumerate(chunks, 1):
            print(f"   Chunk {i} (score: {score:.3f}): {chunk[:80]}...")
        
        # Get response
        result = rag.query(q, response_style="brief")
        print(f"\n🤖 Response: {result['response']}")
        
        # Check if response seems to be from document
        response_lower = result['response'].lower()
        if "i don't have information" in response_lower or "knowledge base" in response_lower:
            print("   ⚠️ Warning: Model couldn't find answer in documents")
        else:
            print("   ✅ Response generated from document context")

if __name__ == "__main__":
    quick_test()