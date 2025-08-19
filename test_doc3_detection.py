"""
Test script to verify doc3 detection logic without LLM generation
"""

from rag import RAGEngine

def test_doc3_detection():
    """Test doc3 query detection based on retrieved chunks"""
    
    print("\n" + "="*60)
    print("TESTING DOC3 DETECTION LOGIC")
    print("="*60)
    
    # Initialize RAG engine
    rag = RAGEngine()
    
    # Test queries
    test_queries = [
        ("How can I sign up for the free JAK Cream trial at NSC?", True),
        ("What is the process to enroll in the ruxolitinib trial?", True),
        ("Tell me about the JAK inhibitor trial in Singapore", True),
        ("What is vitiligo?", False),
        ("What are the symptoms of vitiligo?", False),
        ("How common is vitiligo in Singapore?", False),
    ]
    
    for query, expected_doc3 in test_queries:
        print(f"\n{'-'*60}")
        print(f"Query: {query}")
        print(f"Expected Doc3: {expected_doc3}")
        
        # Search for relevant chunks
        search_results = rag.search_similar_chunks(query)
        
        # Extract just the chunk texts
        context_chunks = [chunk for chunk, score in search_results]
        
        # Check if this is a doc3 query (JAK cream trial)
        is_doc3_query = any("JAK" in chunk and "NSC" in chunk and "trial" in chunk.lower() for chunk in context_chunks)
        
        print(f"Detected as Doc3: {is_doc3_query}")
        
        # Show chunk scores
        print("\nRetrieved chunks (showing first 100 chars):")
        for i, (chunk, score) in enumerate(search_results, 1):
            # Clean chunk for display (remove special characters that might cause encoding issues)
            chunk_preview = chunk[:100].encode('ascii', 'ignore').decode('ascii')
            print(f"  {i}. Score: {score:.3f} - {chunk_preview}...")
            # Check if this chunk contains doc3 markers
            if "JAK" in chunk and "NSC" in chunk:
                print(f"     ^ Contains JAK and NSC markers")
        
        # Verify detection
        if is_doc3_query == expected_doc3:
            print("[PASS] Detection correct!")
        else:
            print("[FAIL] Detection incorrect!")
    
    print(f"\n{'='*60}")
    print("DOC3 DETECTION TEST COMPLETE")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    test_doc3_detection()