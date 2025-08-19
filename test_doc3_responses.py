"""
Test script to verify doc3 detailed response functionality
"""

import json
import time
from rag import RAGEngine

def test_doc3_queries():
    """Test various queries that should trigger doc3 detailed responses"""
    
    print("\n" + "="*60)
    print("TESTING DOC3 DETAILED RESPONSES")
    print("="*60)
    
    # Initialize RAG engine
    rag = RAGEngine()
    
    # Test queries - some should trigger doc3, others should not
    test_queries = [
        # Doc3-related queries (should trigger detailed responses)
        "How can I sign up for the free JAK Cream trial at NSC?",
        "What is the process to enroll in the ruxolitinib trial?",
        "Tell me about the JAK inhibitor trial in Singapore",
        "How to get the free JAK cream treatment?",
        "What are the steps to join the NSC vitiligo trial?",
        
        # Non-doc3 queries (should use normal response style)
        "What is vitiligo?",
        "What are the symptoms of vitiligo?",
        "How common is vitiligo in Singapore?",
        "What causes vitiligo?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'-'*60}")
        print(f"Query {i}: {query}")
        print(f"{'-'*60}")
        
        # Process query
        start_time = time.time()
        result = rag.query(query)
        
        # Extract response details
        response = result.get("response", "")
        is_doc3 = result.get("is_doc3_query", False)
        processing_time = result.get("processing_time", 0)
        
        # Display results
        print(f"Is Doc3 Query: {is_doc3}")
        print(f"Response Length: {len(response)} characters")
        print(f"Processing Time: {processing_time:.2f} seconds")
        print(f"\nResponse Preview (first 500 chars):")
        print(response[:500] + ("..." if len(response) > 500 else ""))
        
        # Check if doc3 queries are getting detailed responses
        if "JAK" in query or "NSC" in query or "trial" in query:
            if is_doc3 and len(response) > 600:
                print("✅ Doc3 query correctly identified and detailed response provided")
            else:
                print("⚠️ Doc3 query but response may not be detailed enough")
        else:
            if not is_doc3 and len(response) < 800:
                print("✅ Non-doc3 query correctly handled with standard response")
            else:
                print("⚠️ Non-doc3 query but response seems too detailed")
        
        # Brief pause between queries
        time.sleep(1)
    
    print(f"\n{'='*60}")
    print("DOC3 TESTING COMPLETE")
    print(f"{'='*60}\n")

def test_streaming_doc3():
    """Test streaming responses for doc3 queries"""
    
    print("\n" + "="*60)
    print("TESTING DOC3 STREAMING RESPONSES")
    print("="*60)
    
    rag = RAGEngine()
    
    test_query = "How can I sign up for the free JAK Cream trial at NSC?"
    print(f"\nStreaming Query: {test_query}")
    print("-"*60)
    
    response_chunks = []
    is_doc3 = False
    
    # Collect streaming response
    for chunk in rag.query_with_stream(test_query):
        if chunk.startswith("data: "):
            try:
                data = json.loads(chunk[6:chunk.find('\n')])
                if data.get("content"):
                    response_chunks.append(data["content"])
                if data.get("done"):
                    is_doc3 = data.get("is_doc3_query", False)
            except:
                pass
    
    full_response = "".join(response_chunks)
    
    print(f"Is Doc3 Query: {is_doc3}")
    print(f"Response Length: {len(full_response)} characters")
    print(f"\nStreaming Response Preview (first 500 chars):")
    print(full_response[:500] + ("..." if len(full_response) > 500 else ""))
    
    if is_doc3 and len(full_response) > 600:
        print("\n✅ Doc3 streaming query correctly identified and detailed response provided")
    else:
        print("\n⚠️ Issue with doc3 streaming response")
    
    print(f"\n{'='*60}")
    print("STREAMING TEST COMPLETE")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Run tests
    test_doc3_queries()
    test_streaming_doc3()