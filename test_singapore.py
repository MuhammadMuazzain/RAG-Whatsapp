#!/usr/bin/env python
"""
Test Singapore population query accuracy
"""

from rag import RAGEngine

def test_singapore_query():
    print("\n" + "="*60)
    print("TESTING SINGAPORE POPULATION QUERY")
    print("="*60)
    
    rag = RAGEngine()
    
    # Test queries about Singapore
    queries = [
        "What percentage of Singapore population has vitiligo?",
        "How much of Singapore is affected by vitiligo?",
        "What is the prevalence of vitiligo in Singapore?",
        "Singapore vitiligo statistics"
    ]
    
    print("\n📊 CORRECT ANSWER FROM DOCUMENT: 0.7% of the population in Singapore")
    print("-" * 60)
    
    for query in queries:
        print(f"\n❓ Query: {query}")
        
        # Get the chunks being used
        chunks = rag.search_similar_chunks(query, top_k=3)
        print(f"\n📚 Retrieved {len(chunks)} chunks:")
        
        # Check if any chunk contains the Singapore data
        singapore_found = False
        for i, (chunk, score) in enumerate(chunks, 1):
            if 'singapore' in chunk.lower() or '0.7' in chunk:
                print(f"   ✅ Chunk {i} contains Singapore data (score: {score:.3f})")
                singapore_found = True
                # Show the relevant part
                sg_pos = chunk.lower().find('singapore')
                if sg_pos > 0:
                    print(f"      '{chunk[max(0,sg_pos-50):sg_pos+100]}'")
            else:
                print(f"   Chunk {i} (score: {score:.3f}) - no Singapore data")
        
        if not singapore_found:
            print("   ⚠️ WARNING: Singapore data not in retrieved chunks!")
        
        # Get the response
        result = rag.query(query, response_style="brief")
        response = result['response']
        
        print(f"\n🤖 Bot Response: {response}")
        
        # Check accuracy
        if '0.7' in response:
            print("   ✅ CORRECT: Response contains accurate 0.7% figure")
        elif '1' in response or '5' in response or 'don\'t have' in response.lower():
            print("   ❌ INCORRECT: Response doesn't have the specific Singapore data")
        else:
            print("   ⚠️ Check response for accuracy")
        
        print("-" * 60)

if __name__ == "__main__":
    test_singapore_query()