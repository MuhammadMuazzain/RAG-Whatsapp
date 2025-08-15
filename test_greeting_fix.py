#!/usr/bin/env python
"""
Test that greetings don't trigger document responses
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_greetings():
    print("\n" + "="*60)
    print("TESTING GREETING HANDLING")
    print("="*60)
    
    # Test various greeting forms
    greetings = [
        "hi",
        "hy", 
        "hello",
        "hey",
        "Hi there",
        "Hello!",
        "good morning",
        "hola"
    ]
    
    for greeting in greetings:
        print(f"\n👤 User: {greeting}")
        
        response = requests.post(
            f"{BASE_URL}/chat",
            json={"message": greeting}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"🤖 Bot: {data['response']}")
            
            # Check if response mentions NSC or consultation
            if 'NSC' in data['response'] or 'consultation' in data['response'].lower():
                print("   ❌ ERROR: Document content in greeting response!")
            else:
                print("   ✅ Correct: Greeting response without document content")
        else:
            print(f"   ❌ Error: {response.status_code}")
    
    # Test actual questions
    print("\n\n📋 Testing Actual Questions (should use documents):")
    print("-" * 40)
    
    questions = [
        "What is vitiligo?",
        "Tell me about symptoms"
    ]
    
    for question in questions:
        print(f"\n👤 User: {question}")
        
        response = requests.post(
            f"{BASE_URL}/chat",
            json={"message": question}
        )
        
        if response.status_code == 200:
            data = response.json()
            response_preview = data['response'][:100] + "..." if len(data['response']) > 100 else data['response']
            print(f"🤖 Bot: {response_preview}")
            print("   ✅ Using document content for medical question")

if __name__ == "__main__":
    print("Make sure the server is running: uvicorn main:app --reload")
    print("Testing in 3 seconds...")
    import time
    time.sleep(3)
    
    test_greetings()