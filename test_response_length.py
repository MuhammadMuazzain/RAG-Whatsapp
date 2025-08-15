#!/usr/bin/env python
"""
Test response length control
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def count_lines_and_words(text):
    """Count lines and words in text"""
    lines = text.strip().split('\n')
    words = len(text.split())
    chars = len(text)
    return len(lines), words, chars

def test_response_lengths():
    print("\n" + "="*60)
    print("TESTING RESPONSE LENGTH CONTROL")
    print("="*60)
    
    # Test questions that should get brief responses
    test_cases = [
        ("What is vitiligo?", "brief"),
        ("I want to know about vitiligo", "brief"),
        ("Tell me about vitiligo symptoms", "brief"),
        ("How is vitiligo treated?", "brief"),
        ("Explain vitiligo in detail", "detailed"),
        ("Give me comprehensive information about vitiligo treatments", "detailed")
    ]
    
    for question, expected_style in test_cases:
        print(f"\n{'='*60}")
        print(f"👤 User: {question}")
        print(f"Expected style: {expected_style}")
        print("-" * 40)
        
        response = requests.post(
            f"{BASE_URL}/chat",
            json={"message": question}
        )
        
        if response.status_code == 200:
            data = response.json()
            bot_response = data['response']
            lines, words, chars = count_lines_and_words(bot_response)
            
            print(f"🤖 Bot Response:")
            print(bot_response)
            print("-" * 40)
            print(f"📊 Stats:")
            print(f"   Lines: {lines}")
            print(f"   Words: {words}")
            print(f"   Characters: {chars}")
            print(f"   Style used: {data.get('response_style', 'unknown')}")
            
            # Check if response meets expectations
            if expected_style == "brief":
                if words > 50:
                    print("   ❌ TOO LONG for brief response!")
                else:
                    print("   ✅ Good length for brief response")
            elif expected_style == "detailed":
                if words < 30:
                    print("   ❌ Too short for detailed response")
                else:
                    print("   ✅ Good length for detailed response")
        else:
            print(f"❌ Error: {response.status_code}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("Brief responses should be: 1-2 sentences, <50 words")
    print("Moderate responses should be: 2-3 sentences, <75 words")
    print("Detailed responses should be: 4-5 sentences, <150 words")

if __name__ == "__main__":
    print("Make sure server is running: uvicorn main:app --reload")
    print("Testing in 3 seconds...")
    import time
    time.sleep(3)
    
    test_response_lengths()