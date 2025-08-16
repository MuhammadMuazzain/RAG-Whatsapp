#!/usr/bin/env python
"""
Test support group link functionality
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_support_link():
    print("\n" + "="*60)
    print("TESTING SUPPORT GROUP LINK FUNCTIONALITY")
    print("="*60)
    
    # Test scenarios
    test_cases = [
        {
            "name": "First vitiligo question (should show link)",
            "session": "test_session_1",
            "messages": [
                "What is vitiligo?"
            ],
            "expect_link": [True]
        },
        {
            "name": "Greeting then vitiligo question (should show link)",
            "session": "test_session_2",
            "messages": [
                "Hi",
                "Tell me about vitiligo symptoms"
            ],
            "expect_link": [False, True]
        },
        {
            "name": "NSC trial question (NO link)",
            "session": "test_session_3",
            "messages": [
                "How to sign up for free trial at NSC?"
            ],
            "expect_link": [False]
        },
        {
            "name": "Multiple questions (link only once)",
            "session": "test_session_4",
            "messages": [
                "What causes vitiligo?",
                "What are the symptoms?",
                "How is it treated?"
            ],
            "expect_link": [True, False, False]
        },
        {
            "name": "Mixed NSC and vitiligo (link for vitiligo only)",
            "session": "test_session_5",
            "messages": [
                "Tell me about NSC trial",
                "What is vitiligo?",
                "More about the free consultation"
            ],
            "expect_link": [False, True, False]
        }
    ]
    
    for test in test_cases:
        print(f"\n📋 Test: {test['name']}")
        print("-" * 40)
        
        for i, message in enumerate(test['messages']):
            print(f"\n👤 User: {message}")
            
            response = requests.post(
                f"{BASE_URL}/chat",
                json={
                    "message": message,
                    "session_id": test['session']
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                bot_response = data['response']
                
                # Check if link is in response
                has_link = "vitiligosupportgroup.com" in bot_response
                expected = test['expect_link'][i]
                
                # Show response (truncated)
                if len(bot_response) > 200:
                    display_response = bot_response[:200] + "..."
                else:
                    display_response = bot_response
                
                print(f"🤖 Bot: {display_response}")
                
                # Verify expectation
                if has_link == expected:
                    if has_link:
                        print("   ✅ Link shown as expected")
                    else:
                        print("   ✅ No link as expected")
                else:
                    if expected and not has_link:
                        print("   ❌ ERROR: Link should be shown but wasn't")
                    else:
                        print("   ❌ ERROR: Link shown when it shouldn't be")
            else:
                print(f"   ❌ Error: {response.status_code}")
        
        time.sleep(0.5)  # Small delay between test cases
    
    print("\n" + "="*60)
    print("✅ SUPPORT LINK TESTS COMPLETE")
    print("="*60)
    print("\nExpected behavior:")
    print("1. Link shows ONCE per conversation session")
    print("2. Link shows for general vitiligo questions")
    print("3. Link does NOT show for NSC/trial questions")
    print("4. Link does NOT repeat in same session")

if __name__ == "__main__":
    print("Make sure server is running: uvicorn main:app --reload")
    print("Testing in 3 seconds...")
    time.sleep(3)
    
    test_support_link()