#!/usr/bin/env python
"""
Test support link scenarios
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_scenario(session_id, messages):
    """Test a sequence of messages"""
    print(f"\n{'='*60}")
    print(f"Testing session: {session_id}")
    print('='*60)
    
    for i, msg in enumerate(messages, 1):
        print(f"\nMessage {i}: '{msg}'")
        print("-" * 40)
        
        response = requests.post(
            f"{BASE_URL}/chat",
            json={
                "message": msg,
                "session_id": session_id
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            response_text = data.get('response', '')
            
            # Check for link
            has_link = "vitiligosupportgroup.com" in response_text
            print(f"Has link: {has_link}")
            
            # Show last part of response
            if has_link:
                print("Response ends with:", response_text[-150:])
            else:
                print("Response snippet:", response_text[:100] + "...")
        else:
            print(f"Error: {response.status_code}")
        
        time.sleep(0.5)  # Small delay between messages

# Test 1: Greeting then vitiligo question
test_scenario("test_greeting_first", [
    "Hi",
    "What is vitiligo?",
    "Tell me more about the symptoms"
])

# Test 2: Direct vitiligo question
test_scenario("test_direct_vitiligo", [
    "What is vitiligo?",
    "How is it treated?",
    "What are the symptoms?"
])

# Test 3: NSC trial questions (should NOT show link)
test_scenario("test_nsc_trial", [
    "Hi",
    "How can I sign up for the NSC trial?",
    "Tell me about the free consultation"
])

# Test 4: Mixed questions
test_scenario("test_mixed", [
    "What is vitiligo?",  # Should show link here
    "How to sign up for NSC trial?",  # Should NOT show link
    "Tell me about symptoms"  # Should NOT show link (already shown)
])

print("\n" + "="*60)
print("SUMMARY: Link should appear ONCE per session for vitiligo")
print("questions, but NOT for NSC/trial questions")
print("="*60)