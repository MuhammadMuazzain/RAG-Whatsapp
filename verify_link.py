#!/usr/bin/env python
"""
Verify support link is working
"""

import requests
import json

BASE_URL = "http://localhost:8000"

print("\nTesting support link...")
print("-" * 40)

# Test 1: Fresh session, vitiligo question
response = requests.post(
    f"{BASE_URL}/chat",
    json={
        "message": "What is vitiligo?",
        "session_id": "fresh_test_session_999"
    }
)

if response.status_code == 200:
    data = response.json()
    response_text = data.get('response', '')
    
    print(f"Response length: {len(response_text)} chars")
    print(f"\nResponse ends with:")
    print(response_text[-200:] if len(response_text) > 200 else response_text)
    
    if "vitiligosupportgroup.com" in response_text:
        print("\n[SUCCESS] Link is present!")
    else:
        print("\n[FAILED] Link is NOT present")
        print("\nChecking session info:")
        print(f"Session ID: {data.get('session_id')}")
        print(f"Intent: {data.get('intent')}")
else:
    print(f"Error: {response.status_code}")

# Test 2: Check if it's in streaming endpoint
print("\n\nTesting streaming endpoint...")
print("-" * 40)

response = requests.post(
    f"{BASE_URL}/chat/stream",
    json={
        "message": "Tell me about vitiligo",
        "session_id": "stream_test_999"
    },
    stream=True
)

if response.status_code == 200:
    full_response = ""
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            full_response += line_str
    
    if "vitiligosupportgroup.com" in full_response:
        print("[SUCCESS] Link is present in streaming!")
    else:
        print("[FAILED] Link is NOT present in streaming")
        print("Last 200 chars:", full_response[-200:] if len(full_response) > 200 else full_response)