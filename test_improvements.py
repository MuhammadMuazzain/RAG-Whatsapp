#!/usr/bin/env python
"""
Test script to demonstrate the chatbot improvements
"""

import requests
import json
import time

# API endpoint
BASE_URL = "http://localhost:8000"

def test_conversation(messages, session_id="test_session"):
    """Test a conversation flow"""
    print("\n" + "="*60)
    print("TESTING CONVERSATION FLOW")
    print("="*60)
    
    for message in messages:
        print(f"\n👤 User: {message}")
        
        # Send request
        response = requests.post(
            f"{BASE_URL}/chat",
            json={"message": message, "session_id": session_id}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"🤖 Bot: {data['response']}")
            
            # Show additional info
            if data.get('intent'):
                print(f"   [Intent: {data['intent']}, Style: {data.get('response_style', 'default')}]")
            if data.get('processing_time'):
                print(f"   [Processing time: {data['processing_time']:.2f}s]")
        else:
            print(f"❌ Error: {response.status_code}")
        
        time.sleep(1)  # Small delay between messages

def main():
    """Run test scenarios"""
    
    print("\n" + "="*60)
    print("CHATBOT IMPROVEMENT TEST SUITE")
    print("="*60)
    
    # Test 1: Greeting handling
    print("\n📋 Test 1: Greeting Handling")
    print("-" * 40)
    test_conversation([
        "Hi",
        "Hello there!"
    ], "greeting_test")
    
    # Test 2: Brief vs Detailed responses
    print("\n📋 Test 2: Response Length Control")
    print("-" * 40)
    test_conversation([
        "What is vitiligo in brief?",
        "Tell me more about the symptoms",
        "Give me a detailed explanation of the treatments"
    ], "length_test")
    
    # Test 3: Follow-up questions
    print("\n📋 Test 3: Context-Aware Follow-ups")
    print("-" * 40)
    test_conversation([
        "What causes vitiligo?",
        "Is it hereditary?",
        "What about stress?"
    ], "followup_test")
    
    # Test 4: Mixed conversation
    print("\n📋 Test 4: Natural Conversation Flow")
    print("-" * 40)
    test_conversation([
        "Hello",
        "I'd like to know about vitiligo",
        "What are the main symptoms?",
        "How is it diagnosed?",
        "Thanks for the help!",
        "Bye"
    ], "mixed_test")
    
    # Test 5: Concurrent requests
    print("\n📋 Test 5: Concurrent Request Handling")
    print("-" * 40)
    
    import threading
    
    def send_request(session_id, message):
        response = requests.post(
            f"{BASE_URL}/chat",
            json={"message": message, "session_id": session_id}
        )
        if response.status_code == 200:
            data = response.json()
            print(f"Session {session_id}: {data['response'][:50]}...")
    
    threads = []
    questions = [
        ("session1", "What is vitiligo?"),
        ("session2", "Tell me about skin conditions"),
        ("session3", "How to treat pigmentation loss?")
    ]
    
    for session_id, message in questions:
        t = threading.Thread(target=send_request, args=(session_id, message))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED")
    print("="*60)
    
    # Show session statistics
    stats_response = requests.get(f"{BASE_URL}/api/sessions")
    if stats_response.status_code == 200:
        stats = stats_response.json()
        print(f"\n📊 Session Statistics:")
        print(f"   Active sessions: {stats.get('active_sessions', 0)}")
        print(f"   Total messages: {stats.get('total_messages', 0)}")

if __name__ == "__main__":
    main()