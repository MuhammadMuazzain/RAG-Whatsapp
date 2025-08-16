#!/usr/bin/env python
"""
Complete flow test matching main.py logic
"""

from rag import RAGEngine
from conversation_manager import ConversationManager
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    session_id: str = None

# Initialize like in main.py
rag_engine = RAGEngine()
conversation_manager = ConversationManager()

def simulate_chat_endpoint(request: ChatRequest):
    """Simulate the exact logic from main.py /chat endpoint"""
    
    # Process message with conversation manager
    conv_result = conversation_manager.process_message(request.message, request.session_id)
    
    # Check for greeting (like in main.py)
    if conv_result['intent'] == 'greeting' and conv_result['confidence'] > 0.7:
        return {
            "response": conv_result['quick_response'] or "Hello! How can I help you today?",
            "status": "success",
            "intent": conv_result['intent']
        }
    
    # For everything else, use RAG
    response_style = conv_result['response_style']
    result = rag_engine.query(request.message, response_style=response_style)
    
    # Get raw response first
    raw_response = result.get("response", "I couldn't find an answer to your question.")
    
    # Check if we should add support group link
    should_show = conversation_manager.should_show_support_link(request.message, conv_result['context'])
    print(f"DEBUG: Should show link: {should_show}")
    
    # TEMPORARY: Force link for vitiligo queries
    if 'vitiligo' in request.message.lower() and not conv_result['context'].support_link_shown:
        should_show = True
        print("DEBUG: FORCING link to show for vitiligo query")
    
    if should_show:
        # Add link to raw response
        raw_response += "\n\nFor community support and to connect with others, visit: vitiligosupportgroup.com"
        conv_result['context'].support_link_shown = True
        print(f"DEBUG: Link added. Response length: {len(raw_response)}")
    
    # Now format (or not format to avoid cutting)
    response_text = raw_response
    
    return {
        "response": response_text,
        "status": "success",
        "intent": conv_result['intent'],
        "response_style": response_style
    }

# Test
print("\n" + "="*60)
print("FULL FLOW TEST")
print("="*60)

test_messages = [
    "What is vitiligo?",
    "Tell me about symptoms",
    "How to sign up for NSC trial?"
]

for msg in test_messages:
    print(f"\nTesting: '{msg}'")
    print("-" * 40)
    
    request = ChatRequest(message=msg, session_id="test_session")
    response = simulate_chat_endpoint(request)
    
    print(f"Response length: {len(response['response'])} chars")
    print(f"Has link: {'vitiligosupportgroup.com' in response['response']}")
    
    if len(response['response']) > 200:
        print(f"Response ends with: ...{response['response'][-150:]}")
    else:
        print(f"Response: {response['response']}")