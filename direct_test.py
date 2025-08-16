#!/usr/bin/env python
"""
Direct test without server
"""

from rag import RAGEngine
from conversation_manager import ConversationManager

# Initialize
rag = RAGEngine()
cm = ConversationManager()

# Test message
message = "What is vitiligo?"
session_id = "direct_test"

print(f"\nTesting: '{message}'")
print("-" * 50)

# Process with conversation manager
conv_result = cm.process_message(message, session_id)
print(f"Intent: {conv_result['intent']}")
print(f"Use RAG: {conv_result['use_rag']}")

# Get RAG response
result = rag.query(message)
response = result.get("response", "No response")

print(f"\nRAG Response (first 200 chars):")
print(response[:200])

# Check if should show link
should_show = cm.should_show_support_link(message, conv_result['context'])
print(f"\nShould show link: {should_show}")
print(f"Is vitiligo query: {cm.is_vitiligo_query(message)}")
print(f"Is NSC query: {cm.is_nsc_trial_query(message)}")
print(f"Link already shown: {conv_result['context'].support_link_shown}")

# Add link if should show
if should_show:
    response += cm.support_link
    print("\n[ADDED] Link added to response")
else:
    print("\n[NOT ADDED] Link NOT added")

print(f"\nFinal response ends with:")
print(response[-200:])

# Verify link is in response
if "vitiligosupportgroup.com" in response:
    print("\n[SUCCESS] Link is in final response!")
else:
    print("\n[FAILED] Link is NOT in final response!")