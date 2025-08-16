#!/usr/bin/env python
"""
Debug why support link isn't showing
"""

from conversation_manager import ConversationManager

def test_link_logic():
    print("\n" + "="*60)
    print("DEBUGGING SUPPORT LINK LOGIC")
    print("="*60)
    
    cm = ConversationManager()
    
    # Test messages
    test_messages = [
        "What is vitiligo?",
        "Tell me about vitiligo",
        "vitiligo symptoms",
        "How to treat vitiligo",
        "NSC trial information",
        "Free consultation at NSC"
    ]
    
    for msg in test_messages:
        print(f"\nMessage: '{msg}'")
        print("-" * 40)
        
        # Create fresh session for each test
        session = cm.get_or_create_session(f"test_{msg[:10]}")
        
        # Check detections
        is_vitiligo = cm.is_vitiligo_query(msg)
        is_nsc = cm.is_nsc_trial_query(msg)
        should_show = cm.should_show_support_link(msg, session)
        
        print(f"  Is vitiligo query: {is_vitiligo}")
        print(f"  Is NSC query: {is_nsc}")
        print(f"  Link already shown: {session.support_link_shown}")
        print(f"  Should show link: {should_show}")
        
        if should_show:
            print("  ✅ Link WILL be shown")
            session.support_link_shown = True
        else:
            print("  ❌ Link will NOT be shown")
    
    print("\n" + "="*60)
    print("Testing with same session (multiple questions):")
    print("-" * 40)
    
    session = cm.get_or_create_session("continuous_session")
    
    questions = [
        "What is vitiligo?",
        "Tell me more about symptoms",
        "What treatments are available?"
    ]
    
    for i, msg in enumerate(questions, 1):
        print(f"\nQuestion {i}: '{msg}'")
        should_show = cm.should_show_support_link(msg, session)
        print(f"  Should show link: {should_show}")
        
        if should_show:
            print("  ✅ Link shown")
            session.support_link_shown = True

if __name__ == "__main__":
    test_link_logic()