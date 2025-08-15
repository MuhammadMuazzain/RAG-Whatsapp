"""
Conversation Manager for intelligent chat handling
Includes intent classification, context management, and response control
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import hashlib

logger = logging.getLogger(__name__)

class IntentClassifier:
    """Classify user intents for appropriate response handling"""
    
    def __init__(self):
        # Intent patterns
        self.patterns = {
            "greeting": [
                r'\b(hi|hello|hey|greetings?|good\s*(morning|afternoon|evening|day))\b',
                r'\b(what\'?s\s*up|howdy|sup)\b',
                r'^(hi|hello|hey)[\s!?]*$'
            ],
            "farewell": [
                r'\b(bye|goodbye|see\s*you|farewell|take\s*care|good\s*night)\b',
                r'\b(thanks?|thank\s*you|appreciate|cheers)\b.*\b(bye|goodbye)?\b',
                r'^(bye|goodbye|thanks?)[\s!?]*$'
            ],
            "help_request": [
                r'\b(help|assist|support|guide|how\s*to)\b',
                r'\b(can\s*you|could\s*you|would\s*you)\s*(help|assist|explain)\b',
                r'\bwhat\s*can\s*you\s*do\b'
            ],
            "detail_request": [
                r'\b(tell\s*me\s*more|more\s*details?|elaborate|explain\s*further)\b',
                r'\b(what\s*about|how\s*about|and)\b.*\?$',
                r'\b(continue|go\s*on|keep\s*going)\b'
            ],
            "brief_request": [
                r'\b(brief|summary|summarize|short|quick)\b',
                r'\b(in\s*short|briefly|tldr|tl;?dr)\b',
                r'\bjust\s*tell\s*me\b'
            ],
            "yes_no_question": [
                r'^(is|are|was|were|did|does|do|can|could|should|would|will)\b',
                r'\?$'  # Ends with question mark
            ],
            "definition_question": [
                r'\bwhat\s*(is|are)\b',
                r'\bdefine\b',
                r'\bmean(s|ing)?\b'
            ],
            "symptom_question": [
                r'\b(symptom|sign|indicator|manifestation)s?\b',
                r'\bhow\s*(do|does|can)\s*I\s*know\b',
                r'\blook\s*like\b'
            ],
            "treatment_question": [
                r'\b(treat|cure|remedy|medicine|therapy|management)\b',
                r'\bhow\s*to\s*(treat|cure|manage)\b',
                r'\bwhat\s*helps?\b'
            ],
            "cause_question": [
                r'\b(cause|reason|why|trigger|factor)s?\b',
                r'\bwhat\s*causes?\b',
                r'\bwhy\s*does?\b'
            ]
        }
        
        # Compile patterns
        self.compiled_patterns = {}
        for intent, patterns in self.patterns.items():
            self.compiled_patterns[intent] = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def classify(self, message: str) -> Tuple[str, float]:
        """
        Classify the intent of a message
        Returns: (intent_type, confidence_score)
        """
        message = message.strip()
        
        # Check each intent
        scores = {}
        for intent, patterns in self.compiled_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern.search(message):
                    score += 1
            if score > 0:
                scores[intent] = score / len(patterns)
        
        # Special handling for very short messages and common greetings
        message_lower = message.lower().strip()
        
        # Check for common greeting variations (including typos)
        greeting_words = ['hi', 'hy', 'hello', 'hey', 'helo', 'hola', 'hai', 'hii', 'hiii', 
                         'greetings', 'good morning', 'good afternoon', 'good evening', 
                         'morning', 'afternoon', 'evening', 'howdy', 'sup', "what's up"]
        
        # Exact match or very close match for greetings
        if message_lower in greeting_words:
            return ("greeting", 0.99)
        
        # Check if message is primarily a greeting (for slightly longer messages)
        if len(message.split()) <= 3:
            if any(word in message_lower for word in greeting_words):
                return ("greeting", 0.95)
            if any(word in message_lower for word in ['bye', 'goodbye', 'thanks', 'thank you']):
                return ("farewell", 0.95)
        
        # Return highest scoring intent
        if scores:
            best_intent = max(scores.items(), key=lambda x: x[1])
            return best_intent
        
        # Default to question if no clear intent
        return ("question", 0.5)
    
    def get_response_style(self, message: str) -> str:
        """Determine if user wants brief or detailed response - DEFAULT TO BRIEF"""
        intent, confidence = self.classify(message)
        
        # Check for explicit length requests
        message_lower = message.lower()
        if any(word in message_lower for word in ['detail', 'elaborate', 'explain fully', 'tell me everything', 'comprehensive']):
            return "detailed"
        elif any(word in message_lower for word in ['brief', 'summary', 'short', 'quick', 'tldr']):
            return "brief"
        
        # Intent-based defaults - PREFER BRIEF
        if intent == "detail_request":
            return "detailed"
        elif intent in ["definition_question", "yes_no_question"]:
            return "brief"
        elif intent in ["symptom_question", "treatment_question", "cause_question"]:
            return "brief"  # Changed from moderate to brief
        
        # Check message length as hint
        word_count = len(message.split())
        if word_count < 8:  # Most questions are short
            return "brief"
        elif word_count > 20:  # Very long questions might want detail
            return "moderate"
        
        return "brief"  # DEFAULT TO BRIEF instead of moderate


class ConversationContext:
    """Manage conversation context and history"""
    
    def __init__(self, session_id: str, max_history: int = 10):
        self.session_id = session_id
        self.history = deque(maxlen=max_history)
        self.current_topic = None
        self.last_intent = None
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.user_preferences = {
            "response_length": "moderate",
            "detail_level": "auto"
        }
    
    def add_message(self, role: str, content: str, intent: str = None):
        """Add a message to conversation history"""
        self.history.append({
            "role": role,
            "content": content,
            "intent": intent,
            "timestamp": datetime.now().isoformat()
        })
        self.last_activity = datetime.now()
        if intent:
            self.last_intent = intent
    
    def get_context_summary(self) -> str:
        """Get a summary of recent conversation context"""
        if not self.history:
            return ""
        
        recent = list(self.history)[-3:]  # Last 3 exchanges
        summary = []
        for msg in recent:
            if msg["role"] == "user":
                summary.append(f"User asked: {msg['content'][:50]}...")
        
        return " ".join(summary)
    
    def is_follow_up(self, message: str) -> bool:
        """Check if message is a follow-up to previous topic"""
        follow_up_indicators = [
            "more", "else", "also", "and", "what about",
            "tell me more", "continue", "go on", "that"
        ]
        
        message_lower = message.lower()
        
        # Check for pronouns referring to previous topic
        if any(word in message_lower for word in ["it", "this", "that", "they"]):
            if len(message.split()) < 6:  # Short message with pronoun
                return True
        
        # Check for follow-up indicators
        for indicator in follow_up_indicators:
            if indicator in message_lower:
                return True
        
        return False
    
    def should_expire(self, timeout_minutes: int = 30) -> bool:
        """Check if conversation context should expire"""
        return datetime.now() - self.last_activity > timedelta(minutes=timeout_minutes)


class ConversationManager:
    """Main conversation manager integrating all components"""
    
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.sessions = {}  # session_id -> ConversationContext
        self.greeting_responses = [
            "Hello! I'm here to help answer your medical questions. What would you like to know?",
            "Hi there! How can I assist you with your health-related queries today?",
            "Greetings! I'm ready to help with any medical information you need. What's on your mind?",
            "Hello! Feel free to ask me anything about health and medical topics.",
        ]
        self.farewell_responses = [
            "Goodbye! Take care and stay healthy!",
            "Thank you for chatting. Have a great day!",
            "Bye! Feel free to come back if you have more questions.",
            "Take care! Wishing you good health!",
        ]
    
    def get_or_create_session(self, session_id: str = None) -> ConversationContext:
        """Get existing session or create new one"""
        if not session_id:
            # Generate session ID from timestamp
            session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12]
        
        # Clean expired sessions
        expired = [sid for sid, ctx in self.sessions.items() if ctx.should_expire()]
        for sid in expired:
            del self.sessions[sid]
        
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationContext(session_id)
        
        return self.sessions[session_id]
    
    def process_message(self, message: str, session_id: str = None) -> Dict:
        """
        Process a message with full context awareness
        Returns: {
            'intent': str,
            'response_style': str,
            'use_rag': bool,
            'quick_response': str (if applicable),
            'context': ConversationContext
        }
        """
        context = self.get_or_create_session(session_id)
        intent, confidence = self.intent_classifier.classify(message)
        response_style = self.intent_classifier.get_response_style(message)
        
        # Add to history
        context.add_message("user", message, intent)
        
        result = {
            'intent': intent,
            'confidence': confidence,
            'response_style': response_style,
            'use_rag': True,
            'quick_response': None,
            'context': context,
            'is_follow_up': context.is_follow_up(message)
        }
        
        # Handle greetings
        if intent == "greeting" and confidence > 0.7:
            import random
            result['quick_response'] = random.choice(self.greeting_responses)
            result['use_rag'] = False
        
        # Handle farewells
        elif intent == "farewell" and confidence > 0.7:
            import random
            result['quick_response'] = random.choice(self.farewell_responses)
            result['use_rag'] = False
        
        # Handle help requests
        elif intent == "help_request" and confidence > 0.7:
            result['quick_response'] = (
                "I can help you with medical information! You can ask me about:\n"
                "• Disease definitions and explanations\n"
                "• Symptoms and signs\n"
                "• Treatment options\n"
                "• Causes and risk factors\n"
                "What would you like to know?"
            )
            result['use_rag'] = False
        
        # For follow-ups, use previous context
        elif result['is_follow_up']:
            result['use_previous_context'] = True
        
        return result
    
    def format_response(self, response: str, style: str = "brief") -> str:
        """Format response based on requested style - AGGRESSIVE SHORTENING"""
        if not response:
            return response
        
        # Split into sentences
        sentences = response.split('. ')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if style == "brief":
            # Return ONLY first 1-2 sentences, max 150 chars
            result = sentences[0] if sentences else response
            if len(result) > 150:
                result = result[:147] + "..."
            return result
        
        elif style == "moderate":
            # Return 2-3 sentences max, under 250 chars
            result = '. '.join(sentences[:2]) if len(sentences) > 1 else sentences[0]
            if len(result) > 250:
                result = result[:247] + "..."
            return result + '.'
        
        else:  # detailed
            # Even detailed should be reasonable - max 4 sentences
            if len(sentences) > 4:
                return '. '.join(sentences[:4]) + '.'
            return response
    
    def get_session_stats(self) -> Dict:
        """Get statistics about active sessions"""
        return {
            "active_sessions": len(self.sessions),
            "total_messages": sum(len(s.history) for s in self.sessions.values()),
            "sessions": [
                {
                    "id": s.session_id,
                    "messages": len(s.history),
                    "last_activity": s.last_activity.isoformat()
                }
                for s in self.sessions.values()
            ]
        }


# Example usage
if __name__ == "__main__":
    manager = ConversationManager()
    
    # Test messages
    test_messages = [
        "Hi",
        "What is vitiligo?",
        "Tell me more about the symptoms",
        "How is it treated?",
        "Thanks, bye!"
    ]
    
    session_id = "test_session"
    
    for msg in test_messages:
        print(f"\nUser: {msg}")
        result = manager.process_message(msg, session_id)
        print(f"Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
        print(f"Style: {result['response_style']}")
        print(f"Use RAG: {result['use_rag']}")
        if result['quick_response']:
            print(f"Response: {result['quick_response']}")