import json
import logging
import hashlib
import hmac
from datetime import datetime
from typing import Dict, Optional
import requests
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel
import uvicorn
import asyncio
from typing import AsyncGenerator

from rag import RAGEngine
from conversation_manager import ConversationManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    logger.error("config.json not found. Please create it with your AI.Sensy credentials.")
    config = {}

app = FastAPI(
    title="RAG WhatsApp Bot",
    description="WhatsApp chatbot powered by RAG and Mistral-7B",
    version="1.0.0"
)


rag_engine = None
conversation_manager = None


class WhatsAppMessage(BaseModel):
    """Incoming WhatsApp message from AI.Sensy"""
    message_id: str
    from_number: str
    to_number: str
    text: str
    timestamp: Optional[str] = None
    
class WebhookRequest(BaseModel):
    """AI.Sensy webhook payload"""
    event: str
    data: Dict

@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine and conversation manager on startup"""
    global rag_engine, conversation_manager
    try:
        # Initialize RAG engine
        rag_engine = RAGEngine()
        logger.info("RAG engine initialized successfully")
        
        # Initialize conversation manager
        conversation_manager = ConversationManager()
        logger.info("Conversation manager initialized successfully")
        
        # Warm up the engine
        rag_engine.warm_up()
        
        # Health check
        health = rag_engine.health_check()
        logger.info(f"RAG health check: {health}")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        
@app.get("/")
async def root():
    """Serve the chat interface HTML"""
    html_path = Path("templates/chat.html")
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        return JSONResponse(
            {"error": "Chat interface not found"},
            status_code=404
        )

class ChatRequest(BaseModel):
    """Chat request from web interface"""
    message: str
    session_id: Optional[str] = None


# @app.post("/chat/stream")
# async def chat_stream_endpoint(request: ChatRequest):
#     def event_stream():
#         for chunk in rag_engine.query_stream(request.message):
#             yield chunk

#     return StreamingResponse(event_stream(), media_type="text/plain")




@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Handle streaming chat messages with SSE and conversation context"""
    if not rag_engine or not conversation_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")

    async def generate() -> AsyncGenerator[str, None]:
        try:
            # Process message with conversation manager
            result = conversation_manager.process_message(request.message, request.session_id)
            
            # Handle greetings and farewells - be more aggressive
            if result['intent'] == 'greeting' and result['confidence'] > 0.7:
                response = result['quick_response'] or "Hello! How can I help you today?"
                yield f"data: {{\"content\": \"{response}\"}}\n\n"
                yield f"data: {{\"done\": true}}\n\n"
                return
            elif result['intent'] == 'farewell' and result['confidence'] > 0.7:
                response = result['quick_response'] or "Goodbye! Take care!"
                yield f"data: {{\"content\": \"{response}\"}}\n\n"
                yield f"data: {{\"done\": true}}\n\n"
                return
            
            # For everything else, use RAG
            response_style = result['response_style']
            
            # Track if we should show support link
            should_show_link = conversation_manager.should_show_support_link(request.message, result['context'])
            
            # Check if this is a doc3 query (we'll detect it from the first chunk)
            is_doc3_query = False
            
            # Collect the response to control length if link needs to be added
            if should_show_link:
                # Collect full response first to truncate if needed
                full_response = ""
                max_length_without_link = 400  # Same as non-streaming endpoint
                
                for chunk in rag_engine.query_with_stream(request.message, response_style=response_style):
                    # Extract content from SSE format
                    if chunk.startswith("data: "):
                        try:
                            data = json.loads(chunk[6:chunk.find('\n')])
                            if data.get("content"):
                                full_response += data["content"]
                        except:
                            pass
                
                # Truncate if needed to fit link
                if len(full_response) > max_length_without_link:
                    truncated = full_response[:max_length_without_link]
                    last_period = truncated.rfind('.')
                    if last_period > 0:
                        full_response = truncated[:last_period + 1]
                    else:
                        full_response = truncated + "..."
                
                # Add the link to the response
                link_message = "\n\nFor community support and to connect with others, visit: vitiligosupportgroup.com"
                full_response_with_link = full_response + link_message
                
                # Stream the combined response word by word for smooth display
                words = full_response_with_link.split(' ')
                for i, word in enumerate(words):
                    if i > 0:
                        yield f"data: {{\"content\": \" {word}\"}}\n\n"
                    else:
                        yield f"data: {{\"content\": \"{word}\"}}\n\n"
                    await asyncio.sleep(0.01)  # Small delay for smooth streaming
                
                result['context'].support_link_shown = True
                logger.info(f"Support link added to streaming response. Total length: {len(full_response_with_link)}")
                
                # Send done signal
                yield f"data: {{\"done\": true}}\n\n"
            else:
                # Normal streaming without link
                for chunk in rag_engine.query_with_stream(request.message, response_style=response_style):
                    yield chunk
                    await asyncio.sleep(0)  # Allow other coroutines to run
                
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable proxy buffering
        }
    )

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Handle non-streaming chat messages with conversation context"""
    if not rag_engine or not conversation_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        # Process message with conversation manager
        conv_result = conversation_manager.process_message(request.message, request.session_id)
        
        # Handle greetings and farewells - be more aggressive about catching them
        if conv_result['intent'] == 'greeting' and conv_result['confidence'] > 0.7:
            # Always return greeting response, don't use RAG
            return JSONResponse({
                "response": conv_result['quick_response'] or "Hello! How can I help you today?",
                "status": "success",
                "intent": conv_result['intent'],
                "session_id": conv_result['context'].session_id
            })
        elif conv_result['intent'] == 'farewell' and conv_result['confidence'] > 0.7:
            # Always return farewell response, don't use RAG
            return JSONResponse({
                "response": conv_result['quick_response'] or "Goodbye! Take care!",
                "status": "success",
                "intent": conv_result['intent'],
                "session_id": conv_result['context'].session_id
            })
        
        # For everything else, ALWAYS use RAG to ensure document-based responses
        response_style = conv_result['response_style']
        result = rag_engine.query(request.message, response_style=response_style)
        
        # Get raw response first
        raw_response = result.get("response", "I couldn't find an answer to your question.")
        
        # Check if we should add support group link BEFORE formatting
        should_show = conversation_manager.should_show_support_link(request.message, conv_result['context'])
        logger.info(f"Should show support link: {should_show} for message: {request.message[:50]}")
        
        # TEMPORARY: Force link to show for vitiligo queries for debugging
        if 'vitiligo' in request.message.lower() and not conv_result['context'].support_link_shown:
            should_show = True
            logger.info("FORCING link to show for vitiligo query")
        
        # Calculate space for link
        link_text = "\n\nFor community support and to connect with others, visit: vitiligosupportgroup.com"
        
        if should_show:
            # IMPORTANT: Truncate response to ensure link fits
            # Reserve space for the link (about 90 characters)
            max_length_without_link = 400  # Adjust this based on your needs
            
            if len(raw_response) > max_length_without_link:
                # Find last complete sentence before limit
                truncated = raw_response[:max_length_without_link]
                last_period = truncated.rfind('.')
                if last_period > 0:
                    raw_response = truncated[:last_period + 1]
                else:
                    raw_response = truncated + "..."
            
            # Now add link - it will definitely fit
            raw_response += link_text
            conv_result['context'].support_link_shown = True
            logger.info(f"Support link added. Final length: {len(raw_response)}")
        
        response_text = raw_response
        
        # Add to conversation history
        conv_result['context'].add_message("assistant", response_text)
        
        return JSONResponse({
            "response": response_text,
            "status": "success",
            "intent": conv_result['intent'],
            "response_style": response_style,
            "session_id": conv_result['context'].session_id,
            "processing_time": result.get("processing_time", 0)
        })
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        return JSONResponse(
            {"response": "I encountered an error processing your message. Please try again.",
             "status": "error",
             "error": str(e)},
            status_code=500
        )
@app.post("/chat/clear-cache")
async def clear_cache():
    """Clear the RAG engine cache"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    try:
        rag_engine.clear_cache()
        return JSONResponse({"status": "success", "message": "Cache cleared"})
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return JSONResponse(
            {"status": "error", "error": str(e)},
            status_code=500
        )

class PerformanceModeRequest(BaseModel):
    """Performance mode change request"""
    mode: str  # "speed", "quality", or "balanced"

@app.post("/chat/performance-mode")
async def set_performance_mode(request: PerformanceModeRequest):
    """Change the RAG engine performance mode"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    try:
        rag_engine.set_performance_mode(request.mode)
        return JSONResponse({
            "status": "success", 
            "message": f"Performance mode set to: {request.mode}"
        })
    except Exception as e:
        logger.error(f"Error setting performance mode: {e}")
        return JSONResponse(
            {"status": "error", "error": str(e)},
            status_code=500
        )

@app.get("/api/health")
async def api_health():
    """API health check endpoint"""
    return {
        "status": "running",
        "service": "RAG WhatsApp Bot",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/test-link")
async def test_link():
    """Test endpoint to verify link functionality"""
    if not rag_engine or not conversation_manager:
        return {"error": "Services not initialized"}
    
    message = "What is vitiligo?"
    session_id = "test_endpoint"
    
    # Process message
    conv_result = conversation_manager.process_message(message, session_id)
    
    # Get RAG response
    result = rag_engine.query(message)
    response = result.get("response", "No response")
    
    # Check if should show link
    should_show = conversation_manager.should_show_support_link(message, conv_result['context'])
    
    # Add link if needed
    if should_show:
        response += conversation_manager.support_link
        conv_result['context'].support_link_shown = True
    
    return {
        "message": message,
        "should_show_link": should_show,
        "link_in_response": "vitiligosupportgroup.com" in response,
        "response_preview": response[-200:] if len(response) > 200 else response
    }

@app.get("/api/sessions")
async def get_sessions():
    """Get active conversation sessions"""
    if not conversation_manager:
        return JSONResponse({"error": "Conversation manager not initialized"}, status_code=503)
    
    return JSONResponse(conversation_manager.get_session_stats())

@app.get("/health")
async def health_check():
    """Detailed health check"""
    if rag_engine:
        rag_health = rag_engine.health_check()
    else:
        rag_health = {"status": "not initialized"}
    
    return {
        "api": "healthy",
        "rag": rag_health,
        "config": {
            "sensy_api_key": "configured" if config.get("sensy_api_key") else "missing",
            "webhook_secret": "configured" if config.get("webhook_secret") else "missing"
        }
    }

def verify_webhook_signature(request_body: bytes, signature: str) -> bool:
    """Verify webhook signature from AI.Sensy"""
    if not config.get("webhook_secret"):
        logger.warning("Webhook secret not configured, skipping verification")
        return True
    
    expected_signature = hmac.new(
        config["webhook_secret"].encode('utf-8'),
        request_body,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature, expected_signature)

def send_whatsapp_reply(to_number: str, message: str) -> bool:
    """Send reply via AI.Sensy API"""
    logger.info(f"Sending WhatsApp reply to {to_number}")
    
    if not config.get("sensy_api_key") or not config.get("sensy_api_url"):
        logger.error("AI.Sensy API credentials not configured")
        return False
    
    try:
        headers = {
            "Authorization": f"Bearer {config['sensy_api_key']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "to": to_number,
            "type": "text",
            "text": {
                "body": message
            }
        }
        
        response = requests.post(
            f"{config['sensy_api_url']}/messages",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            logger.info(f"Successfully sent message to {to_number}")
            return True
        else:
            logger.error(f"Failed to send message: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error sending WhatsApp message: {e}")
        return False

# @app.post("/whatsapp-webhook")

# async def whatsapp_webhook(request: Request):
#     body = await request.body()
#     print("BODY RAW:", body)
#     data = await request.json()
#     print("PARSED JSON:", json.dumps(data, indent=2))



# async def whatsapp_webhook(request: Request):
#     """Handle incoming WhatsApp messages from AI.Sensy"""
    
#     # Get raw body for signature verification
#     body = await request.body()
    
#     # Signature verification if configured bypassed
#     # signature = request.headers.get("X-Sensy-Signature", "")
#     # if config.get("webhook_secret") and not verify_webhook_signature(body, signature):
#     #     logger.warning("Invalid webhook signature")
#     #     raise HTTPException(status_code=401, detail="Invalid signature")

#     # Verify signature if configured
#     signature = request.headers.get("X-Sensy-Signature", "")
#     # if config.get("webhook_secret"):
#     #  Add this line to skip signature verification for testing
#     #     if request.headers.get("X-Test-Bypass") == "true":
#     #         logger.info("Bypassing signature verification for testing")
#     #     elif not verify_webhook_signature(body, signature):
#     #         logger.warning("Invalid webhook signature")
#     #         raise HTTPException(status_code=401, detail="Invalid signature")

    
#     # Parse request
#     # try:
#     #     payload = json.loads(body)
#     #     logger.info(f"Received webhook: {payload.get('event', 'unknown')}")
#     # except json.JSONDecodeError:
#     #     logger.error("Invalid JSON in webhook request")
#     #     raise HTTPException(status_code=400, detail="Invalid JSON")




    
#     # Handle different event types
#     # event_type = payload.get("event")



#     try:
#         data = json.loads(body)  # Rename to 'data' to represent the full object
#         logger.info(f"Received webhook: {data.get('event', 'unknown')}")
#     except json.JSONDecodeError:
#         logger.error("Invalid JSON in webhook request")
#         raise HTTPException(status_code=400, detail="Invalid JSON")

#     event_type = data.get("event")
#     payload = data.get("payload", {})  # Extract actual payload


#     payload = data.get("")


#     if "question" in data and "phone" in data:
#         text = data["question"]
#         from_number = data["phone"]

#         if not rag_engine:
#             logger.error("RAG engine not initialized")
#             return JSONResponse({"status": "error", "reason": "RAG not initialized"})







    
#     if event_type == "message.received":
#         # Extract message data
#         message_data = payload.get("data", {})
#         from_number = message_data.get("from")
#         text = message_data.get("text", {}).get("body", "")
#         message_id = message_data.get("id")
        
#         logger.info(f"Message from {from_number}: {text[:50]}...")
        
#         if not text:
#             return JSONResponse({"status": "ignored", "reason": "no text content"})
        
#         # Check if RAG engine is initialized
#         if not rag_engine:
#             logger.error("RAG engine not initialized")
#             send_whatsapp_reply(
#                 from_number,
#                 "I'm sorry, the service is temporarily unavailable. Please try again later."
#             )
#             return JSONResponse({"status": "error", "reason": "RAG not initialized"})
        
#         try:
#             # Process with RAG
#             logger.info(f"Processing query with RAG: {text}")
#             result = rag_engine.query(text)
            
#             # Send response
#             response_text = result.get("response", "I couldn't find an answer to your question.")
            
#             # Truncate if too long for WhatsApp
#             if len(response_text) > 4000:
#                 response_text = response_text[:3997] + "..."
            
#             # Send reply
#             success = send_whatsapp_reply(from_number, response_text)
            
#             # Log interaction
#             logger.info(f"Interaction completed - Message ID: {message_id}, Success: {success}")
            
#             return JSONResponse({
#                 "status": "success",
#                 "message_id": message_id,
#                 "response_sent": success
#             })
            
#         except Exception as e:
#             logger.error(f"Error processing message: {e}")
#             send_whatsapp_reply(
#                 from_number,
#                 "I encountered an error processing your message. Please try again."
#             )
#             return JSONResponse({"status": "error", "error": str(e)})
    
#     elif event_type == "message.sent":
#         # Log sent message confirmation
#         logger.info("Message sent confirmation received")
#         return JSONResponse({"status": "acknowledged"})
    
#     elif event_type == "message.delivered":
#         # Log delivery confirmation
#         logger.info("Message delivered confirmation received")
#         return JSONResponse({"status": "acknowledged"})
    
#     else:
#         logger.info(f"Unhandled event type: {event_type}")
#         return JSONResponse({"status": "ignored", "event": event_type})







@app.post("/whatsapp-webhook")
async def whatsapp_webhook(request: Request):
    """Handle incoming WhatsApp messages from AI.Sensy"""
    body = await request.body()
    logger.info("BODY RAW: %s", body)

    try:
        data = json.loads(body)
        logger.info("PARSED JSON: %s", json.dumps(data, indent=2))
    except json.JSONDecodeError:
        logger.error("Invalid JSON in webhook request")
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # ✅ Case 1: Manual test from Flow with "question" and "phone"
    if "question" in data and "phone" in data:
        text = data["question"]
        from_number = data["phone"]

        if not rag_engine:
            logger.error("RAG engine not initialized")
            return JSONResponse({"status": "error", "reason": "RAG not initialized"})

        try:
            # Get or create session for this phone number
            if conversation_manager:
                session = conversation_manager.get_or_create_session(f"whatsapp_{from_number}")
                
                # Check if we should show support link
                should_show_link = conversation_manager.should_show_support_link(text, session)
            else:
                should_show_link = False
            
            result = rag_engine.query(text)
            response_text = result.get("response", "Sorry, I couldn't find a response.")
            
            # Add support link if appropriate
            if should_show_link and conversation_manager:
                response_text += conversation_manager.support_link
                session.support_link_shown = True
            
            if len(response_text) > 4000:
                response_text = response_text[:3997] + "..."
            success = send_whatsapp_reply(from_number, response_text)
            return JSONResponse({"status": "success", "response_sent": success})
        except Exception as e:
            logger.error(f"RAG processing failed: {e}")
            return JSONResponse({"status": "error", "error": str(e)})

    # ✅ Case 2: Normal WhatsApp message
    event_type = data.get("event")
    payload = data.get("payload", {})

    if event_type == "message.received":
        message_data = payload.get("data", {})
        from_number = message_data.get("from")
        text = message_data.get("text", {}).get("body", "")
        message_id = message_data.get("id")

        logger.info(f"Message from {from_number}: {text[:50]}...")

        if not text:
            return JSONResponse({"status": "ignored", "reason": "no text content"})

        if not rag_engine:
            logger.error("RAG engine not initialized")
            send_whatsapp_reply(
                from_number,
                "Sorry, the service is temporarily unavailable. Please try again later."
            )
            return JSONResponse({"status": "error", "reason": "RAG not initialized"})

        try:
            # Get or create session for this phone number
            if conversation_manager:
                session = conversation_manager.get_or_create_session(f"whatsapp_{from_number}")
                
                # Check if we should show support link
                should_show_link = conversation_manager.should_show_support_link(text, session)
            else:
                should_show_link = False
            
            result = rag_engine.query(text)
            response_text = result.get("response", "Sorry, no response found.")
            
            # Add support link if appropriate
            if should_show_link and conversation_manager:
                response_text += conversation_manager.support_link
                session.support_link_shown = True
            
            if len(response_text) > 4000:
                response_text = response_text[:3997] + "..."
            success = send_whatsapp_reply(from_number, response_text)
            logger.info(f"Replied to {from_number} - Message ID: {message_id}, Success: {success}")
            return JSONResponse({
                "status": "success",
                "message_id": message_id,
                "response_sent": success
            })
        except Exception as e:
            logger.error(f"Processing error: {e}")
            send_whatsapp_reply(
                from_number,
                "I encountered an error processing your message. Please try again."
            )
            return JSONResponse({"status": "error", "error": str(e)})

    # ✅ Case 3: Sent/delivered confirmations
    elif event_type in ["message.sent", "message.delivered"]:
        logger.info(f"Received event: {event_type}")
        return JSONResponse({"status": "acknowledged"})

    # 🔁 Fallback for other events
    logger.info(f"Unhandled event type: {event_type}")
    return JSONResponse({"status": "ignored", "event": event_type})
























@app.post("/test-rag")



async def test_rag_endpoint(query: str):
    """Test endpoint for RAG queries"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    try:
        result = rag_engine.query(query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Run the FastAPI server"""
    port = config.get("server_port", 8000)
    host = config.get("server_host", "0.0.0.0")
    
    logger.info(f"Starting WhatsApp RAG Bot on {host}:{port}")
    logger.info("Make sure Ollama is running with Mistral model loaded")
    logger.info(f"Webhook URL: http://your-domain:{port}/whatsapp-webhook")
    
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()