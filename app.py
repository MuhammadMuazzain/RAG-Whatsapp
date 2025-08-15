# For Hugging Face Spaces deployment
import gradio as gr
from rag import RAGEngine
from conversation_manager import ConversationManager

# Initialize
rag_engine = RAGEngine()
conversation_manager = ConversationManager()

def chat_function(message, history):
    """Gradio chat function"""
    # Process with conversation manager
    result = conversation_manager.process_message(message)
    
    # Get response
    if not result['use_rag']:
        response = result['quick_response']
    else:
        rag_result = rag_engine.query(message, response_style=result['response_style'])
        response = rag_result['response']
    
    return response

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat_function,
    title="Vitiligo Medical Chatbot",
    description="Ask questions about vitiligo and get answers from medical documents",
    examples=["What is vitiligo?", "What are the symptoms?", "How is it treated?"],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)