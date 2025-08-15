# RAG WhatsApp Bot

A production-ready WhatsApp chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions based on PDF documents. Built with FastAPI, FAISS, Mistral-7B (via Ollama), and AI.Sensy for WhatsApp integration.

## 🚀 Features

- **PDF Processing**: Extract and chunk PDF documents for knowledge base
- **Semantic Search**: Use sentence-transformers and FAISS for relevant content retrieval
- **LLM Generation**: Generate responses using Mistral-7B running locally via Ollama
- **WhatsApp Integration**: Receive and respond to WhatsApp messages via AI.Sensy
- **Modular Design**: Clean separation of embedding, RAG, and API logic
- **Production Ready**: Logging, error handling, and webhook verification

## 📋 Prerequisites

1. **Python 3.8+**
2. **Ollama** installed with Mistral model:
   ```bash
   # Install Ollama (https://ollama.ai)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull Mistral model
   ollama pull mistral
   
   # Run Ollama server
   ollama serve
   ```
3. **AI.Sensy Account** for WhatsApp API access

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   cd rag_whatsapp_bot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure AI.Sensy credentials**:
   Edit `config.json` with your credentials:
   ```json
   {
     "sensy_api_key": "YOUR_ACTUAL_API_KEY",
     "sensy_api_url": "https://api.sensy.ai/v1",
     "webhook_secret": "YOUR_WEBHOOK_SECRET"
   }
   ```

## 📚 Usage

### Step 1: Process PDF Documents

Process your first PDF to create the knowledge base:
```bash
python embed.py your_document.pdf
```

Add more PDFs to the existing knowledge base:
```bash
python embed.py another_document.pdf --append
```

This will:
- Extract text from the PDF
- Split into ~300 word chunks
- Generate embeddings using `multi-qa-MiniLM-L6-cos-v1`
- Store in FAISS index at `vector_store/`

### Step 2: Start the WhatsApp Bot

1. **Ensure Ollama is running**:
   ```bash
   # In a separate terminal
   ollama serve
   ```

2. **Start the FastAPI server**:
   ```bash
   python main.py
   ```
   
   The server will start on `http://localhost:8000`

3. **Configure webhook in AI.Sensy**:
   - Set webhook URL: `https://your-domain.com/whatsapp-webhook`
   - Or use ngrok for local testing: `ngrok http 8000`

### Step 3: Test the Bot

1. **Via API** (for testing):
   ```bash
   curl -X POST "http://localhost:8000/test-rag?query=What%20is%20the%20main%20topic"
   ```

2. **Via WhatsApp**:
   - Send a message to your AI.Sensy WhatsApp number
   - The bot will search the knowledge base and respond

## 🏗️ Architecture

```
rag_whatsapp_bot/
├── embed.py          # PDF → Chunks → Embeddings → FAISS
├── rag.py           # Search + Mistral generation
├── main.py          # FastAPI server + WhatsApp webhook
├── config.json      # Configuration
└── vector_store/    # FAISS index + metadata
    ├── faiss.index
    └── chunks.json
```

### Data Flow

1. **Embedding Phase**:
   ```
   PDF → Text Extraction → Chunking → Embeddings → FAISS Index
   ```

2. **Query Phase**:
   ```
   WhatsApp Message → Embed Query → FAISS Search → Top 3 Chunks
   → Format Prompt → Mistral-7B → Response → WhatsApp Reply
   ```

## 🔧 Configuration

Edit `config.json` to customize:

- `chunk_size`: Words per chunk (default: 300)
- `max_chunks_per_query`: Context chunks for LLM (default: 3)
- `mistral_temperature`: Response creativity (0-1)
- `server_port`: API server port

## 📊 Monitoring

Check logs for:
- PDF processing status
- Incoming WhatsApp messages
- RAG search results
- LLM generation
- Response delivery

Health check endpoints:
- `GET /` - Basic health check
- `GET /health` - Detailed component status

## 🚨 Troubleshooting

### "Ollama not connected"
- Ensure Ollama is running: `ollama serve`
- Check if Mistral is installed: `ollama list`

### "FAISS index not found"
- Process at least one PDF first: `python embed.py document.pdf`

### "Webhook signature invalid"
- Verify `webhook_secret` in config.json matches AI.Sensy settings

### Poor quality responses
- Add more relevant PDFs to knowledge base
- Adjust `chunk_size` for better context
- Check if PDFs are being extracted correctly

## 🔒 Security

- Webhook signatures are verified using HMAC-SHA256
- API keys are stored in config.json (use environment variables in production)
- No user data is permanently stored

## 🚀 Production Deployment

1. Use environment variables instead of config.json
2. Deploy behind HTTPS (required for webhooks)
3. Use a process manager like PM2 or systemd
4. Set up proper logging and monitoring
5. Consider using GPU for faster embeddings

## 📝 License

MIT License - feel free to use and modify!

## 🤝 Contributing

Pull requests welcome! Please ensure code follows the existing style and includes appropriate logging.