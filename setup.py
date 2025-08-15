"""
Setup script to verify installation and download models
"""

import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_ollama():
    """Check if Ollama is installed and running"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("✅ Ollama is installed")
            
            # Check if Mistral is available
            if 'mistral' in result.stdout:
                logger.info("✅ Mistral model is available")
                return True
            else:
                logger.warning("⚠️  Mistral model not found. Installing...")
                subprocess.run(['ollama', 'pull', 'mistral'])
                logger.info("✅ Mistral model installed")
                return True
        else:
            logger.error("❌ Ollama command failed")
            return False
    except FileNotFoundError:
        logger.error("❌ Ollama not found. Please install from https://ollama.ai")
        return False

def check_dependencies():
    """Check if Python dependencies are installed"""
    required = ['fastapi', 'sentence_transformers', 'faiss', 'fitz']
    missing = []
    
    for module in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        logger.error(f"❌ Missing dependencies: {', '.join(missing)}")
        logger.info("Run: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All Python dependencies installed")
    return True

def create_directories():
    """Create necessary directories"""
    Path("vector_store").mkdir(exist_ok=True)
    logger.info("✅ Created vector_store directory")

def download_embedding_model():
    """Pre-download the embedding model"""
    try:
        logger.info("Downloading embedding model (first time only)...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
        logger.info("✅ Embedding model ready")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download embedding model: {e}")
        return False

def main():
    """Run setup checks"""
    print("\n" + "="*50)
    print("RAG WhatsApp Bot Setup")
    print("="*50 + "\n")
    
    # Check all components
    checks = [
        ("Ollama & Mistral", check_ollama),
        ("Python Dependencies", check_dependencies),
        ("Directories", create_directories),
        ("Embedding Model", download_embedding_model)
    ]
    
    all_good = True
    for name, check_func in checks:
        logger.info(f"\nChecking {name}...")
        if not check_func():
            all_good = False
    
    print("\n" + "="*50)
    if all_good:
        print("✅ Setup complete! You're ready to go.")
        print("\nNext steps:")
        print("1. Process a PDF: python embed.py your_document.pdf")
        print("2. Start Ollama: ollama serve")
        print("3. Run the bot: python main.py")
    else:
        print("❌ Setup incomplete. Please fix the issues above.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()