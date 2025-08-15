#!/bin/bash
# Deployment script for VPS

echo "🚀 Deploying Vitiligo Chatbot..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10
sudo apt install python3.10 python3-pip python3-venv -y

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Mistral model
ollama pull mistral

# Clone your repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Create systemd service
sudo cat > /etc/systemd/system/chatbot.service << EOF
[Unit]
Description=Vitiligo Chatbot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/YOUR_REPO
Environment="PATH=/home/$USER/YOUR_REPO/venv/bin"
ExecStartPre=/usr/local/bin/ollama serve
ExecStart=/home/$USER/YOUR_REPO/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Start service
sudo systemctl enable chatbot
sudo systemctl start chatbot

echo "✅ Deployment complete! Running on port 8000"