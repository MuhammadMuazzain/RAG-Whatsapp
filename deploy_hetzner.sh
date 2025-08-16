#!/bin/bash
# Hetzner VPS Deployment Script

echo "🚀 Deploying to Hetzner VPS..."

# SSH into your Hetzner server and run this:

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt install docker-compose -y

# Clone your repository
git clone https://github.com/YOUR_USERNAME/vitiligo-chatbot.git
cd vitiligo-chatbot

# Build and run with Docker
sudo docker build -t vitiligo-bot .
sudo docker run -d \
  --name chatbot \
  --restart always \
  -p 80:8000 \
  -p 443:8000 \
  vitiligo-bot

# Setup SSL with Caddy (automatic HTTPS)
sudo apt install caddy -y
sudo cat > /etc/caddy/Caddyfile << EOF
your-domain.com {
    reverse_proxy localhost:8000
}
EOF
sudo systemctl restart caddy

echo "✅ Deployment complete!"
echo "Access at: https://your-domain.com"