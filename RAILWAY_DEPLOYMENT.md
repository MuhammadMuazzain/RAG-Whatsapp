# 🚀 Railway Deployment Guide for Vitiligo Chatbot

## Prerequisites
- GitHub account
- Railway account (sign up at railway.app)
- Your code pushed to GitHub

## Step-by-Step Deployment Process

### 1️⃣ Prepare Your GitHub Repository

1. **Create a new GitHub repository** (if not already done)
```bash
git init
git add .
git commit -m "Initial commit for vitiligo chatbot"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/vitiligo-chatbot.git
git push -u origin main
```

2. **Ensure these files are in your repo:**
- ✅ Dockerfile (already created)
- ✅ railway.json (already created)
- ✅ .dockerignore (already created)
- ✅ requirements.txt
- ✅ All Python files (main.py, rag.py, etc.)
- ✅ vector_store/ folder with your FAISS index

### 2️⃣ Deploy to Railway

1. **Go to [Railway.app](https://railway.app)**

2. **Click "Start a New Project"**

3. **Select "Deploy from GitHub repo"**

4. **Connect your GitHub account** (if first time)

5. **Select your repository** `vitiligo-chatbot`

6. **Railway will automatically detect the Dockerfile**

### 3️⃣ Configure Environment Variables

In Railway dashboard, go to your project and click on "Variables":

Add these environment variables:

```env
# Required
PORT=8000
OLLAMA_HOST=http://127.0.0.1:11434

# If you have AI.Sensy webhook credentials
SENSY_API_KEY=your_api_key_here
SENSY_API_URL=https://api.aisensy.com/v1
WEBHOOK_SECRET=your_webhook_secret
```

### 4️⃣ Configure Your config.json

Update your `config.json` for production:

```json
{
    "sensy_api_key": "YOUR_AISENSY_API_KEY",
    "sensy_api_url": "https://api.aisensy.com/v1",
    "webhook_secret": "YOUR_WEBHOOK_SECRET",
    "server_host": "0.0.0.0",
    "server_port": 8000
}
```

### 5️⃣ Deploy

1. **Click "Deploy" in Railway**

2. **Watch the build logs** - it will:
   - Build the Docker image
   - Install dependencies
   - Download Ollama and Mistral model
   - Start the service

3. **Get your deployment URL**
   - It will be something like: `https://vitiligo-chatbot-production.up.railway.app`

### 6️⃣ Update AI.Sensy Webhook

1. Go to your AI.Sensy dashboard
2. Update webhook URL to:
   ```
   https://YOUR-APP-NAME.up.railway.app/whatsapp-webhook
   ```

### 7️⃣ Test Your Deployment

1. **Check health endpoint:**
   ```
   https://YOUR-APP-NAME.up.railway.app/health
   ```

2. **Test chat interface:**
   ```
   https://YOUR-APP-NAME.up.railway.app/
   ```

3. **Test WhatsApp integration:**
   - Send a message to your WhatsApp bot number

## 🔧 Troubleshooting

### If deployment fails:

1. **Check Railway logs:**
   - Click on your deployment
   - View "Deploy Logs" for build errors
   - View "Runtime Logs" for runtime errors

2. **Common issues:**

**Issue: Ollama fails to start**
```bash
# Solution: Increase sleep time in Dockerfile
sleep 15  # instead of sleep 10
```

**Issue: Out of memory**
```bash
# Solution: Railway free tier has 512MB RAM limit
# Upgrade to Hobby plan ($5/month) for 8GB RAM
```

**Issue: Port binding error**
```bash
# Solution: Use $PORT environment variable
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
```

### If WhatsApp webhook doesn't work:

1. **Verify webhook URL in AI.Sensy**
2. **Check Railway logs for incoming requests**
3. **Ensure config.json has correct credentials**

## 📊 Monitor Your App

1. **Railway Dashboard shows:**
   - Memory usage
   - CPU usage
   - Network requests
   - Logs

2. **Set up alerts:**
   - Go to Settings → Notifications
   - Enable deployment failure alerts

## 💰 Cost Estimation

- **Free Tier:** $5 credit/month (usually lasts 5-7 days)
- **Hobby Plan:** $5/month + usage
- **Your app will use approximately:**
  - Memory: ~1-2GB (with Ollama + Mistral)
  - CPU: Low-moderate
  - **Estimated cost:** $5-10/month

## 🔄 Updating Your App

When you make changes:

```bash
# Make your changes locally
git add .
git commit -m "Update chatbot responses"
git push origin main

# Railway will automatically redeploy!
```

## 🎯 Quick Checklist

Before deploying, ensure:
- [ ] All files committed to GitHub
- [ ] config.json has API credentials
- [ ] vector_store/ folder has your FAISS index
- [ ] Dockerfile is in root directory
- [ ] requirements.txt is up to date

## 🆘 Need Help?

- Railway Discord: https://discord.gg/railway
- Railway Docs: https://docs.railway.app
- Check logs: Railway Dashboard → Your Project → Logs

---

**Your deployment URL will be:**
```
https://[your-app-name].up.railway.app
```

**WhatsApp Webhook URL:**
```
https://[your-app-name].up.railway.app/whatsapp-webhook
```

Good luck with your deployment! 🚀