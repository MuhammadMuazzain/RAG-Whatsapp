#!/usr/bin/env python
"""
Test script to verify Railway deployment
"""

import requests
import sys

def test_deployment(base_url):
    """Test the deployed application"""
    
    print(f"\n🧪 Testing deployment at: {base_url}")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1️⃣ Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("   ✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Root endpoint
    print("\n2️⃣ Testing root endpoint (chat UI)...")
    try:
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            print("   ✅ Chat UI is accessible")
        else:
            print(f"   ❌ Chat UI failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Chat API
    print("\n3️⃣ Testing chat API...")
    try:
        response = requests.post(
            f"{base_url}/chat",
            json={"message": "Hello"},
            timeout=30
        )
        if response.status_code == 200:
            print("   ✅ Chat API working")
            data = response.json()
            print(f"   Bot response: {data.get('response', 'No response')[:100]}...")
        else:
            print(f"   ❌ Chat API failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: WhatsApp webhook
    print("\n4️⃣ Testing WhatsApp webhook...")
    try:
        response = requests.post(
            f"{base_url}/whatsapp-webhook",
            json={
                "event": "message.received",
                "payload": {
                    "data": {
                        "from": "1234567890",
                        "text": {"body": "Test message"},
                        "id": "test123"
                    }
                }
            },
            timeout=30
        )
        if response.status_code == 200:
            print("   ✅ WhatsApp webhook working")
        else:
            print(f"   ⚠️ WhatsApp webhook returned: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 Deployment test complete!")
    print("\nNext steps:")
    print("1. Update your AI.Sensy webhook URL to:")
    print(f"   {base_url}/whatsapp-webhook")
    print("2. Test by sending a WhatsApp message")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_deployment.py <deployment-url>")
        print("Example: python test_deployment.py https://myapp.up.railway.app")
        sys.exit(1)
    
    url = sys.argv[1].rstrip('/')
    test_deployment(url)