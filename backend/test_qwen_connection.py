"""
Quick standalone test — run this once on your server to confirm your
Qwen API key actually works, before relying on it inside the full app.

Usage on your Alibaba server:
    cd ~/lifeline-relay-ai/backend
    python3 test_qwen_connection.py
"""

import os
import asyncio
import httpx
from dotenv import load_dotenv

load_dotenv()

QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"


async def test_connection():
    if not QWEN_API_KEY:
        print("ERROR: QWEN_API_KEY not found. Check your .env file exists and has the right variable name.")
        return

    print(f"Using API key: {QWEN_API_KEY[:8]}...(hidden)")
    print("Sending test request to Qwen...")

    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "qwen-plus",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Respond with ONLY JSON: {\"status\": \"ok\", \"message\": \"<short greeting>\"}"},
            {"role": "user", "content": "Say hello and confirm you're working."}
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(QWEN_BASE_URL, headers=headers, json=payload)
            print(f"HTTP status: {response.status_code}")
            response.raise_for_status()
            data = response.json()
            print("\nSUCCESS — Qwen responded:")
            print(data["choices"][0]["message"]["content"])
    except httpx.HTTPStatusError as e:
        print(f"\nFAILED — HTTP error: {e.response.status_code}")
        print(f"Response body: {e.response.text}")
    except Exception as e:
        print(f"\nFAILED — {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(test_connection())
