import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv("GROQ_API_KEY")
print(f"Testing Groq Key: {api_key[:12]}...")

try:
    client = Groq(api_key=api_key)
    res = client.chat.completions.create(
        messages=[{"role": "user", "content": "test"}],
        model="llama-3.3-70b-versatile"
    )
    print("Success! Key is valid!")
    print(res.choices[0].message.content)
except Exception as e:
    print(f"❌ Failed: {e}")
