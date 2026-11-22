import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def ping_tool(word: str) -> dict:
    """Echo a word back. Call this when asked to ping."""
    return {"echo": word}


def try_call(model_name, with_tools):
    try:
        cfg = types.GenerateContentConfig(
            max_output_tokens=200,
            tools=[ping_tool] if with_tools else None,
        )
        prompt = "Use the ping_tool with the word hello, then reply OK." if with_tools else "Reply with the word OK."
        client.models.generate_content(model=model_name, contents=prompt, config=cfg)
        return "OK"
    except errors.APIError as e:
        return f"FAIL {e.code} {e.status}"
    except Exception as e:
        return f"FAIL {type(e).__name__}"


print("GEMINI_MODEL          =", repr(os.getenv("GEMINI_MODEL")))
print("GEMINI_FALLBACK_MODEL =", repr(os.getenv("GEMINI_FALLBACK_MODEL")))
print()

names = []
for m in client.models.list():
    if "generateContent" in (m.supported_actions or []) and "gemini" in m.name.lower():
        names.append(m.name.replace("models/", ""))

print(f"Found {len(names)} models. Testing each one...\n")
print(f"{'MODEL':50} {'chat':22} tools")
print("-" * 95)

working = []
for n in sorted(names):
    chat = try_call(n, with_tools=False)
    time.sleep(1)
    tools = try_call(n, with_tools=True) if chat == "OK" else "-"
    time.sleep(1)
    print(f"{n:50} {chat:22} {tools}")
    if chat == "OK" and tools == "OK":
        working.append(n)

print("\nModels that work for your chatbot (chat + tools):")
for n in working:
    print("  ", n)
print("\nPut one in .env as GEMINI_MODEL=<name> and another as GEMINI_FALLBACK_MODEL=<name>, then restart uvicorn.")