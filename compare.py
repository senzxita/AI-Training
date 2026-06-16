import os
import anthropic
import json
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

prompt = "Understanding the Use of AI in the present age"


# ---------------- OPENAI ----------------
def call_openai():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return {
        "text": res.choices[0].message.content,
        "tokens": res.usage.total_tokens
    }


# ---------------- ANTHROPIC ----------------
def call_anthropic():
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    res = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return res.content[0].text
    

# ---------------- LATENCY AND TOKENS ----------------
def run_and_measure(func):
    start = time.time()
    result = func()
    end = time.time()

    print(type(result))
    print(result)

    result["latency"] = round(end - start, 3)
    #return result

# ---------------- RESPONSE QUALITY ----------------
def score_quality(text):
    score = 0

    # Length-based scoring
    if len(text) > 500:
        score += 5
    elif len(text) > 250:
        score += 3
    else:
        score += 1

    # Check for important concepts
    keywords = [
        "AI",
        "machine learning",
        "automation",
        "data",
        "decision"
    ]

    keyword_score = sum(
        1 for word in keywords
        if word.lower() in text.lower()
    )

    score += keyword_score

    # Limit score to maximum of 10
    return min(score, 10)


# ---------------- RUN BENCHMARK ----------------
def run_benchmark(func):
    # Start timer
    start = time.time()

    # Call the AI model function
    result = func()

    # Stop timer
    end = time.time()

    # Calculate latency
    latency = round(end - start, 3)

    # Calculate quality score
    quality = score_quality(result["text"])

    # Return all benchmark metrics
    return {
        "response": result["text"],
        "tokens_used": result["tokens"],
        "latency_seconds": latency,
        "quality_score": quality
    }

# ---------------- MAIN ----------------
def main():
    results = {
        "openai": call_openai(),
        "anthropic": call_anthropic()
    }

    print(json.dumps(results, indent=4))

    with open("ai_outputs.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nSaved to ai_outputs.json")
    print("Benchmark completed. Results saved to ai_benchmark.json")

# ---------------- PRINT JSON AS TEXT ----------------
def print_json_as_text():
    with open("ai_outputs.json", "r") as f:
        data = json.load(f)

    print("\n===== AI RESPONSES (TEXT FORMAT) =====\n")

    print("OPENAI:\n")
    print(data["openai"])

    print("\n------------------------\n")

    print("ANTHROPIC:\n")
    print(data["anthropic"])

    print("\n------------------------\n")

# ---------------- RUN ORDER FIX ----------------
if __name__ == "__main__":
    main()
    print_json_as_text()