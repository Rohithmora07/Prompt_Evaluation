import os
import json
import re
from dotenv import load_dotenv

from google import genai
from google.genai import errors as genai_errors
from langsmith import Client
from langsmith.run_helpers import traceable

# ---------------------------
# ENV
# ---------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# ---------------------------
# CLIENTS
# ---------------------------
gemini = genai.Client(api_key=GEMINI_API_KEY)
ls_client = Client()

# ---------------------------
# MODELS
# ---------------------------
ANSWER_MODELS = [
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-2.5-flash-lite",
]
JUDGE_MODELS = [
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-2.5-flash-lite",
]

# ---------------------------
# DATASET
# ---------------------------
DATASET = [
    "Explain LangSmith in simple terms",
    "What is overfitting in ML?",
    "Define LLM hallucination with example",
    "What is vector database?"
]

# ---------------------------
# GENERATE ANSWER
# ---------------------------
def generate_answer(prompt: str):
    for model in ANSWER_MODELS:
        try:
            response = gemini.models.generate_content(
                model=model,
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"Model failed: {model} → {e}")

    raise Exception("All models failed")

# ---------------------------
# LLM AS JUDGE
# ---------------------------
def judge_answer(prompt: str, answer: str) -> str:
    judge_prompt = f"""
You are an evaluation system.

Evaluate the answer based on:
- correctness (0-10)
- clarity (0-10)
- completeness (0-10)

Return ONLY valid JSON. Do not wrap it in markdown code fences:
{{
  "correctness": int,
  "clarity": int,
  "completeness": int,
  "total": int,
  "reason": string
}}

QUESTION:
{prompt}

ANSWER:
{answer}
"""

    last_error = None
    for model in JUDGE_MODELS:
        try:
            response = gemini.models.generate_content(
                model=model,
                contents=judge_prompt
            )
            return response.text
        except genai_errors.ClientError as e:
            last_error = e
            print(f"Judge model failed: {model} -> {e}")

            if e.status_code != 429:
                break
        except Exception as e:
            last_error = e
            print(f"Judge model failed: {model} -> {e}")

    return json.dumps({
        "error": "Judge model failed",
        "reason": str(last_error)
    })

def clean_json_text(text: str) -> str:
    text = text.strip()

    fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return text[start:end + 1]

    return text

def parse_judge_output(text: str) -> dict:
    cleaned_text = clean_json_text(text)

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        return {
            "error": "Failed to parse judge output",
            "raw": text
        }

# ---------------------------
# SINGLE EVALUATION RUN
# ---------------------------
@traceable(name="eval_run")
def run_eval(prompt: str):
    answer = generate_answer(prompt)
    judge_raw = judge_answer(prompt, answer)
    judge = parse_judge_output(judge_raw)

    return {
        "prompt": prompt,
        "answer": answer,
        "judge": judge
    }

# ---------------------------
# BATCH RUNNER
# ---------------------------
def run_batch():
    results = []

    for prompt in DATASET:
        print(f"Running: {prompt}")
        try:
            result = run_eval(prompt)
        except Exception as e:
            result = {
                "prompt": prompt,
                "answer": None,
                "judge": {
                    "error": "Evaluation failed",
                    "reason": str(e),
                },
            }
        results.append(result)

    return results

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    output = run_batch()

    print("\n\n===== FINAL RESULTS =====\n")
    for r in output:
        print("PROMPT:", r["prompt"])
        print("JUDGE:", r["judge"])
        print("-" * 50)
