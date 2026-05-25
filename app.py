import os
import json
import re
import streamlit as st
from google import genai
from langsmith import Client
from langsmith.run_helpers import traceable
 
# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Prompt Evaluator",
    page_icon="🧠",
    layout="wide"
)
 
# ---------------------------
# LOAD KEYS (from Streamlit Secrets or env)
# ---------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
LANGCHAIN_API_KEY = st.secrets.get("LANGCHAIN_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
 
if not GEMINI_API_KEY or not LANGCHAIN_API_KEY:
    st.error("❌ API keys missing. Add GEMINI_API_KEY and LANGCHAIN_API_KEY in Streamlit Secrets.")
    st.stop()
 
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
 
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
# CORE FUNCTIONS
# ---------------------------
def generate_answer(prompt: str):
    for model in ANSWER_MODELS:
        try:
            response = gemini.models.generate_content(model=model, contents=prompt)
            return response.text
        except Exception as e:
            st.warning(f"Model {model} failed: {e}")
    raise Exception("All answer models failed")
 
 
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
            response = gemini.models.generate_content(model=model, contents=judge_prompt)
            return response.text
        except Exception as e:
            last_error = e
    return json.dumps({"error": "Judge model failed", "reason": str(last_error)})
 
 
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
    cleaned = clean_json_text(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"error": "Failed to parse judge output", "raw": text}
 
 
@traceable(name="eval_run")
def run_eval(prompt: str):
    answer = generate_answer(prompt)
    judge_raw = judge_answer(prompt, answer)
    judge = parse_judge_output(judge_raw)
    return {"prompt": prompt, "answer": answer, "judge": judge}
 
 
# ---------------------------
# UI
# ---------------------------
st.title("🧠 Prompt Evaluation Framework")
st.markdown("Evaluate LLM prompt responses using **Gemini** as the judge, tracked via **LangSmith**.")
st.divider()
 
st.subheader("📝 Enter Prompts to Evaluate")
default_prompts = (
    "Explain LangSmith in simple terms\n"
    "What is overfitting in ML?\n"
    "Define LLM hallucination with example\n"
    "What is vector database?"
)
prompts_input = st.text_area(
    "One prompt per line:",
    value=default_prompts,
    height=150
)
 
run_btn = st.button("▶ Run Evaluation", type="primary", use_container_width=True)
 
if run_btn:
    prompts = [p.strip() for p in prompts_input.strip().splitlines() if p.strip()]
    if not prompts:
        st.warning("Please enter at least one prompt.")
        st.stop()
 
    st.divider()
    st.subheader("📊 Results")
 
    for i, prompt in enumerate(prompts):
        with st.spinner(f"Evaluating: *{prompt}*"):
            try:
                result = run_eval(prompt)
                judge = result["judge"]
 
                with st.expander(f"**{i+1}. {prompt}**", expanded=True):
                    st.markdown("**🤖 Answer:**")
                    st.write(result["answer"])
 
                    if "error" in judge:
                        st.error(f"Judge error: {judge.get('reason', judge.get('raw', ''))}")
                    else:
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Correctness", f"{judge.get('correctness', '?')}/10")
                        col2.metric("Clarity", f"{judge.get('clarity', '?')}/10")
                        col3.metric("Completeness", f"{judge.get('completeness', '?')}/10")
                        col4.metric("Total", f"{judge.get('total', '?')}/30")
                        st.markdown(f"**💬 Reason:** {judge.get('reason', 'N/A')}")
 
            except Exception as e:
                st.error(f"Failed to evaluate prompt: {e}")
 
    st.success("✅ Evaluation complete! Results are also logged to LangSmith.")
 