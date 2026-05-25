# 📊 Prompt Evaluation Framework

A lightweight Python-based framework to evaluate and compare LLM prompt outputs using structured datasets.

---

## 🚀 Features

- Run batch evaluations on multiple prompts
- Compare model outputs programmatically
- CSV-based prompt management
- Simple Python execution pipeline
- Easy to extend for different LLMs / APIs

---

## 📁 Project Structure

prompt-evaluation-framework/
│
├── evaluate.py        # Main evaluation script
├── test.py            # Test runs / experiments
├── prompts.csv        # Input prompts dataset
├── .env               # API keys / environment variables
├── venv/              # Virtual environment
└── README.md

---

## ⚙️ Setup Instructions

### 1. Create virtual environment

python -m venv venv

---

### 2. Activate environment

Windows:
venv\Scripts\activate

---

### 3. Install dependencies

pip install -r requirements.txt

If requirements.txt is not present, install required libraries manually.

---

### 4. Configure environment variables

Create a `.env` file:

API_KEY=your_api_key_here
MODEL_NAME=gpt-4 or other model

---

## ▶️ How to Run

### Run evaluation

python evaluate.py

---

### Run test script

python test.py

---

## 📊 Input Format (prompts.csv)

Example:

prompt_id,prompt
1,Explain machine learning in simple terms
2,Write a Python function to reverse a string

---

## 🧠 How It Works

1. Loads prompts from `prompts.csv`
2. Sends each prompt to the model/API
3. Collects responses
4. Evaluates or logs outputs
5. Returns results for analysis

---

## 🔧 Future Improvements

- Add scoring metrics (accuracy, relevance, toxicity)
- Add LangSmith / LangChain integration
- Web dashboard for evaluation results
- Support multiple LLM providers

---

## 🧑‍💻 Author

Built for experimenting with prompt evaluation pipelines and LLM testing workflows.

---

## 📌 Notes

- Never commit `.env` file
- Extend `evaluate.py` for custom evaluation logic
- Keep prompts structured for consistent evaluation