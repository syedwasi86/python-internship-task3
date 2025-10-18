#!/usr/bin/env python3
"""
nlp_chatbot.py

A simple, self-contained command-line chatbot using spaCy + NLTK + scikit-learn.
Features:
- Preprocessing with NLTK (tokenize, stopwords)
- spaCy for named entity extraction and simple pattern matching (greetings, bye)
- TF-IDF + cosine similarity retrieval over a small knowledge base
- Fallback small-talk handling

Usage:
1. Create a virtual environment (recommended)
   python3 -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate    # Windows

2. Install dependencies:
   pip install -r requirements.txt
   # or:
   pip install spacy nltk scikit-learn
   python -m spacy download en_core_web_sm

3. Run:
   python nlp_chatbot.py

"""

import os
import sys
import re
import json
import time
from typing import List, Tuple

# NLP libraries
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.matcher import Matcher

# --- Setup / Downloads (first-run safe checks) ---

def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

ensure_nltk_data()

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except Exception:
    print("spaCy model en_core_web_sm not found. Attempting to download...\n")
    os.system(f"{sys.executable} -m spacy download en_core_web_sm")
    nlp = spacy.load('en_core_web_sm')

# --- Knowledge base (small sample) ---
KB = json.load(open("knowledge_base.json"))


# Precompute a list of KB questions for vectorization
kb_questions = [item['q'] for item in KB]

# --- Preprocessing utilities ---
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words('english'))

word_pattern = re.compile(r"\b\w+\b")

def preprocess(text: str) -> str:
    """Lowercase, remove non-alphanumeric, remove stopwords (lightweight)."""
    text = text.lower()
    tokens = word_pattern.findall(text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

# Preprocess KB
kb_questions_preprocessed = [preprocess(q) for q in kb_questions]

# --- TF-IDF vectorizer on KB ---
vectorizer = TfidfVectorizer()
X_kb = vectorizer.fit_transform(kb_questions_preprocessed)

# --- spaCy pattern matcher for simple intents ---
matcher = Matcher(nlp.vocab)

# Greeting patterns
patterns_greet = [[{"LOWER": "hi"}], [{"LOWER": "hello"}], [{"LOWER": "hey"}], [{"LOWER": "good"}, {"LOWER": "morning"}], [{"LOWER": "good"}, {"LOWER": "evening"}]]
matcher.add("GREET", patterns_greet)

# Bye patterns
patterns_bye = [[{"LOWER": "bye"}], [{"LOWER": "goodbye"}], [{"LOWER": "see"}, {"LOWER": "you"}], [{"LOWER": "exit"}], [{"LOWER": "quit"}]]
matcher.add("BYE", patterns_bye)

# Thanks patterns
patterns_thanks = [[{"LOWER": "thanks"}], [{"LOWER": "thank"}, {"LOWER": "you"}], [{"LOWER": "ty"}]]
matcher.add("THANKS", patterns_thanks)

# --- Chatbot core logic ---

def detect_intent_spacy(text: str) -> List[str]:
    doc = nlp(text)
    matches = matcher(doc)
    intents = []
    for match_id, start, end in matches:
        intent = nlp.vocab.strings[match_id]
        intents.append(intent)
    return intents


def retrieve_answer_by_similarity(user_text: str, threshold: float = 0.35) -> Tuple[str, float, int]:
    """Return best matching KB answer and similarity score. If score < threshold, return empty string."""
    pre = preprocess(user_text)
    v = vectorizer.transform([pre])
    sims = cosine_similarity(v, X_kb)[0]
    best_idx = sims.argmax()
    best_score = float(sims[best_idx])
    if best_score >= threshold:
        return KB[best_idx]['a'], best_score, best_idx
    else:
        return "", best_score, best_idx


def handle_user_input(user_text: str) -> str:
    user_text = user_text.strip()
    if not user_text:
        return "Please type something — I can't read blank messages."

    # Quick checks: math question simple eval (very limited and safe)
    math_match = re.match(r"^calculate\s+([0-9()+\-*/.\s]+)$", user_text.lower())
    if math_match:
        try:
            expr = math_match.group(1)
            # VERY small sandbox: allow digits and operators only
            if re.fullmatch(r"[0-9()+\-*/.\s]+", expr):
                result = eval(expr, {"__builtins__": {}}, {})
                return f"Result: {result}"
        except Exception:
            return "I couldn't calculate that expression."

    # Intent detection
    intents = detect_intent_spacy(user_text)
    if 'GREET' in intents:
        return "Hello! I'm CodTech Assistant. How can I help you today?"
    if 'BYE' in intents:
        return "Goodbye! If you need anything else, come back and chat."
    if 'THANKS' in intents:
        return "You're welcome — happy to help!"

    # Named entity extraction (informational)
    doc = nlp(user_text)
    ents = [(ent.text, ent.label_) for ent in doc.ents]

    # Try retrieval from KB
    answer, score, idx = retrieve_answer_by_similarity(user_text)
    if answer:
        debug_info = f"\n\n(TRACE: matched KB Q#{idx} with score={score:.2f})"
        return answer + debug_info

    # If no KB answer, try small heuristics
    if len(ents) > 0:
        ent_preview = ", ".join([f"{t}({l})" for t,l in ents])
        return f"I noticed you mentioned: {ent_preview}. I don't have a direct answer in my KB — try phrasing as a question or add more details."

    # Fallback: ask clarifying question and offer top suggestions
    # Provide top-3 closest KB questions as suggestions
    pre = preprocess(user_text)
    v = vectorizer.transform([pre])
    sims = cosine_similarity(v, X_kb)[0]
    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:3]
    suggestions = []
    for i, s in ranked:
        suggestions.append(f"{KB[i]['q']} (score {s:.2f})")
    sugg_text = "\n".join(suggestions)
    return (
        "I couldn't find a confident answer. You can try rephrasing or choose one of these related questions:\n"
        + sugg_text
    )

# --- Command-line chat loop ---

def chat_loop():
    print("\nWelcome to CodTech NLP Chatbot (type 'exit' or 'quit' to stop).\n")
    time.sleep(0.2)
    conversation = []
    while True:
        try:
            user = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting chat. Goodbye!")
            break
        if user.strip().lower() in ("exit", "quit"):
            print("Bot: Goodbye! Good luck with your internship.")
            break
        bot_resp = handle_user_input(user)
        print(f"Bot: {bot_resp}\n")
        conversation.append({"user": user, "bot": bot_resp})

    # Save conversation log
    try:
        with open("chat_log.json", "w", encoding="utf-8") as f:
            json.dump(conversation, f, indent=2)
        print("Conversation saved to chat_log.json")
    except Exception as e:
        print("Couldn't save chat log:", e)


if __name__ == '__main__':
    chat_loop()
