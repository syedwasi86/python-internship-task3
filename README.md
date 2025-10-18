# python-internship-task3
# 🤖 AI Chatbot with NLP

A simple **Natural Language Processing (NLP)** based chatbot built using **Python**, **spaCy**, and **NLTK**.
This chatbot can understand user queries, detect greetings and intents, and retrieve answers from a small knowledge base using TF-IDF and cosine similarity.

---

## 🧠 Features

* Text preprocessing with **NLTK** (tokenization + stopword removal)
* Entity recognition and intent detection with **spaCy**
* Question–Answer retrieval using **TF-IDF + cosine similarity**
* Handles greetings, thanks, and exit phrases
* Saves conversation logs to `chat_log.json`
* Easily expandable knowledge base (JSON or in-code list)

---

## 🧰 Technologies Used

* Python 3.10+
* spaCy
* NLTK
* scikit-learn

---

## ⚙️ Installation (Windows)

1. **Clone this repository**

   ```bash
   git clone https://github.com/<your-username>/AI-Chatbot-With-NLP.git
   cd AI-Chatbot-With-NLP
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

4. **(Optional)** Edit `knowledge_base.json` to add your own questions and answers.

---

## 🚀 Run the Chatbot

```bash
python nlp_chatbot.py
```

Example:

```
Welcome to CodTech NLP Chatbot (type 'exit' or 'quit' to stop).

You: hi
Bot: Hello! I'm CodTech Assistant. How can I help you today?

You: what is nlp
Bot: Natural Language Processing (NLP) is the field that gives computers the ability to understand text...
```

When you exit, a `chat_log.json` file is automatically created with your conversation.

---

## 📚 Knowledge Base

You can store the chatbot’s question-answer pairs directly in the code (`KB` list)
or in an external JSON file:

```json
[
    {"q": "what is ai", "a": "AI stands for Artificial Intelligence."},
    {"q": "what is python", "a": "Python is a popular programming language used for AI and automation."}
]
```

---

## 🧩 Project Structure

```
├── nlp_chatbot.py        # Main chatbot script
├── requirements.txt      # Python dependencies
├── knowledge_base.json   # Optional knowledge base
├── chat_log.json         # Conversation logs (auto-generated)
└── README.md             # Project documentation
```

---

## 🏁 Output Example

```
You: who created you
Bot: I was created as part of a CodTech AI internship project.
```

