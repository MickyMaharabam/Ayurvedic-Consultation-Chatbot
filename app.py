import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AyuTalk", page_icon="🌿")
st.title("🌿 TriDosha Talk")
st.caption("AI based Ayurvedic Consultation Chatbot")

# ---------------- MODEL PATH ----------------
MODEL_PATH = "mickymaharabam/AyuTalk_model"
CONFIDENCE_THRESHOLD = 0.5  # Lowered threshold to allow multi-dosha prediction

# ---------------- OPENAI CLIENT ----------------
client = OpenAI()  # Reads API key from environment

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD BERT MODEL ----------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- DOSHA DATA ----------------
DOSHA_MAP = {0: "Vata", 1: "Pitta", 2: "Kapha"}

DOSHA_INFO = {
    "Vata": "Vata governs movement. Imbalance may cause dryness, anxiety, insomnia, constipation.",
    "Pitta": "Pitta governs metabolism. Imbalance may cause acidity, burning sensation, anger.",
    "Kapha": "Kapha governs structure. Imbalance may cause heaviness, lethargy, weight gain."
}

DOSHA_REMEDIES = {
    "Vata": "Eat warm, cooked foods; maintain routine; practice yoga and meditation.",
    "Pitta": "Eat cooling foods; avoid spicy items; stay hydrated; practice calming activities.",
    "Kapha": "Eat light meals; exercise regularly; avoid heavy and oily foods."
}

# ---------------- GPT FALLBACK ----------------
def ask_gpt(symptoms):
    try:
        prompt = f"""
You are an Ayurvedic assistant.

User input:
{symptoms}

Tasks:
1. Identify possible dosha imbalance (single or combined).
2. Explain the reasoning briefly.
3. Suggest lifestyle and dietary remedies.
4. Avoid medicine names.
5. Do NOT provide medical diagnosis.
"""
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt
        )
        return response.output_text
    except Exception:
        return """
🧠 **Possible Dosha Imbalance: Vata**

General guidance:
• Follow a fixed daily routine  
• Eat warm, nourishing foods  
• Practice meditation and breathing  
• Ensure proper sleep  

*This guidance is general and not a medical diagnosis.*
"""

# ---------------- SESSION MEMORY ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- DISPLAY CHAT ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- USER INPUT ----------------
user_input = st.chat_input("Describe your symptoms...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    user_text = user_input.lower()

    # ---------------- KEYWORDS ----------------
    greeting_keywords = [
        "hello", "hi", "hey", "namaste",
        "good morning", "good evening", "good afternoon"
    ]

    symptom_keywords = [
        "pain", "fever", "dryness", "anxiety", "constipation",
        "acidity", "burning", "heaviness", "lethargy", "insomnia",
        "cough", "cold", "headache", "fatigue", "nausea", "focus"
    ]

    # ---------------- DECISION LOGIC ----------------

    # GREETING
    if any(greet in user_text for greet in greeting_keywords):
        bot_reply = """
🙏 **Namaste! Welcome to TriDosha Talk 🌿**

I can help you with:
• Understanding possible dosha imbalance  
• Ayurvedic lifestyle and diet remedies  
• General wellness guidance  

Please describe your symptoms to begin.
"""

    # UNKNOWN / GENERAL TEXT → GPT
    elif not any(word in user_text for word in symptom_keywords):
        bot_reply = ask_gpt(user_input)

    # SYMPTOM TEXT → BERT
    else:
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).squeeze()  # shape: [3]

        # Pick all doshas above threshold
        multi_doshas = [DOSHA_MAP[i] for i, p in enumerate(probs) if p >= CONFIDENCE_THRESHOLD]

        if multi_doshas:
            bot_reply = ""
            for dosha in multi_doshas:
                bot_reply += f"### 🧘 Predicted Dosha: **{dosha}**\n"
                bot_reply += f"🩺 About {dosha}: {DOSHA_INFO[dosha]}\n"
                bot_reply += f"🌿 Remedies: {DOSHA_REMEDIES[dosha]}\n\n"
            bot_reply += "*This guidance is general and not a medical diagnosis.*"
        else:
            # Low confidence → GPT reasoning
            bot_reply = ask_gpt(user_input)

    # ---------------- SAVE & DISPLAY ----------------
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
