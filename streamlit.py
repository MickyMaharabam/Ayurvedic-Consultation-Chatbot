import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
import pandas as pd
import json

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AyuTalk", page_icon="🌿", layout="wide")
st.title("🌿 TriDosha Talk")
st.caption("AI-based Ayurvedic Consultation (General Guidance Only)")

MODEL_PATH = "mickymaharabam/AyuTalk_model"
CONFIDENCE_THRESHOLD = 0.4

# ---------------- OPENAI CLIENT ----------------
client = OpenAI()

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- DOSHA INFO ----------------
DOSHA_MAP = {0: "Vata", 1: "Pitta", 2: "Kapha"}

DOSHA_INFO = {
    "Vata": "Movement, dryness, anxiety, constipation.",
    "Pitta": "Metabolism, heat, acidity, inflammation.",
    "Kapha": "Structure, mucus, heaviness, lethargy."
}

DOSHA_REMEDIES = {
    "Vata": "Warm foods, routine, meditation.",
    "Pitta": "Cooling foods, hydration, calm activities.",
    "Kapha": "Light diet, exercise, avoid oily foods."
}

DOSHA_EMOJI = {"Vata": "💨", "Pitta": "🔥", "Kapha": "💧"}

# ---------------- GPT MULTI-DOSHA (FORCED) ----------------
def ask_gpt(symptoms, ml_probs=None):
    """
    GPT is FORCED to evaluate ALL doshas independently
    """
    context = ""
    if ml_probs is not None:
        context = f"""
ML model probabilities:
Vata: {ml_probs[0]:.2f}
Pitta: {ml_probs[1]:.2f}
Kapha: {ml_probs[2]:.2f}
"""

    prompt = f"""
You are an Ayurvedic expert assistant.

IMPORTANT RULES:
- Symptoms may involve MULTIPLE dosha imbalances.
- Evaluate Vata, Pitta, Kapha independently.
- Assign confidence scores between 0 and 1.
- Identify primary and secondary dosha.
- Do NOT give medical diagnosis.
- Avoid medicine names.

{context}

Symptoms:
"{symptoms}"

Respond ONLY in valid JSON:
{{
  "Vata": number,
  "Pitta": number,
  "Kapha": number,
  "primary_dosha": string,
  "secondary_dosha": string,
  "reasoning": string
}}
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    return json.loads(response.output_text)

# ---------------- SESSION ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- USER INPUT ----------------
user_input = st.chat_input("Describe your symptoms...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ---------------- ML PREDICTION ----------------
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
        probs = torch.softmax(outputs.logits, dim=1).squeeze().cpu().numpy()

    ml_doshas = [DOSHA_MAP[i] for i, p in enumerate(probs) if p >= CONFIDENCE_THRESHOLD]

    # ---------------- USE GPT FOR REFINEMENT ----------------
    gpt_result = ask_gpt(user_input, probs)

    gpt_doshas = [
        d for d in ["Vata", "Pitta", "Kapha"]
        if gpt_result[d] >= CONFIDENCE_THRESHOLD
    ]

    final_doshas = list(dict.fromkeys(ml_doshas + gpt_doshas))

    # ---------------- DISPLAY ----------------
    with st.chat_message("assistant"):
        st.markdown("### 🧠 Dosha Analysis")

        for dosha in final_doshas:
            st.markdown(f"**{DOSHA_EMOJI[dosha]} {dosha}**")
            with st.expander("About"):
                st.write(DOSHA_INFO[dosha])
            with st.expander("🌿 Remedies"):
                st.write(DOSHA_REMEDIES[dosha])

        st.markdown("### 📊 Confidence Scores")
        df = pd.DataFrame({
            "Dosha": ["Vata", "Pitta", "Kapha"],
            "Confidence": [
                gpt_result["Vata"],
                gpt_result["Pitta"],
                gpt_result["Kapha"]
            ]
        }).set_index("Dosha")

        st.bar_chart(df)

        st.markdown("### 📖 Explanation")
        st.write(gpt_result["reasoning"])

        st.info("⚠️ This is AI-generated general Ayurvedic guidance, not a medical diagnosis.")

    st.session_state.messages.append({
        "role": "assistant",
        "content": f"Primary: {gpt_result['primary_dosha']}, Secondary: {gpt_result['secondary_dosha']}"
    })
