import streamlit as st
import torch
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="SentimentIQ · Amazon Reviews",
    page_icon="🔍",
    layout="centered"
)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0a0a0f;
    color: #e2e2e8;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 760px; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #e2e2e8 0%, #7c6af7 60%, #e040fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem;
}
.hero p {
    color: #6b6b7e;
    font-size: 0.82rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0;
}

/* ── Section labels ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #7c6af7;
    margin-bottom: 0.5rem;
}

/* ── Text area ── */
textarea {
    background: #12121c !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 10px !important;
    color: #e2e2e8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.88rem !important;
    caret-color: #7c6af7;
    resize: none !important;
    transition: border-color 0.2s ease !important;
}
textarea:focus {
    border-color: #7c6af7 !important;
    box-shadow: 0 0 0 3px rgba(124, 106, 247, 0.12) !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
/* Primary (first column) */
[data-testid="column"]:first-child .stButton > button {
    background: linear-gradient(135deg, #7c6af7, #e040fb) !important;
    border: none !important;
    color: #fff !important;
    box-shadow: 0 4px 20px rgba(124,106,247,0.35) !important;
}
[data-testid="column"]:first-child .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 28px rgba(124,106,247,0.5) !important;
}
/* Secondary (second column) */
[data-testid="column"]:last-child .stButton > button {
    background: transparent !important;
    border: 1px solid #2a2a3e !important;
    color: #6b6b7e !important;
}
[data-testid="column"]:last-child .stButton > button:hover {
    border-color: #7c6af7 !important;
    color: #e2e2e8 !important;
}

/* ── Result card ── */
.result-card {
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin: 1.5rem 0;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    animation: fadeSlideIn 0.4s ease;
}
.result-card.positive {
    background: linear-gradient(135deg, rgba(0,200,130,0.08), rgba(0,200,130,0.03));
    border: 1px solid rgba(0,200,130,0.25);
}
.result-card.negative {
    background: linear-gradient(135deg, rgba(255,77,100,0.08), rgba(255,77,100,0.03));
    border: 1px solid rgba(255,77,100,0.25);
}
.result-card.neutral {
    background: linear-gradient(135deg, rgba(255,185,50,0.08), rgba(255,185,50,0.03));
    border: 1px solid rgba(255,185,50,0.25);
}
.result-emoji { font-size: 2.4rem; }
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #6b6b7e;
    margin-bottom: 0.2rem;
}
.result-sentiment {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}
.result-card.positive .result-sentiment { color: #00c882; }
.result-card.negative .result-sentiment { color: #ff4d64; }
.result-card.neutral  .result-sentiment { color: #ffb932; }

/* ── Confidence bar chart tweaks ── */
.confidence-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #6b6b7e;
    margin: 1.4rem 0 0.6rem;
}

/* ── Divider ── */
hr { border-color: #1e1e2e !important; margin: 2.5rem 0 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #12121c !important;
    border: 1px dashed #2a2a3e !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #7c6af7 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e1e2e !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    background: linear-gradient(135deg, #7c6af7, #e040fb) !important;
    border: none !important;
    color: #fff !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.4rem !important;
    box-shadow: 0 4px 20px rgba(124,106,247,0.3) !important;
    margin-top: 0.8rem;
}

/* ── Warning / error ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    font-family: 'DM Mono', monospace !important;
}

/* ── Fade animation ── */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Batch section subheader ── */
.batch-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e2e8;
    margin-bottom: 0.3rem;
}
.batch-subtext {
    font-size: 0.78rem;
    color: #6b6b7e;
    margin-bottom: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Load Model (Cached)
# -------------------------
model_path = "distilbert_v2_large_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

with open(f"{model_path}/label_map.json") as f:
    label_map = json.load(f)

# -------------------------
# Prediction Function
# -------------------------
def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=128
    )
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return label_map[str(pred)], probs.cpu().numpy()[0]

# -------------------------
# Session State Init
# -------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# -------------------------
# Hero Header
# -------------------------
st.markdown("""
<div class="hero">
  <h1>SentimentIQ</h1>
  <p>Amazon Review · Sentiment Analysis Engine</p>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Input Area
# -------------------------
st.markdown('<div class="section-label">Review Input</div>', unsafe_allow_html=True)
review_input = st.text_area(
    label="review_text",
    label_visibility="collapsed",
    placeholder="Paste a product review here — e.g. \"This blender is an absolute beast. Smoothies in 10 seconds flat.\"",
    height=160
)

col1, col2 = st.columns([3, 1])

if col1.button("⚡ Analyze Sentiment"):
    if review_input.strip():
        with st.spinner("Running inference…"):
            sentiment, probabilities = predict_sentiment(review_input)
        st.session_state.prediction = (sentiment, probabilities)
    else:
        st.warning("Please enter a review before analyzing.")

if col2.button("✕ Clear"):
    st.session_state.prediction = None
    st.rerun()

# -------------------------
# Display Prediction
# -------------------------
if st.session_state.prediction:
    sentiment, probabilities = st.session_state.prediction

    emoji_map  = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}
    class_map  = {"Positive": "positive", "Negative": "negative", "Neutral": "neutral"}
    emoji      = emoji_map.get(sentiment, "⚪")
    card_class = class_map.get(sentiment, "neutral")

    st.markdown(f"""
    <div class="result-card {card_class}">
        <div class="result-emoji">{emoji}</div>
        <div>
            <div class="result-label">Detected Sentiment</div>
            <div class="result-sentiment">{sentiment}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="confidence-header">Confidence Breakdown</div>', unsafe_allow_html=True)

    df_probs = pd.DataFrame({
        "Sentiment": [label_map[str(i)] for i in range(len(label_map))],
        "Confidence": probabilities
    })

    # Style the bar chart with custom colors via altair
    import altair as alt
    color_scale = alt.Scale(
        domain=["Positive", "Neutral", "Negative"],
        range=["#00c882", "#ffb932", "#ff4d64"]
    )
    chart = (
        alt.Chart(df_probs)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("Sentiment:N", axis=alt.Axis(labelColor="#6b6b7e", tickColor="#1e1e2e", domainColor="#1e1e2e", labelFont="DM Mono")),
            y=alt.Y("Confidence:Q", axis=alt.Axis(format=".0%", labelColor="#6b6b7e", tickColor="#1e1e2e", domainColor="#1e1e2e", labelFont="DM Mono"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Sentiment:N", scale=color_scale, legend=None),
            tooltip=[alt.Tooltip("Sentiment"), alt.Tooltip("Confidence", format=".1%")]
        )
        .properties(height=220, background="#0a0a0f", padding={"left": 10, "top": 10, "right": 10, "bottom": 10})
        .configure_view(strokeOpacity=0)
    )
    st.altair_chart(chart, use_container_width=True)

# -------------------------
# Batch Section
# -------------------------
st.markdown("---")

st.markdown('<div class="batch-header">Batch Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="batch-subtext">Upload a CSV with a <code>review</code> column to classify multiple reviews at once.</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, quotechar='"', skipinitialspace=True, on_bad_lines='skip')

    if "review" not in df.columns:
        st.error("CSV must contain a column named `review`.")
    else:
        progress_bar = st.progress(0, text="Analyzing reviews…")
        batch_results = []
        total = len(df)

        for i, text in enumerate(df["review"]):
            sentiment, _ = predict_sentiment(str(text))
            batch_results.append(sentiment)
            progress_bar.progress((i + 1) / total, text=f"Processed {i+1} / {total}")

        progress_bar.empty()
        df["Predicted Sentiment"] = batch_results

        # Summary metrics
        counts = df["Predicted Sentiment"].value_counts()
        m1, m2, m3 = st.columns(3)
        m1.metric("🟢 Positive", counts.get("Positive", 0))
        m2.metric("🟡 Neutral",  counts.get("Neutral",  0))
        m3.metric("🔴 Negative", counts.get("Negative", 0))

        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Results CSV",
            data=csv,
            file_name="predicted_sentiments.csv",
            mime="text/csv"
        )