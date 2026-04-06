import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load("sentiment_model.pkl")

model = load_model()

st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f172a, #1e293b, #334155);
        color: white;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    .hero {
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 30px rgba(0,0,0,0.25);
        margin-bottom: 2rem;
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 10px;
    }

    .hero p {
        font-size: 18px;
        opacity: 0.95;
    }

    .custom-card {
        background: rgba(255,255,255,0.08);
        padding: 1.5rem;
        border-radius: 18px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        backdrop-filter: blur(10px);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .positive-box {
        background: linear-gradient(135deg, #16a34a, #22c55e);
        padding: 1rem;
        border-radius: 14px;
        color: white;
        font-weight: bold;
        text-align: center;
        font-size: 20px;
    }

    .negative-box {
        background: linear-gradient(135deg, #dc2626, #ef4444);
        padding: 1rem;
        border-radius: 14px;
        color: white;
        font-weight: bold;
        text-align: center;
        font-size: 20px;
    }

    .neutral-box {
        background: linear-gradient(135deg, #f59e0b, #fbbf24);
        padding: 1rem;
        border-radius: 14px;
        color: white;
        font-weight: bold;
        text-align: center;
        font-size: 20px;
    }

    textarea, .stTextArea textarea {
        border-radius: 12px !important;
        border: 2px solid #94a3b8 !important;
        padding: 12px !important;
        font-size: 16px !important;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #06b6d4, #3b82f6);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-size: 18px;
        font-weight: 600;
        transition: 0.3s;
    }

    .stButton > button:hover {
        transform: scale(1.03);
        background: linear-gradient(135deg, #0891b2, #2563eb);
    }

    .upload-box {
        background: rgba(255,255,255,0.08);
        padding: 1.5rem;
        border-radius: 18px;
        margin-top: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="hero">
        <h1>Sentiment Analysis Web App</h1>
        <p>Analyze reviews and social media posts to detect positive, negative, or neutral sentiment.</p>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("Enter Text for Prediction")
    user_input = st.text_area("Type your review or post here", height=180)

    if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            prediction = model.predict([user_input])[0]

            if prediction.lower() == "positive":
                st.markdown(f'<div class="positive-box">Predicted Sentiment: {prediction}</div>', unsafe_allow_html=True)
            elif prediction.lower() == "negative":
                st.markdown(f'<div class="negative-box">Predicted Sentiment: {prediction}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="neutral-box">Predicted Sentiment: {prediction}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("App Info")
    st.write("This app predicts sentiment from text using a trained machine learning model.")
    st.write("Supported outputs:")
    st.write("- Positive")
    st.write("- Negative")
    st.write("- Neutral")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("## Batch Prediction")

st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview")
    st.dataframe(df.head())

    if "text" in df.columns:
        df["predicted_sentiment"] = model.predict(df["text"])
        st.write("Prediction Results")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="predicted_sentiment_results.csv",
            mime="text/csv"
        )
    else:
        st.error("The uploaded CSV must contain a column named 'text'.")
st.markdown('</div>', unsafe_allow_html=True)