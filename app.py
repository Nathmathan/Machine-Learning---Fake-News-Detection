import streamlit as st
from src.predict import predict

st.title("📰 Fake News Detector")

user_input = st.text_area("Paste a news article:")

if st.button("Analyze"):
    if user_input.strip():
        result = predict(user_input)
        st.subheader(f"Prediction: {result}")
    else:
        st.warning("Please enter some text.")