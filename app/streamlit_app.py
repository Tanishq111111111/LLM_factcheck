import streamlit as st


st.set_page_config(page_title="LLM FactCheck", layout="wide")

st.title("LLM FactCheck")
st.write("Dashboard scaffold only. Build the pilot pipeline before adding UI complexity.")

st.subheader("Expected Inputs")
st.write("- Benchmark questions")
st.write("- Model predictions")
st.write("- Retrieved evidence")
st.write("- Evaluation labels and metrics")
