import streamlit as st
from tokenizers import Tokenizer
import os

st.set_page_config(page_title="Tokenizer Visualizer", layout="centered")
st.title("🧠 HuggingFace Tokenizer Visualizer")

# Upload tokenizer.json
uploaded_file = st.file_uploader(
    "Upload a HuggingFace tokenizer JSON file", type=["json"]
)

if uploaded_file:
    # Save the file temporarily
    tokenizer_path = os.path.join("/tmp", uploaded_file.name)
    with open(tokenizer_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
        st.success("✅ Tokenizer loaded successfully!")

        # Input text
        user_input = st.text_area("Enter text to tokenize", height=150)

        if user_input:
            encoded = tokenizer.encode(user_input)

            st.subheader("🔠 Tokens")
            st.code(encoded.tokens)

            st.subheader("🔢 Token IDs")
            st.code(encoded.ids)

            st.subheader("🧭 Offsets (start, end)")
            st.code(encoded.offsets)

    except Exception as e:
        st.error("❌ Failed to load tokenizer or tokenize text.")
        st.exception(e)
else:
    st.info("📄 Please upload a `tokenizer.json` file to get started.")
