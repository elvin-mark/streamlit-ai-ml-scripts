import streamlit as st
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from collections import Counter, defaultdict
import tempfile
import os
import random

# Title
st.title("📚 N-gram Generator and Token Predictor")

# Upload file
uploaded_file = st.file_uploader("Upload a `.txt` file", type="txt")

# Tokenizer setup
if uploaded_file:
    text = uploaded_file.read().decode("utf-8")

    st.subheader("🔧 Train Tokenizer on Uploaded Text")
    if st.button("Train Tokenizer"):
        # Create a temporary file with the uploaded text
        with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt") as tmp:
            tmp.write(text)
            tmp_path = tmp.name

        # Initialize and train tokenizer
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        )
        tokenizer.train([tmp_path], trainer)

        # Save tokenizer to disk to reuse
        tokenizer_path = os.path.join(tempfile.gettempdir(), "ngram_tokenizer.json")
        tokenizer.save(tokenizer_path)

        st.session_state["tokenizer_path"] = tokenizer_path
        st.success("Tokenizer trained and saved!")

# Load and use trained tokenizer
if "tokenizer_path" in st.session_state:
    tokenizer = Tokenizer.from_file(st.session_state["tokenizer_path"])

    # Tokenize the full text
    tokens = tokenizer.encode(text).tokens
    st.subheader("🧱 Tokenized Text Preview")
    st.write(tokens[:50])

    # N-gram generation
    st.subheader("📊 N-gram Generation")
    n = st.slider("Select N for N-gram", 1, 10, 2)
    ngrams = list(zip(*[tokens[i:] for i in range(n)]))
    freq = Counter(ngrams)

    # Build prediction model: (n-1)-gram → next token probabilities
    predict_dict = defaultdict(Counter)
    for ng in ngrams:
        prefix, next_token = tuple(ng[:-1]), ng[-1]
        predict_dict[prefix][next_token] += 1

    # Display most common n-grams
    most_common = freq.most_common(20)
    st.write("Most Common N-grams:")
    st.table(
        [{"N-gram": " ".join(ng), "Frequency": count} for ng, count in most_common]
    )

    # Optional bar chart
    if st.checkbox("📈 Show Bar Chart of Top N-grams"):
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.DataFrame(most_common, columns=["N-gram", "Frequency"])
        df["N-gram"] = df["N-gram"].apply(lambda x: " ".join(x))

        fig, ax = plt.subplots()
        df.plot(kind="bar", x="N-gram", y="Frequency", ax=ax, legend=False)
        st.pyplot(fig)

    # N-gram testing
    st.subheader("🔍 Test an N-gram")
    user_input = st.text_input(f"Enter a phrase of {n} tokens:")
    if user_input:
        input_tokens = tokenizer.encode(user_input).tokens
        if len(input_tokens) != n:
            st.warning(
                f"Input must have exactly {n} tokens. You provided {len(input_tokens)}."
            )
        else:
            found = tuple(input_tokens) in freq
            if found:
                st.success("✅ Found in the generated N-grams!")
            else:
                st.error("❌ Not found in the generated N-grams.")

    # Predict next token
    st.subheader("🔮 Predict Next Token")
    next_input = st.text_input(f"Enter {n-1} tokens to predict the next one:")
    if next_input:
        input_tokens = tokenizer.encode(next_input).tokens
        if len(input_tokens) != n - 1:
            st.warning(
                f"Input must have exactly {n-1} tokens. You provided {len(input_tokens)}."
            )
        else:
            key = tuple(input_tokens)
            predictions = predict_dict.get(key, {})
            if not predictions:
                st.error("❌ No prediction found for that prefix.")
            else:
                sorted_predictions = predictions.most_common(5)
                st.write("Top predictions:")
                st.table(
                    [
                        {"Token": tok, "Frequency": freq}
                        for tok, freq in sorted_predictions
                    ]
                )

    # Text generation
    st.subheader("📝 Generate Text")
    gen_input = st.text_input(f"Enter {n-1} starting tokens to generate text:")
    gen_len = st.slider("Number of tokens to generate", 1, 50, 10)
    if gen_input:
        input_tokens = tokenizer.encode(gen_input).tokens
        if len(input_tokens) != n - 1:
            st.warning(f"Start must be exactly {n-1} tokens.")
        else:
            generated = input_tokens[:]
            for _ in range(gen_len):
                key = tuple(generated[-(n - 1) :])
                candidates = predict_dict.get(key)
                if not candidates:
                    break
                next_tok = random.choices(
                    list(candidates.keys()), weights=candidates.values()
                )[0]
                generated.append(next_tok)

            st.success("📝 Generated Text:")
            st.write(" ".join(generated))
