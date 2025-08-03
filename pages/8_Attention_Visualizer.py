import streamlit as st
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel

st.set_page_config(page_title="Transformer Attention Visualizer", layout="wide")
st.title("🧠 Transformer Attention Visualizer")

st.markdown(
    """
Explore the self-attention maps of the **BERT** model. See how each token attends to others across layers and heads.
"""
)


# --------------------------
# Load Model & Tokenizer
# --------------------------
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()

# --------------------------
# User Input
# --------------------------
text = st.text_area(
    "✏️ Input a sentence to visualize its attention map:",
    "The quick brown fox jumps over the lazy dog.",
    height=100,
)

if not text.strip():
    st.warning("Please enter a sentence above to see the attention map.")
    st.stop()

# --------------------------
# Tokenize and Get Attention
# --------------------------
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
attention = torch.stack(outputs.attentions)  # (layers, batch, heads, seq_len, seq_len)
attention = attention.squeeze(1)  # remove batch dimension

num_layers, num_heads, seq_len, _ = attention.shape

# --------------------------
# User Controls
# --------------------------
col1, col2 = st.columns([1, 3])

with col1:
    layer = st.slider("🔁 Layer", 0, num_layers - 1, 0)
    view_mode = st.radio("👁️ Attention View", ["Single Head", "Average All Heads"])

    if view_mode == "Single Head":
        head = st.slider("🧠 Head", 0, num_heads - 1, 0)


# --------------------------
# Plot Attention Heatmap(s)
# --------------------------
def plot_attention(attn_matrix, tokens, title):
    fig, ax = plt.subplots(figsize=(min(1.2 * len(tokens), 20), 6))
    sns.heatmap(
        attn_matrix,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis",
        square=True,
        cbar=True,
        ax=ax,
        linewidths=0.1,
        linecolor="white",
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Key", fontsize=12)
    ax.set_ylabel("Query", fontsize=12)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig


with col2:
    if view_mode == "Average All Heads":
        attn_matrix = attention[layer].mean(dim=0).detach().numpy()
        fig = plot_attention(
            attn_matrix, tokens, f"Layer {layer} — Averaged over all Heads"
        )
        st.pyplot(fig)
    else:
        attn_matrix = attention[layer, head].detach().numpy()
        fig = plot_attention(attn_matrix, tokens, f"Layer {layer}, Head {head}")
        st.pyplot(fig)

# --------------------------
# Extra Details
# --------------------------
with st.expander("🔍 Token List", expanded=False):
    st.write(tokens)

with st.expander("📐 Attention Shape Info", expanded=False):
    st.text(f"Shape of attention: {attention.shape}")
    st.text(f"Number of tokens: {len(tokens)}")
