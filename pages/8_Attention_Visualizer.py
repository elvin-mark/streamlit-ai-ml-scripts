import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from transformers import (
    BertTokenizer,
    BertModel,
    GPT2Tokenizer,
    GPT2Model,
    RobertaTokenizer,
    RobertaModel,
    DistilBertTokenizer,
    DistilBertModel,
)
import warnings

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="🧠 Transformer Attention Visualizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 Advanced Transformer Attention Visualizer")
st.markdown(
    """
*Explore the inner workings of transformer models by visualizing self-attention patterns across layers and heads.*
"""
)

# Model configurations
MODEL_CONFIGS = {
    "BERT Base": {
        "tokenizer": BertTokenizer,
        "model": BertModel,
        "name": "bert-base-uncased",
        "description": "Bidirectional encoder, great for understanding context",
    },
    "DistilBERT": {
        "tokenizer": DistilBertTokenizer,
        "model": DistilBertModel,
        "name": "distilbert-base-uncased",
        "description": "Lighter version of BERT, faster inference",
    },
    "RoBERTa Base": {
        "tokenizer": RobertaTokenizer,
        "model": RobertaModel,
        "name": "roberta-base",
        "description": "Robustly optimized BERT, different training approach",
    },
    "GPT-2 Small": {
        "tokenizer": GPT2Tokenizer,
        "model": GPT2Model,
        "name": "gpt2",
        "description": "Autoregressive model, good for text generation patterns",
    },
}

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")

    # Model selection
    selected_model = st.selectbox(
        "🤖 Choose Model",
        list(MODEL_CONFIGS.keys()),
        help="Different models show different attention patterns",
    )

    st.info(f"**{selected_model}**: {MODEL_CONFIGS[selected_model]['description']}")

    st.divider()

    # Analysis options
    st.subheader("🔍 Analysis Options")
    show_token_details = st.checkbox("Show Token Details", value=True)
    show_attention_stats = st.checkbox("Show Attention Statistics", value=True)
    show_head_comparison = st.checkbox("Compare Multiple Heads", value=False)
    normalize_attention = st.checkbox("Normalize Attention Weights", value=False)

    st.divider()

    # Visualization options
    st.subheader("🎨 Visualization")
    color_scheme = st.selectbox(
        "Color Scheme",
        ["viridis", "plasma", "inferno", "magma", "blues", "reds", "greens"],
    )

    show_colorbar = st.checkbox("Show Color Bar", value=True)
    heatmap_size = st.slider("Heatmap Size", 400, 800, 600, step=50)


# Load model function with caching
@st.cache_resource
def load_transformer_model(model_name):
    """Load tokenizer and model with error handling"""
    try:
        config = MODEL_CONFIGS[model_name]
        tokenizer_class = config["tokenizer"]
        model_class = config["model"]
        model_path = config["name"]

        # Handle GPT-2 special case (no pad token)
        if "gpt2" in model_path:
            tokenizer = tokenizer_class.from_pretrained(model_path)
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer = tokenizer_class.from_pretrained(model_path)

        model = model_class.from_pretrained(model_path, output_attentions=True)
        model.eval()

        return tokenizer, model, None
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"


# Text preprocessing
def preprocess_text(text, max_length=128):
    """Clean and prepare text for processing"""
    text = text.strip()
    if len(text) > max_length * 4:  # Rough character limit
        text = text[: max_length * 4] + "..."
        st.warning(f"Text truncated to ~{max_length * 4} characters for performance")
    return text


# Attention analysis functions
def get_attention_data(text, tokenizer, model, max_length=128):
    """Get attention data from model"""
    try:
        # Tokenize with attention to length
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        attention = torch.stack(
            outputs.attentions
        )  # (layers, batch, heads, seq_len, seq_len)
        attention = attention.squeeze(1)  # remove batch dimension

        return tokens, attention, None
    except Exception as e:
        return None, None, f"Error processing text: {str(e)}"


def create_attention_heatmap(
    attention_matrix, tokens, title, color_scheme, show_colorbar, size
):
    """Create interactive attention heatmap using Plotly"""
    fig = go.Figure(
        data=go.Heatmap(
            z=attention_matrix,
            x=tokens,
            y=tokens,
            colorscale=color_scheme,
            showscale=show_colorbar,
            hoverongaps=False,
            hovertemplate="Query: %{y}<br>Key: %{x}<br>Attention: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Key Tokens",
        yaxis_title="Query Tokens",
        width=size,
        height=size,
        xaxis=dict(tickangle=45),
        font=dict(size=10),
    )

    return fig


def create_attention_summary_plot(attention, tokens):
    """Create summary statistics plot"""
    num_layers, num_heads, seq_len, _ = attention.shape

    # Calculate attention statistics
    layer_attention_means = []
    layer_attention_stds = []

    for layer in range(num_layers):
        layer_attn = attention[layer].mean(dim=0)  # Average over heads
        layer_attention_means.append(layer_attn.mean().item())
        layer_attention_stds.append(layer_attn.std().item())

    # Create subplot
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Attention by Layer",
            "Attention Distribution",
            "Head Diversity",
            "Token Attention Received",
        ),
        specs=[[{"secondary_y": True}, {}], [{}, {}]],
    )

    # Plot 1: Layer-wise attention
    fig.add_trace(
        go.Scatter(
            x=list(range(num_layers)),
            y=layer_attention_means,
            mode="lines+markers",
            name="Mean Attention",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(num_layers)),
            y=layer_attention_stds,
            mode="lines+markers",
            name="Std Attention",
            yaxis="y2",
        ),
        row=1,
        col=1,
    )

    # Plot 2: Attention distribution
    all_attention = attention.flatten().numpy()
    fig.add_trace(
        go.Histogram(x=all_attention, nbinsx=50, name="Attention Distribution"),
        row=1,
        col=2,
    )

    # Plot 3: Head diversity (entropy)
    head_entropies = []
    for layer in range(num_layers):
        for head in range(num_heads):
            attn_head = attention[layer, head]
            # Calculate entropy for each query
            entropy = -torch.sum(attn_head * torch.log(attn_head + 1e-10), dim=-1)
            head_entropies.append(entropy.mean().item())

    fig.add_trace(
        go.Scatter(y=head_entropies, mode="markers", name="Head Entropy"), row=2, col=1
    )

    # Plot 4: Token attention received
    total_attention_received = attention.sum(
        dim=(0, 1, 2)
    ).numpy()  # Sum over all layers, heads, queries
    fig.add_trace(
        go.Bar(x=tokens, y=total_attention_received, name="Total Attention Received"),
        row=2,
        col=2,
    )

    fig.update_layout(height=600, showlegend=True)
    fig.update_xaxes(title_text="Layer", row=1, col=1)
    fig.update_xaxes(title_text="Attention Weight", row=1, col=2)
    fig.update_xaxes(title_text="Head Index", row=2, col=1)
    fig.update_xaxes(title_text="Tokens", row=2, col=2)

    return fig


def create_head_comparison(attention, tokens, layer, selected_heads, color_scheme):
    """Create comparison of multiple attention heads"""
    num_heads_to_show = min(len(selected_heads), 4)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"Head {head}" for head in selected_heads[:num_heads_to_show]],
    )

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for idx, head in enumerate(selected_heads[:num_heads_to_show]):
        row, col = positions[idx]
        attn_matrix = attention[layer, head].detach().numpy()

        fig.add_trace(
            go.Heatmap(
                z=attn_matrix,
                x=tokens,
                y=tokens,
                colorscale=color_scheme,
                showscale=(idx == 0),
                hoverongaps=False,
                hovertemplate=f"Head {head}<br>Query: %{{y}}<br>Key: %{{x}}<br>Attention: %{{z:.3f}}<extra></extra>",
            ),
            row=row,
            col=col,
        )

    fig.update_layout(height=600, title_text=f"Layer {layer} - Head Comparison")
    return fig


# Main interface
st.subheader("📝 Input Text")
default_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Attention is all you need for transformer models.",
    "Natural language processing has revolutionized artificial intelligence.",
    "Machine learning models can understand complex patterns in data.",
]

# Text input options
input_method = st.radio("Choose input method:", ["Type custom text", "Select example"])

if input_method == "Type custom text":
    text = st.text_area(
        "Enter your text:",
        "The quick brown fox jumps over the lazy dog.",
        height=100,
        help="Enter any text to analyze its attention patterns",
    )
else:
    text = st.selectbox("Select example text:", default_texts)

text = preprocess_text(text)

if not text.strip():
    st.warning("⚠️ Please enter some text to analyze!")
    st.stop()

# Load model
with st.spinner(f"Loading {selected_model} model..."):
    tokenizer, model, error = load_transformer_model(selected_model)

if error:
    st.error(error)
    st.stop()

# Process text
with st.spinner("Processing text and extracting attention..."):
    tokens, attention, error = get_attention_data(text, tokenizer, model)

if error:
    st.error(error)
    st.stop()

# Display basic information
num_layers, num_heads, seq_len, _ = attention.shape

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model", selected_model)
with col2:
    st.metric("Layers", num_layers)
with col3:
    st.metric("Heads per Layer", num_heads)
with col4:
    st.metric("Sequence Length", seq_len)

# Main controls
st.subheader("🎛️ Attention Controls")
col1, col2, col3 = st.columns(3)

with col1:
    layer = st.slider("🔁 Layer", 0, num_layers - 1, num_layers // 2)

with col2:
    view_mode = st.selectbox(
        "👁️ View Mode", ["Single Head", "Average All Heads", "Multi-Head Grid"]
    )

with col3:
    if view_mode == "Single Head":
        head = st.slider("🧠 Head", 0, num_heads - 1, 0)
    elif view_mode == "Multi-Head Grid":
        num_heads_show = min(st.slider("Heads to Show", 1, min(num_heads, 9), 4), 9)

# Normalize attention if requested
if normalize_attention:
    # L2 normalization
    attention = attention / (attention.norm(dim=-1, keepdim=True) + 1e-10)

# Main visualization
st.subheader("🔥 Attention Visualization")

if view_mode == "Single Head":
    attn_matrix = attention[layer, head].detach().numpy()
    fig = create_attention_heatmap(
        attn_matrix,
        tokens,
        f"{selected_model} - Layer {layer}, Head {head}",
        color_scheme,
        show_colorbar,
        heatmap_size,
    )
    st.plotly_chart(fig, use_container_width=True)

elif view_mode == "Average All Heads":
    attn_matrix = attention[layer].mean(dim=0).detach().numpy()
    fig = create_attention_heatmap(
        attn_matrix,
        tokens,
        f"{selected_model} - Layer {layer} (Average of all heads)",
        color_scheme,
        show_colorbar,
        heatmap_size,
    )
    st.plotly_chart(fig, use_container_width=True)

else:  # Multi-Head Grid
    selected_heads = list(range(num_heads_show))
    fig = create_head_comparison(attention, tokens, layer, selected_heads, color_scheme)
    st.plotly_chart(fig, use_container_width=True)

# Additional analyses
if show_head_comparison and view_mode == "Single Head":
    st.subheader("🔄 Head Comparison")
    comparison_heads = st.multiselect(
        "Select heads to compare:",
        list(range(num_heads)),
        default=[0, 1, 2, 3] if num_heads >= 4 else list(range(num_heads)),
    )

    if comparison_heads:
        fig_comp = create_head_comparison(
            attention, tokens, layer, comparison_heads, color_scheme
        )
        st.plotly_chart(fig_comp, use_container_width=True)

if show_attention_stats:
    st.subheader("📊 Attention Statistics")
    fig_stats = create_attention_summary_plot(attention, tokens)
    st.plotly_chart(fig_stats, use_container_width=True)

# Detailed information sections
col1, col2 = st.columns(2)

with col1:
    if show_token_details:
        st.subheader("🔤 Token Analysis")

        # Create token dataframe with statistics
        token_stats = []
        for i, token in enumerate(tokens):
            # Calculate statistics for this token
            attention_given = (
                attention[:, :, i, :].mean().item()
            )  # Average attention this token gives
            attention_received = (
                attention[:, :, :, i].mean().item()
            )  # Average attention this token receives

            token_stats.append(
                {
                    "Position": i,
                    "Token": token,
                    "Attention Given": f"{attention_given:.4f}",
                    "Attention Received": f"{attention_received:.4f}",
                    "Token Type": (
                        "Special"
                        if token.startswith("[") or token.startswith("<")
                        else "Word"
                    ),
                }
            )

        token_df = pd.DataFrame(token_stats)
        st.dataframe(token_df, use_container_width=True)

with col2:
    st.subheader("📐 Model Information")

    info_data = {
        "Property": [
            "Model Type",
            "Tokenizer",
            "Layers",
            "Attention Heads",
            "Sequence Length",
            "Total Parameters",
        ],
        "Value": [
            selected_model,
            MODEL_CONFIGS[selected_model]["name"],
            f"{num_layers}",
            f"{num_heads}",
            f"{seq_len}",
            f"~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M",
        ],
    }

    info_df = pd.DataFrame(info_data)
    st.dataframe(info_df, use_container_width=True, hide_index=True)

# Advanced analysis section
with st.expander("🔬 Advanced Analysis", expanded=False):
    st.subheader("Attention Pattern Analysis")

    # Calculate attention pattern metrics
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Layer-wise Attention Focus**")
        for layer_idx in range(num_layers):
            layer_attn = attention[layer_idx].mean(dim=0)  # Average over heads
            # Calculate how focused the attention is (entropy)
            entropy = -torch.sum(
                layer_attn * torch.log(layer_attn + 1e-10), dim=-1
            ).mean()
            focus_score = np.exp(-entropy.item())  # Convert to focus score (0-1)
            st.metric(f"Layer {layer_idx}", f"{focus_score:.3f}")

    with col2:
        st.write("**Head Specialization**")
        head_specializations = []
        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                head_attn = attention[layer_idx, head_idx]
                # Look for diagonal attention (self-attention)
                diagonal_strength = torch.diag(head_attn).mean().item()
                head_specializations.append(
                    {
                        "Layer": layer_idx,
                        "Head": head_idx,
                        "Self-Attention": f"{diagonal_strength:.3f}",
                    }
                )

        spec_df = pd.DataFrame(head_specializations)
        st.dataframe(spec_df.tail(10), use_container_width=True)

# Footer with explanations
st.markdown("---")
st.markdown(
    """
### 📚 Understanding Attention Visualizations

**What you're seeing:**
- **Darker colors** = Higher attention weights (model focuses more on these connections)
- **Rows (Y-axis)** = Query tokens (what's asking for attention)
- **Columns (X-axis)** = Key tokens (what's being attended to)

**Interpretation tips:**
- **Diagonal patterns** = Self-attention (tokens attending to themselves)
- **Vertical lines** = Many tokens attending to one important token
- **Horizontal lines** = One token attending to many others
- **Scattered patterns** = Distributed attention across multiple tokens

**Model differences:**
- **BERT**: Bidirectional, can attend to future tokens
- **GPT-2**: Autoregressive, can only attend to past tokens (triangular pattern)
- **RoBERTa**: Similar to BERT but with different training
- **DistilBERT**: Compressed BERT, fewer layers but similar patterns
"""
)
