import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Text Similarity Dashboard", layout="wide")

st.title("📝 Text Embedding Similarity Dashboard")
st.markdown(
    """
Use the **`all-MiniLM-L6-v2`** sentence-transformer model to compute and visualize text similarity interactively.
"""
)


@st.cache_resource(show_spinner=False)
def load_model():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model


model = load_model()

# --------------------------
# User Input: Text List
# --------------------------
st.sidebar.header("Input Texts")
mode = st.sidebar.radio("Input Mode", ["Single Pair", "Multiple Texts"])

if mode == "Single Pair":
    text1 = st.sidebar.text_input(
        "Text 1", "The quick brown fox jumps over the lazy dog"
    )
    text2 = st.sidebar.text_input("Text 2", "A fast brown fox leaps over a lazy dog")
    texts = [text1, text2]
else:
    multi_text = st.sidebar.text_area(
        "Enter multiple texts (one per line)",
        value="The quick brown fox\nA fast brown fox\nLazy dog sleeps\nJumping over the fence",
    )
    texts = [line.strip() for line in multi_text.splitlines() if line.strip()]
    if len(texts) < 2:
        st.warning("Please enter at least 2 texts.")

# --------------------------
# Similarity Metric
# --------------------------
st.sidebar.header("Similarity Settings")
sim_metric = st.sidebar.selectbox(
    "Similarity Metric", ["Cosine Similarity", "Euclidean Distance"]
)

# --------------------------
# Dimensionality Reduction
# --------------------------
st.sidebar.header("Embedding Visualization")
reduce_dim = st.sidebar.checkbox("Reduce dimension for visualization", value=True)
reducer = st.sidebar.selectbox("Reduction Method", ["PCA", "TSNE"])
n_components = st.sidebar.slider("Reduced Dimensions", 2, 3, 2)

# --------------------------
# Compute Embeddings
# --------------------------
if len(texts) >= 2 and all(texts):
    with st.spinner("Computing embeddings..."):
        embeddings = model.encode(texts, convert_to_tensor=True)

    embeddings_np = embeddings.cpu().numpy()

    # Similarity matrix
    if sim_metric == "Cosine Similarity":
        sim_matrix = cosine_similarity(embeddings_np)
    else:
        # Euclidean distance converted to similarity (smaller distance = higher similarity)
        dist_matrix = euclidean_distances(embeddings_np)
        # Convert distances to similarity scores (simple inversion)
        sim_matrix = 1 / (1 + dist_matrix)

    st.subheader("🔢 Pairwise Similarity Matrix")
    df_sim = pd.DataFrame(sim_matrix, index=texts, columns=texts)
    st.dataframe(df_sim.style.background_gradient(cmap="viridis"))

    # --------------------------
    # Query Similarity
    # --------------------------
    st.subheader("🔍 Query Similarity Search")
    query = st.text_input("Enter a query text:", "")
    if query.strip():
        query_emb = model.encode([query], convert_to_tensor=True).cpu().numpy()
        if sim_metric == "Cosine Similarity":
            sims = cosine_similarity(query_emb, embeddings_np)[0]
        else:
            dist = euclidean_distances(query_emb, embeddings_np)[0]
            sims = 1 / (1 + dist)
        result_df = pd.DataFrame({"Text": texts, "Similarity": sims})
        result_df = result_df.sort_values(by="Similarity", ascending=False)
        st.table(
            result_df.style.background_gradient(cmap="viridis", subset=["Similarity"])
        )

    # --------------------------
    # Embedding Visualization
    # --------------------------
    st.subheader("📊 Embedding Visualization")

    if reduce_dim:
        if reducer == "PCA":
            dr_model = PCA(n_components=n_components, random_state=42)
            reduced = dr_model.fit_transform(embeddings_np)
        else:
            dr_model = TSNE(n_components=n_components, random_state=42, init="random")
            reduced = dr_model.fit_transform(embeddings_np)

        if n_components == 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(
                reduced[:, 0],
                reduced[:, 1],
                c=np.arange(len(texts)),
                cmap="tab10",
                s=80,
            )
            for i, txt in enumerate(texts):
                ax.annotate(
                    txt,
                    (reduced[i, 0], reduced[i, 1]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    ha="left",
                    fontsize=9,
                )
            ax.set_title(f"{reducer} Visualization of Embeddings")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            st.pyplot(fig)
        else:
            import plotly.express as px

            df_vis = pd.DataFrame(reduced, columns=["Dim1", "Dim2", "Dim3"])
            df_vis["Text"] = texts
            fig = px.scatter_3d(
                df_vis, x="Dim1", y="Dim2", z="Dim3", text="Text", color="Text"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Enable dimensionality reduction to see embedding visualization.")
else:
    st.warning("Please enter at least two non-empty texts.")
