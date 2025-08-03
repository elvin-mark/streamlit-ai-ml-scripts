import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from io import BytesIO

# Page setup
st.set_page_config(page_title="🔬 PCA Digits Explorer", layout="wide")
st.title("🔬 PCA Explorer - Digits Dataset")


# --------------------------
# Load and preprocess data
# --------------------------
@st.cache_data
def load_data():
    digits = load_digits()
    X = digits.data
    y = digits.target
    return X, y, digits.images, digits.target_names


X, y, images, target_names = load_data()

st.markdown(
    "Explore dimensionality reduction using **Principal Component Analysis (PCA)** on the handwritten digits dataset (64 features per sample)."
)

# Sidebar
st.sidebar.header("⚙️ Settings")
standardize = st.sidebar.checkbox("Standardize Features", value=True)
n_components = st.sidebar.slider("Number of PCA Components", 2, min(64, X.shape[1]), 10)
show_3d = st.sidebar.checkbox("Show 3D PCA Plot", value=True)

# Standardize if needed
if standardize:
    scaler = StandardScaler()
    X_proc = scaler.fit_transform(X)
else:
    X_proc = X

# Apply PCA
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_proc)

# --------------------------
# Scree Plot (Explained Variance)
# --------------------------
st.subheader("📉 Explained Variance (Scree Plot)")
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

fig_scree = go.Figure()
fig_scree.add_trace(
    go.Bar(
        x=[f"PC{i+1}" for i in range(n_components)],
        y=explained_var,
        name="Explained Variance",
    )
)
fig_scree.add_trace(
    go.Scatter(
        x=[f"PC{i+1}" for i in range(n_components)],
        y=cumulative_var,
        mode="lines+markers",
        name="Cumulative Variance",
        line=dict(color="red"),
    )
)

fig_scree.update_layout(
    xaxis_title="Principal Components",
    yaxis_title="Variance Ratio",
    height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_scree, use_container_width=True)

# --------------------------
# 2D PCA Scatter Plot
# --------------------------
st.subheader("📌 2D PCA Scatter Plot")

fig_2d = px.scatter(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    color=y.astype(str),
    labels={"x": "PC1", "y": "PC2", "color": "Digit"},
    title="2D PCA Projection",
)
st.plotly_chart(fig_2d, use_container_width=True)

# --------------------------
# 3D PCA Plot
# --------------------------
if show_3d and n_components >= 3:
    st.subheader("🌐 3D PCA Scatter Plot")

    fig_3d = px.scatter_3d(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        z=X_pca[:, 2],
        color=y.astype(str),
        labels={"x": "PC1", "y": "PC2", "z": "PC3", "color": "Digit"},
        title="3D PCA Projection",
        opacity=0.8,
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# --------------------------
# Download Projected Data
# --------------------------
st.subheader("💾 Download Projected Data")

projected_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
projected_df["Digit"] = y

csv_buffer = BytesIO()
projected_df.to_csv(csv_buffer, index=False)
st.download_button(
    label="📥 Download PCA Data as CSV",
    data=csv_buffer.getvalue(),
    file_name="pca_digits.csv",
    mime="text/csv",
)

# --------------------------
# Bonus: Show image for selected point
# --------------------------
st.subheader("🖼️ Explore Sample Digit")

idx = st.slider("Select data index", 0, len(images) - 1, 0)
st.write(f"Digit Label: **{y[idx]}**")
st.image(images[idx] / 16.0, width=100, caption=f"Image for index {idx}")
