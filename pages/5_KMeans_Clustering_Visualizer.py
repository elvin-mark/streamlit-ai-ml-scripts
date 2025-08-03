import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_blobs, make_moons
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

st.set_page_config(page_title="🔍 K-Means Clustering Visualizer", layout="wide")
st.title("🔍 K-Means Clustering Visualizer")

# Sidebar options
st.sidebar.header("⚙️ Configuration")

dataset_option = st.sidebar.selectbox("Select Dataset", ["Iris", "Blobs", "Moons"])
n_clusters = st.sidebar.slider("Number of Clusters (K)", 1, 10, 3)
random_state = st.sidebar.number_input("Random Seed", value=42, step=1)
rerun = st.sidebar.button("🔁 Rerun Clustering")


# Load dataset
@st.cache_data
def load_data(name):
    if name == "Iris":
        data = load_iris()
        X = data.data
        labels = data.target
        feature_names = data.feature_names
    elif name == "Blobs":
        X, labels = make_blobs(n_samples=300, centers=4, random_state=42)
        feature_names = ["X0", "X1"]
    elif name == "Moons":
        X, labels = make_moons(n_samples=300, noise=0.05, random_state=42)
        feature_names = ["X0", "X1"]
    return X, labels, feature_names


X, true_labels, feature_names = load_data(dataset_option)

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce to 2D for visualization
if X_scaled.shape[1] > 2:
    pca = PCA(n_components=2)
    X_vis = pca.fit_transform(X_scaled)
    st.caption("⚠️ Dimensionality reduced to 2D using PCA for visualization.")
else:
    X_vis = X_scaled

# KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
kmeans.fit(X_scaled)
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Project centroids if PCA was used
if X_scaled.shape[1] > 2:
    centroids_vis = pca.transform(centroids)
else:
    centroids_vis = centroids

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

# Decision boundaries (only in 2D)
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
grid = np.c_[xx.ravel(), yy.ravel()]

# Map back to original space if PCA used
if X_scaled.shape[1] > 2:
    grid_orig = pca.inverse_transform(grid)
else:
    grid_orig = grid

Z = kmeans.predict(grid_orig)
Z = Z.reshape(xx.shape)
ax.contourf(
    xx,
    yy,
    Z,
    cmap=ListedColormap(["#FFDDC1", "#C1FFD7", "#C1D4FF", "#FFE0F1", "#E0C1FF"]),
    alpha=0.3,
)

# Scatter plot
scatter = ax.scatter(
    X_vis[:, 0], X_vis[:, 1], c=cluster_labels, cmap="tab10", s=50, edgecolor="k"
)
ax.scatter(
    centroids_vis[:, 0],
    centroids_vis[:, 1],
    marker="X",
    s=200,
    c="black",
    label="Centroids",
)

ax.set_title(f"K-Means Clustering (K={n_clusters})")
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.legend()
st.pyplot(fig)

# Metrics
st.subheader("📊 Clustering Metrics")
st.write(f"**Inertia (within-cluster sum of squares):** {kmeans.inertia_:.2f}")
st.write(f"**Number of iterations until convergence:** {kmeans.n_iter_}")
st.write(f"**Centroids (projected):**")
st.dataframe(centroids_vis, use_container_width=True)
