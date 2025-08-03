import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go


st.set_page_config(
    page_title="📉 MLP Training & Loss Landscape Visualization", layout="wide"
)
st.title("📉 MLP Training & Loss Landscape Visualization")


# --------------------------
# Dataset prep (Digits)
# --------------------------
@st.cache_data
def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target.reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )

    return (X_train, X_test, y_train, y_test), scaler, encoder


(X_train, X_test, y_train, y_test), scaler, encoder = load_data()


# --------------------------
# MLP Functions
# --------------------------
def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return (x > 0).astype(float)


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy(y_pred, y_true):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))


# Flatten all params into one vector
def flatten_params(W1, b1, W2, b2):
    return np.concatenate([W1.flatten(), b1.flatten(), W2.flatten(), b2.flatten()])


# Unflatten vector back into params
def unflatten_params(vec, shapes):
    s1 = shapes["W1"]
    s2 = shapes["b1"]
    s3 = shapes["W2"]
    s4 = shapes["b2"]
    idx = 0
    W1 = vec[idx : idx + np.prod(s1)].reshape(s1)
    idx += np.prod(s1)
    b1 = vec[idx : idx + np.prod(s2)].reshape(s2)
    idx += np.prod(s2)
    W2 = vec[idx : idx + np.prod(s3)].reshape(s3)
    idx += np.prod(s3)
    b2 = vec[idx : idx + np.prod(s4)].reshape(s4)
    return W1, b1, W2, b2


# --------------------------
# Training with tracking
# --------------------------
def train_with_tracking(X, y, hidden_size, lr, epochs):
    input_size = X.shape[1]
    output_size = y.shape[1]

    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))

    shapes = {"W1": W1.shape, "b1": b1.shape, "W2": W2.shape, "b2": b2.shape}

    losses = []
    param_vectors = []

    for epoch in range(epochs):
        # Forward
        z1 = X.dot(W1) + b1
        a1 = relu(z1)
        z2 = a1.dot(W2) + b2
        a2 = softmax(z2)

        loss = cross_entropy(a2, y)
        losses.append(loss)

        # Save flattened params
        params_vec = flatten_params(W1, b1, W2, b2)
        param_vectors.append(params_vec)

        # Backward
        dz2 = a2 - y
        dW2 = a1.T.dot(dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2.dot(W2.T)
        dz1 = da1 * relu_deriv(z1)
        dW1 = X.T.dot(dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    param_vectors = np.array(param_vectors)
    return losses, param_vectors, shapes


# --------------------------
# Visualize Loss Landscape on PCA grid
# --------------------------
def loss_on_grid(pca, param_mean, param_std, shapes, X, y, grid_x, grid_y):
    Z = np.zeros_like(grid_x)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            # Construct param vector in PCA space
            coord = np.array([grid_x[i, j], grid_y[i, j]])
            param_vec = (
                param_mean
                + coord[0] * param_std[0] * pca.components_[0]
                + coord[1] * param_std[1] * pca.components_[1]
            )
            W1, b1, W2, b2 = unflatten_params(param_vec, shapes)

            # Forward pass
            z1 = X.dot(W1) + b1
            a1 = relu(z1)
            z2 = a1.dot(W2) + b2
            a2 = softmax(z2)
            loss = cross_entropy(a2, y)
            Z[i, j] = loss
    return Z


# --------------------------
# Sidebar controls
# --------------------------
st.sidebar.header("⚙️ Hyperparameters")
hidden_size = st.sidebar.slider("Hidden layer size", 8, 64, 16)
learning_rate = st.sidebar.slider("Learning rate", 0.001, 0.1, 0.01, 0.001)
epochs = st.sidebar.slider("Epochs", 20, 200, 100)

if st.sidebar.button("Train MLP"):
    with st.spinner("Training..."):
        losses, param_vectors, shapes = train_with_tracking(
            X_train, y_train, hidden_size, learning_rate, epochs
        )
        st.session_state.losses = losses
        st.session_state.param_vectors = param_vectors
        st.session_state.shapes = shapes
        st.success("Training completed!")

# --------------------------
# Visualizations
# --------------------------
# After training completed and we have param_vectors (shape: epochs x param_dim)

if "losses" in st.session_state and "param_vectors" in st.session_state:
    losses = st.session_state.losses
    param_vectors = st.session_state.param_vectors
    shapes = st.session_state.shapes

    st.subheader("📈 Loss Evolution")
    fig1, ax1 = plt.subplots()
    ax1.plot(losses, label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    st.pyplot(fig1)

    # Final trained weights vector
    final_params = param_vectors[-1]

    # Center param vectors by subtracting final_params to get perturbations around final
    perturbations = param_vectors - final_params

    # PCA on perturbations (so final point is at origin)
    pca = PCA(n_components=2)
    perturb_proj = pca.fit_transform(perturbations)

    st.subheader("🌄 Loss Landscape Visualization Around Final Model")

    # Create grid in PCA space around origin (final model at center)
    margin = 5
    grid_points = 30
    x_vals = np.linspace(-margin, margin, grid_points)
    y_vals = np.linspace(-margin, margin, grid_points)
    grid_x, grid_y = np.meshgrid(x_vals, y_vals)

    def loss_on_grid_around_final(pca, final_params, shapes, X, y, grid_x, grid_y):
        Z = np.zeros_like(grid_x)
        for i in range(grid_x.shape[0]):
            for j in range(grid_x.shape[1]):
                coord = np.array([grid_x[i, j], grid_y[i, j]])
                # Convert PCA 2D coordinate back to full param space perturbation
                perturb_vec = pca.inverse_transform(coord)
                param_vec = (
                    final_params + perturb_vec
                )  # add perturbation to final params
                W1, b1, W2, b2 = unflatten_params(param_vec, shapes)

                # Forward pass
                z1 = X.dot(W1) + b1
                a1 = relu(z1)
                z2 = a1.dot(W2) + b2
                a2 = softmax(z2)
                loss = cross_entropy(a2, y)
                Z[i, j] = loss
        return Z

    with st.spinner(
        "Computing loss landscape grid around final model (this may take a while)..."
    ):
        Z = loss_on_grid_around_final(
            pca, final_params, shapes, X_train, y_train, grid_x, grid_y
        )

    # 3D surface plot
    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.plot_surface(grid_x, grid_y, Z, cmap="viridis", alpha=0.8)

    # Plot training trajectory projected into perturbation PCA space
    # The final point is now at origin, so shift training trajectory accordingly
    traj_x = perturb_proj[:, 0]
    traj_y = perturb_proj[:, 1]
    traj_z = np.array(losses)

    ax2.plot(
        traj_x,
        traj_y,
        traj_z,
        color="r",
        marker="o",
        label="Training path",
        linewidth=2,
    )
    ax2.set_xlabel("PCA 1 (Perturbation)")
    ax2.set_ylabel("PCA 2 (Perturbation)")
    ax2.set_zlabel("Loss")
    ax2.set_title("Loss Landscape Around Final Model with Training Path")
    ax2.legend()

    st.pyplot(fig2)

    # Contour plot
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    contour = ax3.contourf(grid_x, grid_y, Z, cmap="viridis", levels=50)
    ax3.plot(traj_x, traj_y, "r-o", label="Training path")
    ax3.set_xlabel("PCA 1 (Perturbation)")
    ax3.set_ylabel("PCA 2 (Perturbation)")
    ax3.set_title("Loss Landscape Contour Around Final Model")
    fig3.colorbar(contour)
    ax3.legend()

    st.pyplot(fig3)

    # 3D surface plot (Plotly)
    fig2 = go.Figure(
        data=[
            go.Surface(
                z=Z,
                x=grid_x,
                y=grid_y,
                colorscale="Viridis",
                opacity=0.8,
                showscale=True,
            ),
            go.Scatter3d(
                x=traj_x,
                y=traj_y,
                z=traj_z,
                mode="lines+markers",
                line=dict(color="red", width=4),
                marker=dict(size=3),
                name="Training Path",
            ),
        ]
    )
    fig2.update_layout(
        title="Loss Landscape Around Final Model with Training Path",
        scene=dict(
            xaxis_title="PCA 1 (Perturbation)",
            yaxis_title="PCA 2 (Perturbation)",
            zaxis_title="Loss",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Contour plot (Plotly)
    fig3 = go.Figure()

    fig3.add_trace(
        go.Contour(
            z=Z,
            x=x_vals,
            y=y_vals,
            colorscale="Viridis",
            contours_coloring="heatmap",
            line_smoothing=0.85,
        )
    )

    fig3.add_trace(
        go.Scatter(
            x=traj_x,
            y=traj_y,
            mode="lines+markers",
            marker=dict(color="red"),
            name="Training Path",
        )
    )

    fig3.update_layout(
        title="Loss Landscape Contour Around Final Model",
        xaxis_title="PCA 1 (Perturbation)",
        yaxis_title="PCA 2 (Perturbation)",
        height=600,
    )

    st.plotly_chart(fig3, use_container_width=True)


else:
    st.info("Train the model to see training loss and loss landscape.")
