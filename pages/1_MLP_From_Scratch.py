import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_digits,
    make_classification,
    make_moons,
    make_circles,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import time

# Page configuration
st.set_page_config(
    page_title="🧠 Interactive MLP Playground",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 Interactive Multi-Layer Perceptron Playground")
st.markdown(
    "*Learn how neural networks work by building and training your own MLP from scratch!*"
)

# Dataset configurations
DATASETS = {
    "Iris": {
        "loader": lambda: load_iris(),
        "description": "Classic 3-class flower classification (150 samples, 4 features)",
        "color_map": {0: "red", 1: "green", 2: "blue"},
    },
    "Wine": {
        "loader": lambda: load_wine(),
        "description": "Wine classification (178 samples, 13 features)",
        "color_map": {0: "purple", 1: "orange", 2: "teal"},
    },
    "Make Moons": {
        "loader": lambda: make_moons(n_samples=300, noise=0.2, random_state=42),
        "description": "Two moons dataset - non-linear separable",
        "color_map": {0: "blue", 1: "red"},
    },
    "Make Circles": {
        "loader": lambda: make_circles(
            n_samples=300, noise=0.2, factor=0.5, random_state=42
        ),
        "description": "Concentric circles - non-linear separable",
        "color_map": {0: "cyan", 1: "magenta"},
    },
}


# Enhanced MLP class with step-by-step visualization
class InteractiveMLP:
    def __init__(
        self,
        hidden_sizes: List[int],
        activation: str = "relu",
        learning_rate: float = 0.01,
        regularization: float = 0.0,
    ):
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.regularization = regularization

        # Training history
        self.losses = []
        self.accuracies = []
        self.weights_history = []
        self.gradients_history = []

        # Layer information for visualization
        self.layer_outputs = {}
        self.layer_inputs = {}

    def _activation_func(self, x, derivative=False):
        """Activation function with derivative"""
        if self.activation == "relu":
            if derivative:
                return (x > 0).astype(float)
            return np.maximum(0, x)
        elif self.activation == "tanh":
            if derivative:
                return 1 - np.tanh(x) ** 2
            return np.tanh(x)
        elif self.activation == "sigmoid":
            sigmoid = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            if derivative:
                return sigmoid * (1 - sigmoid)
            return sigmoid
        elif self.activation == "leaky_relu":
            if derivative:
                return np.where(x > 0, 1, 0.01)
            return np.where(x > 0, x, 0.01 * x)

    def _softmax(self, x):
        """Softmax activation for output layer"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _cross_entropy(self, y_pred, y_true):
        """Cross-entropy loss"""
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

    def _initialize_weights(self, input_size: int, output_size: int):
        """Initialize weights with proper scaling"""
        self.layers = []
        layer_sizes = [input_size] + self.hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            # Xavier/He initialization
            if self.activation == "relu":
                scale = np.sqrt(2.0 / layer_sizes[i])
            else:
                scale = np.sqrt(1.0 / layer_sizes[i])

            W = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros((1, layer_sizes[i + 1]))
            self.layers.append({"W": W, "b": b})

    def _forward_pass(self, X, store_activations=False):
        """Forward pass with optional activation storage"""
        activations = [X]
        z_values = []

        for i, layer in enumerate(self.layers):
            z = activations[-1].dot(layer["W"]) + layer["b"]
            z_values.append(z)

            if i == len(self.layers) - 1:  # Output layer
                a = self._softmax(z)
            else:  # Hidden layers
                a = self._activation_func(z)

            activations.append(a)

            # Store for visualization
            if store_activations:
                self.layer_inputs[f"layer_{i}"] = z
                self.layer_outputs[f"layer_{i}"] = a

        return activations, z_values

    def _backward_pass(self, X, y, activations, z_values):
        """Backward pass with gradient computation"""
        gradients = []

        # Output layer gradient
        dz = activations[-1] - y
        for i in range(len(self.layers) - 1, -1, -1):
            dW = activations[i].T.dot(dz) + self.regularization * self.layers[i]["W"]
            db = np.sum(dz, axis=0, keepdims=True)
            gradients.insert(0, {"dW": dW, "db": db})

            if i > 0:  # Not input layer
                da = dz.dot(self.layers[i]["W"].T)
                dz = da * self._activation_func(z_values[i - 1], derivative=True)

        return gradients

    def fit(self, X, y, epochs: int, verbose: bool = False):
        """Train the MLP"""
        self._initialize_weights(X.shape[1], y.shape[1])

        for epoch in range(epochs):
            # Forward pass
            activations, z_values = self._forward_pass(X, store_activations=True)

            # Compute loss
            loss = self._cross_entropy(activations[-1], y)
            if self.regularization > 0:
                # Add L2 regularization
                reg_loss = sum(np.sum(layer["W"] ** 2) for layer in self.layers)
                loss += 0.5 * self.regularization * reg_loss

            # Compute accuracy
            predictions = np.argmax(activations[-1], axis=1)
            true_labels = np.argmax(y, axis=1)
            accuracy = accuracy_score(true_labels, predictions)

            # Store history
            self.losses.append(loss)
            self.accuracies.append(accuracy)

            # Backward pass
            gradients = self._backward_pass(X, y, activations, z_values)

            # Store gradients for visualization
            self.gradients_history.append([g["dW"].copy() for g in gradients])

            # Update weights
            for layer, grad in zip(self.layers, gradients):
                layer["W"] -= self.learning_rate * grad["dW"]
                layer["b"] -= self.learning_rate * grad["db"]

            # Store weights for visualization
            self.weights_history.append([layer["W"].copy() for layer in self.layers])

            if verbose and (epoch + 1) % (epochs // 10) == 0:
                st.write(
                    f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}, Accuracy: {accuracy:.3f}"
                )

        return self

    def predict(self, X):
        """Make predictions"""
        activations, _ = self._forward_pass(X)
        return np.argmax(activations[-1], axis=1)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        activations, _ = self._forward_pass(X)
        return activations[-1]


# Data loading function
@st.cache_data
def load_dataset(dataset_name: str):
    """Load and preprocess dataset"""
    if dataset_name in ["Make Moons", "Make Circles"]:
        X, y = DATASETS[dataset_name]["loader"]()
    else:
        dataset = DATASETS[dataset_name]["loader"]()
        X, y = dataset.data, dataset.target

    # Encode labels
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=np.argmax(y_encoded, axis=1),
    )

    return (X_train, X_test, y_train, y_test), scaler, encoder, dataset_name


# Visualization functions
def create_network_architecture_plot(
    input_size: int, hidden_sizes: List[int], output_size: int
):
    """Create network architecture visualization"""
    fig = go.Figure()

    # Calculate positions
    max_layer_size = max([input_size] + hidden_sizes + [output_size])
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    layer_names = (
        ["Input"] + [f"Hidden {i+1}" for i in range(len(hidden_sizes))] + ["Output"]
    )

    colors = ["lightblue", "lightgreen", "lightcoral", "lightyellow", "lightpink"]

    # Draw nodes
    for layer_idx, (size, name) in enumerate(zip(layer_sizes, layer_names)):
        y_positions = np.linspace(-max_layer_size / 2, max_layer_size / 2, size)
        x_position = layer_idx * 2

        for node_idx, y_pos in enumerate(y_positions):
            fig.add_trace(
                go.Scatter(
                    x=[x_position],
                    y=[y_pos],
                    mode="markers",
                    marker=dict(size=30, color=colors[layer_idx % len(colors)]),
                    name=f"{name} Node {node_idx+1}" if size <= 10 else name,
                    showlegend=(node_idx == 0),
                    hovertemplate=f"{name}<br>Node {node_idx+1}<extra></extra>",
                )
            )

    # Draw connections (simplified for readability)
    for layer_idx in range(len(layer_sizes) - 1):
        current_size = layer_sizes[layer_idx]
        next_size = layer_sizes[layer_idx + 1]

        if current_size <= 10 and next_size <= 10:  # Only draw if manageable
            current_y = np.linspace(
                -max_layer_size / 2, max_layer_size / 2, current_size
            )
            next_y = np.linspace(-max_layer_size / 2, max_layer_size / 2, next_size)

            for i, y1 in enumerate(current_y):
                for j, y2 in enumerate(next_y):
                    fig.add_trace(
                        go.Scatter(
                            x=[layer_idx * 2, (layer_idx + 1) * 2],
                            y=[y1, y2],
                            mode="lines",
                            line=dict(color="gray", width=1),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

    fig.update_layout(
        title="Neural Network Architecture",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400,
        plot_bgcolor="white",
    )

    return fig


def create_training_dashboard(mlp: InteractiveMLP):
    """Create comprehensive training dashboard"""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Training Loss",
            "Training Accuracy",
            "Weight Evolution",
            "Gradient Norms",
        ),
        specs=[[{}, {}], [{}, {}]],
    )

    # Loss plot
    fig.add_trace(
        go.Scatter(y=mlp.losses, mode="lines", name="Loss", line=dict(color="red")),
        row=1,
        col=1,
    )

    # Accuracy plot
    fig.add_trace(
        go.Scatter(
            y=mlp.accuracies, mode="lines", name="Accuracy", line=dict(color="blue")
        ),
        row=1,
        col=2,
    )

    # Weight evolution (norm of first layer weights)
    if mlp.weights_history:
        weight_norms = [np.linalg.norm(weights[0]) for weights in mlp.weights_history]
        fig.add_trace(
            go.Scatter(
                y=weight_norms,
                mode="lines",
                name="Weight Norm",
                line=dict(color="green"),
            ),
            row=2,
            col=1,
        )

    # Gradient norms
    if mlp.gradients_history:
        grad_norms = [np.linalg.norm(grads[0]) for grads in mlp.gradients_history]
        fig.add_trace(
            go.Scatter(
                y=grad_norms,
                mode="lines",
                name="Gradient Norm",
                line=dict(color="purple"),
            ),
            row=2,
            col=2,
        )

    fig.update_layout(height=600, showlegend=False)
    return fig


def create_decision_boundary_plot(
    mlp: InteractiveMLP, X, y, scaler, title="Decision Boundary"
):
    """Create decision boundary visualization for 2D data"""
    if X.shape[1] != 2:
        return None

    # Create mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Make predictions
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = mlp.predict_proba(mesh_points)
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

    # Create plot
    fig = go.Figure()

    # Add contour
    fig.add_trace(
        go.Contour(
            z=Z,
            x=np.arange(x_min, x_max, h),
            y=np.arange(y_min, y_max, h),
            colorscale="RdYlBu",
            opacity=0.3,
            showscale=False,
            contours_coloring="heatmap",
        )
    )

    # Add data points
    unique_labels = np.unique(np.argmax(y, axis=1))
    colors = ["red", "blue", "green", "purple", "orange"]

    for i, label in enumerate(unique_labels):
        mask = np.argmax(y, axis=1) == label
        fig.add_trace(
            go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode="markers",
                marker=dict(color=colors[i % len(colors)], size=8),
                name=f"Class {label}",
            )
        )

    fig.update_layout(
        title=title, xaxis_title="Feature 1", yaxis_title="Feature 2", height=400
    )

    return fig


def create_activation_heatmap(mlp: InteractiveMLP, layer_name: str):
    """Create heatmap of layer activations"""
    if layer_name not in mlp.layer_outputs:
        return None

    activations = mlp.layer_outputs[layer_name]

    fig = go.Figure(
        data=go.Heatmap(z=activations.T, colorscale="Viridis", showscale=True)
    )

    fig.update_layout(
        title=f"Activations in {layer_name}",
        xaxis_title="Sample",
        yaxis_title="Neuron",
        height=300,
    )

    return fig


# Sidebar configuration
with st.sidebar:
    st.header("🎛️ MLP Configuration")

    # Dataset selection
    dataset_name = st.selectbox(
        "📊 Dataset", list(DATASETS.keys()), help="Choose dataset for training"
    )

    st.info(DATASETS[dataset_name]["description"])

    st.divider()

    # Architecture settings
    st.subheader("🏗️ Network Architecture")

    num_hidden_layers = st.slider("Number of Hidden Layers", 1, 3, 1)

    hidden_sizes = []
    for i in range(num_hidden_layers):
        size = st.slider(f"Hidden Layer {i+1} Size", 2, 32, 8, key=f"hidden_{i}")
        hidden_sizes.append(size)

    activation = st.selectbox(
        "Activation Function",
        ["relu", "tanh", "sigmoid", "leaky_relu"],
        help="Activation function for hidden layers",
    )

    st.divider()

    # Training settings
    st.subheader("🏋️ Training Settings")
    learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, step=0.001)
    epochs = st.slider("Training Epochs", 10, 500, 100, step=10)
    regularization = st.slider("L2 Regularization", 0.0, 0.1, 0.0, step=0.001)

    st.divider()

    # Visualization options
    st.subheader("📊 Visualization Options")
    show_architecture = st.checkbox("Show Network Architecture", value=True)
    show_training_curves = st.checkbox("Show Training Progress", value=True)
    show_decision_boundary = st.checkbox("Show Decision Boundary", value=True)
    show_activations = st.checkbox("Show Layer Activations", value=False)

# Main content
st.subheader("🎯 How Multi-Layer Perceptrons Work")

with st.expander("📚 Learn About MLPs", expanded=False):
    st.markdown(
        """
    ### 🧠 What is a Multi-Layer Perceptron?
    
    A **Multi-Layer Perceptron (MLP)** is a type of artificial neural network that consists of:
    - **Input Layer**: Receives the input features
    - **Hidden Layer(s)**: Process information using weights, biases, and activation functions
    - **Output Layer**: Produces the final predictions
    
    ### 🔄 How Training Works
    
    1. **Forward Pass**: 
       - Input flows through the network
       - Each layer applies: `output = activation(weights × input + bias)`
       - Final layer uses softmax for probability distribution
    
    2. **Loss Calculation**: 
       - Compare predictions with true labels
       - Use cross-entropy loss for classification
    
    3. **Backward Pass (Backpropagation)**:
       - Calculate gradients using chain rule
       - Update weights to minimize loss: `weight = weight - learning_rate × gradient`
    
    ### 🎛️ Key Hyperparameters
    
    - **Hidden Layer Size**: More neurons = more capacity, but risk of overfitting
    - **Learning Rate**: Too high = unstable, too low = slow convergence
    - **Activation Function**: Determines how neurons process information
    - **Regularization**: Prevents overfitting by penalizing large weights
    """
    )

# Load dataset
(X_train, X_test, y_train, y_test), scaler, encoder, dataset_name = load_dataset(
    dataset_name
)

# Display dataset info
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Training Samples", len(X_train))
with col2:
    st.metric("Test Samples", len(X_test))
with col3:
    st.metric("Features", X_train.shape[1])
with col4:
    st.metric("Classes", y_train.shape[1])

# Show network architecture
if show_architecture:
    st.subheader("🏗️ Network Architecture")
    arch_fig = create_network_architecture_plot(
        X_train.shape[1], hidden_sizes, y_train.shape[1]
    )
    st.plotly_chart(arch_fig, use_container_width=True)

# Training section
st.subheader("🚀 Train Your MLP")

if st.button("🏋️ Start Training", type="primary"):

    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Create and train MLP
    mlp = InteractiveMLP(
        hidden_sizes=hidden_sizes,
        activation=activation,
        learning_rate=learning_rate,
        regularization=regularization,
    )

    # Training with progress updates
    start_time = time.time()

    status_text.text("Initializing neural network...")
    mlp._initialize_weights(X_train.shape[1], y_train.shape[1])

    # Manual training loop with progress updates
    for epoch in range(epochs):
        # Forward pass
        activations, z_values = mlp._forward_pass(X_train, store_activations=True)

        # Compute loss and accuracy
        loss = mlp._cross_entropy(activations[-1], y_train)
        if regularization > 0:
            reg_loss = sum(np.sum(layer["W"] ** 2) for layer in mlp.layers)
            loss += 0.5 * regularization * reg_loss

        predictions = np.argmax(activations[-1], axis=1)
        true_labels = np.argmax(y_train, axis=1)
        accuracy = accuracy_score(true_labels, predictions)

        # Store history
        mlp.losses.append(loss)
        mlp.accuracies.append(accuracy)

        # Backward pass and update
        gradients = mlp._backward_pass(X_train, y_train, activations, z_values)
        mlp.gradients_history.append([g["dW"].copy() for g in gradients])

        for layer, grad in zip(mlp.layers, gradients):
            layer["W"] -= learning_rate * grad["dW"]
            layer["b"] -= learning_rate * grad["db"]

        mlp.weights_history.append([layer["W"].copy() for layer in mlp.layers])

        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(
            f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, Accuracy: {accuracy:.3f}"
        )

        # Early stopping for demo purposes
        if accuracy > 0.95 and epoch > epochs // 4:
            st.info(f"Early stopping at epoch {epoch+1} due to high accuracy!")
            break

    training_time = time.time() - start_time

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    # Store in session state
    st.session_state.mlp = mlp
    st.session_state.training_time = training_time
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.scaler = scaler
    st.session_state.encoder = encoder

    st.success(f"✅ Training completed in {training_time:.2f} seconds!")

# Results visualization
if "mlp" in st.session_state:
    mlp = st.session_state.mlp
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test

    # Performance metrics
    train_pred = mlp.predict(X_train)
    test_pred = mlp.predict(X_test)
    train_acc = accuracy_score(np.argmax(y_train, axis=1), train_pred)
    test_acc = accuracy_score(np.argmax(y_test, axis=1), test_pred)

    st.subheader("📊 Training Results")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Final Train Loss", f"{mlp.losses[-1]:.4f}")
    with col2:
        st.metric("Train Accuracy", f"{train_acc:.3f}")
    with col3:
        st.metric("Test Accuracy", f"{test_acc:.3f}")
    with col4:
        st.metric("Training Time", f"{st.session_state.training_time:.2f}s")

    # Training dashboard
    if show_training_curves:
        st.subheader("📈 Training Progress")
        dashboard_fig = create_training_dashboard(mlp)
        st.plotly_chart(dashboard_fig, use_container_width=True)

    # Decision boundary (for 2D data)
    if show_decision_boundary and X_train.shape[1] == 2:
        st.subheader("🎯 Decision Boundary")
        boundary_fig = create_decision_boundary_plot(
            mlp, X_train, y_train, st.session_state.scaler
        )
        if boundary_fig:
            st.plotly_chart(boundary_fig, use_container_width=True)

    # Layer activations
    if show_activations and mlp.layer_outputs:
        st.subheader("🔥 Layer Activations")
        layer_options = list(mlp.layer_outputs.keys())
        selected_layer = st.selectbox("Select Layer", layer_options)

        if selected_layer:
            activation_fig = create_activation_heatmap(mlp, selected_layer)
            if activation_fig:
                st.plotly_chart(activation_fig, use_container_width=True)

    # Interactive prediction
    st.subheader("🎮 Interactive Prediction")

    if dataset_name == "Iris":
        feature_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
        class_names = ["Setosa", "Versicolor", "Virginica"]
        default_values = [5.1, 3.5, 1.4, 0.2]
        feature_ranges = [(4.0, 8.0), (2.0, 5.0), (1.0, 7.0), (0.0, 3.0)]
    else:
        feature_names = [f"Feature {i+1}" for i in range(X_train.shape[1])]
        class_names = [f"Class {i}" for i in range(y_train.shape[1])]
        default_values = [0.0] * X_train.shape[1]
        feature_ranges = [(-3.0, 3.0)] * X_train.shape[1]

    # Create input sliders
    cols = st.columns(min(4, len(feature_names)))
    test_input = []

    for i, (name, (min_val, max_val), default) in enumerate(
        zip(feature_names, feature_ranges, default_values)
    ):
        col_idx = i % len(cols)
        with cols[col_idx]:
            value = st.slider(name, min_val, max_val, default, 0.1, key=f"feature_{i}")
            test_input.append(value)

    # Make prediction
    if len(test_input) == X_train.shape[1]:
        # Scale input
        test_scaled = st.session_state.scaler.transform([test_input])

        # Get prediction probabilities
        probabilities = mlp.predict_proba(test_scaled)[0]
        predicted_class = np.argmax(probabilities)

        st.markdown("#### 🎯 Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Class", class_names[predicted_class])
            st.metric("Confidence", f"{probabilities[predicted_class]:.3f}")

        with col2:
            # Probability bar chart
            prob_df = pd.DataFrame({"Class": class_names, "Probability": probabilities})

            fig_prob = px.bar(
                prob_df,
                x="Class",
                y="Probability",
                title="Class Probabilities",
                color="Probability",
                color_continuous_scale="viridis",
            )
            fig_prob.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_prob, use_container_width=True)

    # Confusion matrix
    if y_train.shape[1] <= 10:  # Only for reasonable number of classes
        st.subheader("📊 Confusion Matrix")
        cm = confusion_matrix(np.argmax(y_test, axis=1), test_pred)

        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=class_names,
                y=class_names,
                colorscale="Blues",
                showscale=True,
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16},
                hoverongaps=False,
            )
        )

        fig_cm.update_layout(
            title="Test Set Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400,
        )

        st.plotly_chart(fig_cm, use_container_width=True)

    # Detailed analysis
    with st.expander("📊 Detailed Analysis", expanded=False):

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏗️ Model Architecture Summary")

            # Architecture table
            layer_info = []
            layer_sizes = [X_train.shape[1]] + hidden_sizes + [y_train.shape[1]]
            layer_names = (
                ["Input"]
                + [f"Hidden {i+1}" for i in range(len(hidden_sizes))]
                + ["Output"]
            )
            activations_list = ["None"] + [activation] * len(hidden_sizes) + ["Softmax"]

            total_params = 0
            for i in range(len(layer_sizes) - 1):
                params = (
                    layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
                )  # weights + biases
                total_params += params
                layer_info.append(
                    {
                        "Layer": f"{layer_names[i]} → {layer_names[i+1]}",
                        "Shape": f"({layer_sizes[i]}, {layer_sizes[i+1]})",
                        "Parameters": f"{params:,}",
                        "Activation": activations_list[i + 1],
                    }
                )

            arch_df = pd.DataFrame(layer_info)
            st.dataframe(arch_df, use_container_width=True, hide_index=True)
            st.metric("Total Parameters", f"{total_params:,}")

        with col2:
            st.subheader("📈 Training Statistics")

            # Training statistics
            stats_info = [
                {"Metric": "Initial Loss", "Value": f"{mlp.losses[0]:.6f}"},
                {"Metric": "Final Loss", "Value": f"{mlp.losses[-1]:.6f}"},
                {
                    "Metric": "Loss Reduction",
                    "Value": f"{((mlp.losses[0] - mlp.losses[-1]) / mlp.losses[0] * 100):.1f}%",
                },
                {
                    "Metric": "Best Train Accuracy",
                    "Value": f"{max(mlp.accuracies):.4f}",
                },
                {"Metric": "Final Test Accuracy", "Value": f"{test_acc:.4f}"},
                {"Metric": "Epochs Completed", "Value": f"{len(mlp.losses)}"},
                {"Metric": "Final Learning Rate", "Value": f"{learning_rate:.4f}"},
                {"Metric": "Regularization", "Value": f"{regularization:.4f}"},
            ]

            stats_df = pd.DataFrame(stats_info)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # Weight analysis
        st.subheader("⚖️ Weight Analysis")

        if mlp.weights_history:
            # Weight statistics over time
            weight_stats = []
            for epoch, weights in enumerate(
                mlp.weights_history[:: max(1, len(mlp.weights_history) // 20)]
            ):  # Sample points
                for layer_idx, weight_matrix in enumerate(weights):
                    weight_stats.append(
                        {
                            "Epoch": epoch * max(1, len(mlp.weights_history) // 20),
                            "Layer": f"Layer {layer_idx + 1}",
                            "Mean Weight": np.mean(weight_matrix),
                            "Weight Std": np.std(weight_matrix),
                            "Max Weight": np.max(weight_matrix),
                            "Min Weight": np.min(weight_matrix),
                        }
                    )

            weight_df = pd.DataFrame(weight_stats)

            # Weight evolution plot
            fig_weights = go.Figure()

            for layer_idx in range(len(hidden_sizes) + 1):
                layer_data = weight_df[weight_df["Layer"] == f"Layer {layer_idx + 1}"]
                fig_weights.add_trace(
                    go.Scatter(
                        x=layer_data["Epoch"],
                        y=layer_data["Weight Std"],
                        mode="lines",
                        name=f"Layer {layer_idx + 1} Weight Std",
                    )
                )

            fig_weights.update_layout(
                title="Weight Standard Deviation Evolution",
                xaxis_title="Epoch",
                yaxis_title="Weight Standard Deviation",
                height=400,
            )

            st.plotly_chart(fig_weights, use_container_width=True)

else:
    # Educational content when no model is trained
    st.info(
        "👆 Configure your MLP settings in the sidebar and click 'Start Training' to see the magic happen!"
    )

    st.markdown(
        """
    ## 🎯 What You'll Learn
    
    ### 🧠 **Neural Network Fundamentals**
    - How neurons process information with weights and biases
    - The role of activation functions in introducing non-linearity
    - Forward propagation: from input to prediction
    - Backpropagation: learning from mistakes
    
    ### 📊 **Training Dynamics**
    - Watch loss decrease and accuracy improve in real-time
    - See how different hyperparameters affect learning
    - Understand the relationship between model complexity and performance
    - Visualize how weights evolve during training
    
    ### 🎨 **Interactive Visualizations**
    - **Network Architecture**: See your network structure
    - **Training Progress**: Real-time loss and accuracy curves
    - **Decision Boundaries**: How your model separates classes (2D data)
    - **Layer Activations**: What each layer learns
    - **Weight Evolution**: How parameters change during training
    
    ### 🔧 **Hyperparameter Effects**
    
    | Parameter | Effect | Tips |
    |-----------|--------|------|
    | **Hidden Layer Size** | More neurons = more capacity | Start small, increase if underfitting |
    | **Learning Rate** | Controls step size in optimization | 0.01 is often a good start |
    | **Activation Function** | Determines neuron behavior | ReLU is most common, Tanh for centered data |
    | **Regularization** | Prevents overfitting | Use when training accuracy >> test accuracy |
    | **Number of Layers** | Model depth and complexity | 1-2 hidden layers sufficient for most problems |
    
    ### 🎮 **Interactive Features**
    
    1. **Real-time Prediction**: Test your trained model with custom inputs
    2. **Architecture Visualization**: See your network structure
    3. **Training Monitoring**: Watch learning happen step by step
    4. **Performance Analysis**: Comprehensive metrics and confusion matrices
    5. **Weight Analysis**: Understand what your model learned
    
    ### 📚 **Educational Datasets**
    
    - **Iris**: Perfect for beginners - small, well-behaved dataset
    - **Wine**: More features, practice with higher dimensionality
    - **Make Moons**: Learn about non-linear decision boundaries
    - **Make Circles**: Challenge your model with complex patterns
    
    ## 🚀 Ready to Start?
    
    1. Choose a dataset from the sidebar
    2. Configure your network architecture
    3. Set training parameters
    4. Click "Start Training" and watch the magic!
    5. Experiment with different settings to see how they affect performance
    
    *This playground implements everything from scratch using NumPy, so you can see exactly how neural networks work under the hood!*
    """
    )

# Footer with additional information
st.markdown("---")
st.markdown(
    """
### 🔬 **Implementation Details**

This MLP playground is built entirely with **NumPy** to show you exactly how neural networks work:

- **Forward Pass**: `output = activation(weights @ input + bias)`
- **Loss Function**: Cross-entropy for multi-class classification
- **Backpropagation**: Chain rule to compute gradients
- **Weight Updates**: Gradient descent optimization
- **Regularization**: L2 penalty to prevent overfitting

**Key Learning Concepts:**
- Matrix operations in neural networks
- Gradient computation and backpropagation
- Effect of hyperparameters on training
- Visualization of high-dimensional learning
- Real-world neural network behavior

*Built with ❤️ using Streamlit, NumPy, and Plotly for interactive learning!*
"""
)
