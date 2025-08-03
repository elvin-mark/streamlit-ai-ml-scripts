import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="🧠 MLP Playground", layout="wide")

st.title("🔢 Simple MLP Implementation with NumPy")


# ----------------------
# Data Loading & Setup
# ----------------------
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

    return X_train, X_test, y_train, y_test, scaler, encoder, iris


X_train, X_test, y_train, y_test, scaler, encoder, iris = load_data()

# ----------------------
# MLP Implementation
# ----------------------
st.subheader("🔧 How the MLP Works")
st.markdown(
    """
We build a simple **Multi-Layer Perceptron (MLP)** with:
- An input layer (4 features)
- One hidden layer (user-defined neurons)
- An output layer (3 neurons for 3 Iris classes)

### 🔄 Training Process (Forward & Backpropagation):
1. **Forward Pass**:
   - Compute activations of the hidden layer using ReLU
   - Compute output using softmax for classification

2. **Loss**:
   - Cross-entropy loss for multi-class classification

3. **Backward Pass**:
   - Compute gradients using chain rule
   - Update weights with stochastic gradient descent (SGD)
"""
)

# User hyperparameters
hidden_size = st.slider("Hidden Layer Size", min_value=4, max_value=64, value=8)
lr = st.slider("Learning Rate", 0.001, 1.0, 0.01)
epochs = st.slider("Training Epochs", 10, 500, 100)

# Session state for storing model
if "mlp_weights" not in st.session_state:
    st.session_state.mlp_weights = {}


# MLP Functions
def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return (x > 0).astype(float)


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # for numerical stability
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def cross_entropy(y_pred, y_true):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))


def train(X, y, hidden_size, lr, epochs):
    input_size = X.shape[1]
    output_size = y.shape[1]

    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))

    loss_list = []

    for epoch in range(epochs):
        # --- Forward Pass ---
        z1 = X.dot(W1) + b1
        a1 = relu(z1)
        z2 = a1.dot(W2) + b2
        a2 = softmax(z2)

        # --- Loss ---
        loss = cross_entropy(a2, y)
        loss_list.append(loss)

        # --- Backward Pass ---
        dz2 = a2 - y
        dW2 = a1.T.dot(dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2.dot(W2.T)
        dz1 = da1 * relu_deriv(z1)
        dW1 = X.T.dot(dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # --- Update ---
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    return (W1, b1, W2, b2), loss_list


# Train button
if st.button("🚀 Train Model"):
    with st.spinner("Training..."):
        weights, losses = train(X_train, y_train, hidden_size, lr, epochs)
        st.session_state.mlp_weights = {
            "W1": weights[0],
            "b1": weights[1],
            "W2": weights[2],
            "b2": weights[3],
        }
        st.line_chart(losses, height=200, use_container_width=True)
        st.success("Training completed!")

# ----------------------
# Model Prediction
# ----------------------
st.subheader("🔍 Try the Trained Model")

if st.session_state.mlp_weights:
    W1 = st.session_state.mlp_weights["W1"]
    b1 = st.session_state.mlp_weights["b1"]
    W2 = st.session_state.mlp_weights["W2"]
    b2 = st.session_state.mlp_weights["b2"]

    # Persist input values
    if "test_input" not in st.session_state:
        st.session_state.test_input = [5.1, 3.5, 1.4, 0.2]

    test_input = []
    cols = st.columns(4)
    for i, feature in enumerate(iris.feature_names):
        val = cols[i].slider(
            f"{feature}", -3.0, 3.0, float(st.session_state.test_input[i]), 0.1
        )
        test_input.append(val)
        st.session_state.test_input[i] = val

    # Predict
    x = np.array(test_input).reshape(1, -1)
    z1 = x.dot(W1) + b1
    a1 = relu(z1)
    z2 = a1.dot(W2) + b2
    a2 = softmax(z2)

    predicted_class = iris.target_names[np.argmax(a2)]

    st.markdown("#### 📊 Prediction Result")
    st.write(f"Predicted class: **{predicted_class}**")
    st.bar_chart(a2[0])
else:
    st.info("Train the model first to enable predictions.")

# Footer
st.markdown("---")
st.markdown(
    "🔬 *This app demonstrates the core concepts of forward and backward propagation using NumPy.*"
)
