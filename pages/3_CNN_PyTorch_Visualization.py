import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="CNN Visualizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.stAlert > div {
    background-color: #d4edda;
    border-color: #c3e6cb;
    color: #155724;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🧠 Advanced CNN Visualizer on Digits Dataset")
st.markdown(
    "Explore and understand Convolutional Neural Networks through interactive visualization"
)


# -----------------------
# Data Loading & Caching
# -----------------------
@st.cache_data
def load_and_prepare_data():
    """Load and preprocess the digits dataset"""
    digits = load_digits()
    X = digits.images
    y = digits.target

    # Normalize
    X = X / 16.0
    X = np.expand_dims(X, 1)  # (n, 1, 8, 8)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Convert to tensors
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    return X_train, X_test, y_train, y_test, train_ds, test_ds


# Load data
X_train, X_test, y_train, y_test, train_ds, test_ds = load_and_prepare_data()

# -----------------------
# Sidebar Configuration
# -----------------------
with st.sidebar:
    st.header("🛠️ Model Configuration")

    # Architecture settings
    st.subheader("Architecture")
    n_filters1 = st.slider(
        "Conv1 Filters", 4, 32, 8, help="Number of filters in first convolutional layer"
    )
    n_filters2 = st.slider(
        "Conv2 Filters",
        4,
        64,
        16,
        help="Number of filters in second convolutional layer",
    )
    fc_units = st.slider(
        "FC Hidden Units", 16, 256, 64, help="Number of units in fully connected layer"
    )
    dropout_rate = st.slider(
        "Dropout Rate", 0.0, 0.8, 0.2, step=0.1, help="Dropout rate for regularization"
    )

    # Training settings
    st.subheader("Training")
    learning_rate = st.slider("Learning Rate", 0.0001, 0.1, 0.01, format="%.4f")
    batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=2)
    num_epochs = st.slider("Epochs", 1, 50, 15)
    optimizer_choice = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"])

    # Visualization settings
    st.subheader("Visualization")
    show_architecture = st.checkbox("Show Model Architecture", True)
    show_gradients = st.checkbox("Show Gradient Flow", False)
    colormap = st.selectbox(
        "Colormap", ["viridis", "plasma", "inferno", "magma", "hot"]
    )


# -----------------------
# Enhanced CNN Model
# -----------------------
class EnhancedCNN(nn.Module):
    def __init__(self, f1, f2, fc_units, dropout_rate=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, f1, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(f1)
        self.conv2 = nn.Conv2d(f1, f2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(f2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(f2 * 2 * 2, fc_units)
        self.fc2 = nn.Linear(fc_units, 10)

    def forward(self, x):
        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))  # -> [B, f1, 8, 8]
        x = F.max_pool2d(x, 2)  # -> [B, f1, 4, 4]

        # Conv block 2
        x = F.relu(self.bn2(self.conv2(x)))  # -> [B, f2, 4, 4]
        x = F.max_pool2d(x, 2)  # -> [B, f2, 2, 2]

        # Fully connected layers
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    def get_feature_maps(self, x):
        """Extract intermediate feature maps for visualization"""
        with torch.no_grad():
            fmap1 = F.relu(self.bn1(self.conv1(x)))
            pool1 = F.max_pool2d(fmap1, 2)
            fmap2 = F.relu(self.bn2(self.conv2(pool1)))
            pool2 = F.max_pool2d(fmap2, 2)
        return fmap1, fmap2, pool1, pool2


# -----------------------
# Training with Progress Tracking
# -----------------------
@st.cache_resource
def train_model(f1, f2, fc_units, dropout_rate, lr, epochs, batch_size, optimizer_name):
    """Train the model with comprehensive metrics tracking"""
    model = EnhancedCNN(f1, f2, fc_units, dropout_rate)

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Choose optimizer
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:  # RMSprop
        optimizer = optim.RMSprop(model.parameters(), lr=lr)

    loss_fn = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "learning_rates": [],
    }

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_correct, train_total, train_loss_total = 0, 0, 0.0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total += len(y_batch)

        # Validation phase
        model.eval()
        test_correct, test_total, test_loss_total = 0, 0, 0.0

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                logits = model(x_batch)
                loss = loss_fn(logits, y_batch)

                test_loss_total += loss.item() * len(y_batch)
                preds = logits.argmax(dim=1)
                test_correct += (preds == y_batch).sum().item()
                test_total += len(y_batch)

        # Record metrics
        history["train_loss"].append(train_loss_total / train_total)
        history["train_acc"].append(train_correct / train_total)
        history["test_loss"].append(test_loss_total / test_total)
        history["test_acc"].append(test_correct / test_total)
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])

        scheduler.step()

    return model, history


# -----------------------
# Model Training
# -----------------------
st.subheader("🚀 Training Progress")

# Training button and progress
if st.button("🎯 Train Model", type="primary"):
    with st.spinner("Training model... Please wait"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        model, history = train_model(
            n_filters1,
            n_filters2,
            fc_units,
            dropout_rate,
            learning_rate,
            num_epochs,
            batch_size,
            optimizer_choice,
        )

        progress_bar.progress(100)
        status_text.success("✅ Training Complete!")

        # Store in session state
        st.session_state.model = model
        st.session_state.history = history

# Check if model exists in session state
if "model" not in st.session_state:
    st.info("👆 Click 'Train Model' to start training with your configuration")
    st.stop()

model = st.session_state.model
history = st.session_state.history

# -----------------------
# Model Architecture Visualization
# -----------------------
if show_architecture:
    with st.expander("🏗️ Model Architecture", expanded=True):
        col1, col2 = st.columns([2, 1])

        with col1:
            # Calculate model parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            st.markdown(
                f"""
            **Model Summary:**
            - Conv1: 1 → {n_filters1} filters (3×3)
            - Conv2: {n_filters1} → {n_filters2} filters (3×3)
            - FC1: {n_filters2 * 2 * 2} → {fc_units} units
            - FC2: {fc_units} → 10 units (output)
            - Total Parameters: {total_params:,}
            - Trainable Parameters: {trainable_params:,}
            - Dropout Rate: {dropout_rate}
            """
            )

        with col2:
            st.metric("Total Parameters", f"{total_params:,}")
            st.metric("Model Size (MB)", f"{total_params * 4 / 1024 / 1024:.2f}")

# -----------------------
# Enhanced Training Visualization
# -----------------------
with st.expander("📈 Training Analytics", expanded=True):
    # Create interactive plots with Plotly
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Training Loss",
            "Training Accuracy",
            "Learning Rate",
            "Train vs Test Comparison",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    epochs_range = list(range(1, len(history["train_loss"]) + 1))

    # Loss plot
    fig.add_trace(
        go.Scatter(
            x=epochs_range,
            y=history["train_loss"],
            name="Train Loss",
            line=dict(color="red"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs_range,
            y=history["test_loss"],
            name="Test Loss",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    # Accuracy plot
    fig.add_trace(
        go.Scatter(
            x=epochs_range,
            y=history["train_acc"],
            name="Train Acc",
            line=dict(color="green"),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs_range,
            y=history["test_acc"],
            name="Test Acc",
            line=dict(color="orange"),
        ),
        row=1,
        col=2,
    )

    # Learning rate plot
    fig.add_trace(
        go.Scatter(
            x=epochs_range,
            y=history["learning_rates"],
            name="Learning Rate",
            line=dict(color="purple"),
        ),
        row=2,
        col=1,
    )

    # Overfitting analysis
    gap = [
        abs(train - test)
        for train, test in zip(history["train_acc"], history["test_acc"])
    ]
    fig.add_trace(
        go.Scatter(
            x=epochs_range,
            y=gap,
            name="Accuracy Gap",
            line=dict(color="red", dash="dash"),
        ),
        row=2,
        col=2,
    )

    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Final Train Accuracy", f"{history['train_acc'][-1]:.3f}")
    with col2:
        st.metric("Final Test Accuracy", f"{history['test_acc'][-1]:.3f}")
    with col3:
        st.metric("Best Test Accuracy", f"{max(history['test_acc']):.3f}")
    with col4:
        overfitting = history["train_acc"][-1] - history["test_acc"][-1]
        st.metric("Overfitting Gap", f"{overfitting:.3f}")

# -----------------------
# Filter Visualization
# -----------------------
with st.expander("🔬 Learned Filters", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Conv1 Filters")
        filters1 = model.conv1.weight.data.cpu().numpy()
        fig, axes = plt.subplots(2, min(4, filters1.shape[0] // 2), figsize=(8, 4))
        axes = (
            axes.flatten()
            if filters1.shape[0] > 4
            else [axes] if filters1.shape[0] == 1 else axes
        )

        for i in range(min(8, filters1.shape[0])):
            ax = axes[i] if len(axes) > 1 else axes
            im = ax.imshow(filters1[i, 0], cmap=colormap)
            ax.set_title(f"Filter {i}")
            ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Conv2 Filters")
        filters2 = model.conv2.weight.data.cpu().numpy()
        # Show first channel of each filter
        fig, axes = plt.subplots(2, min(4, filters2.shape[0] // 2), figsize=(8, 4))
        axes = (
            axes.flatten()
            if filters2.shape[0] > 4
            else [axes] if filters2.shape[0] == 1 else axes
        )

        for i in range(min(8, filters2.shape[0])):
            ax = axes[i] if len(axes) > 1 else axes
            im = ax.imshow(filters2[i, 0], cmap=colormap)
            ax.set_title(f"Filter {i}")
            ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)

# -----------------------
# Interactive Sample Analysis
# -----------------------
st.subheader("🔍 Interactive Sample Analysis")

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    sample_idx = st.slider("Test Sample Index", 0, len(X_test) - 1, 0)
    sample_img = torch.tensor(X_test[sample_idx : sample_idx + 1], dtype=torch.float32)
    true_label = y_test[sample_idx]

    # Display original image
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(X_test[sample_idx][0], cmap="gray")
    ax.set_title(f"True Label: {true_label}")
    ax.axis("off")
    st.pyplot(fig)

with col2:
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(sample_img)
        probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()
        predicted_label = torch.argmax(output).item()

    # Prediction visualization
    fig = px.bar(
        x=list(range(10)),
        y=probabilities,
        title="Class Probabilities",
        labels={"x": "Digit Class", "y": "Probability"},
    )
    fig.update_traces(
        marker_color=["red" if i == predicted_label else "lightblue" for i in range(10)]
    )
    st.plotly_chart(fig, use_container_width=True)

with col3:
    # Prediction metrics
    confidence = probabilities[predicted_label]
    is_correct = predicted_label == true_label

    st.metric("Predicted Label", predicted_label)
    st.metric("Confidence", f"{confidence:.3f}")
    if is_correct:
        st.success("✅ Correct")
    else:
        st.error("❌ Incorrect")

# -----------------------
# Feature Maps Visualization
# -----------------------
fmap1, fmap2, pool1, pool2 = model.get_feature_maps(sample_img)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Conv1 Output", "Conv1 Pooled", "Conv2 Output", "Conv2 Pooled"]
)

with tab1:
    n_maps = min(n_filters1, 12)
    cols = 4
    rows = (n_maps + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten() if n_maps > 1 else [axes]

    for i in range(n_maps):
        axes[i].imshow(fmap1[0, i].cpu(), cmap=colormap)
        axes[i].set_title(f"Feature Map {i}")
        axes[i].axis("off")

    # Hide unused subplots
    for i in range(n_maps, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    n_maps = min(n_filters1, 12)
    cols = 4
    rows = (n_maps + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten() if n_maps > 1 else [axes]

    for i in range(n_maps):
        axes[i].imshow(pool1[0, i].cpu(), cmap=colormap)
        axes[i].set_title(f"Pooled Map {i}")
        axes[i].axis("off")

    for i in range(n_maps, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    n_maps = min(n_filters2, 12)
    cols = 4
    rows = (n_maps + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten() if n_maps > 1 else [axes]

    for i in range(n_maps):
        axes[i].imshow(fmap2[0, i].cpu(), cmap=colormap)
        axes[i].set_title(f"Feature Map {i}")
        axes[i].axis("off")

    for i in range(n_maps, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    st.pyplot(fig)

with tab4:
    n_maps = min(n_filters2, 12)
    cols = 4
    rows = (n_maps + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten() if n_maps > 1 else [axes]

    for i in range(n_maps):
        axes[i].imshow(pool2[0, i].cpu(), cmap=colormap)
        axes[i].set_title(f"Final Feature {i}")
        axes[i].axis("off")

    for i in range(n_maps, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    st.pyplot(fig)

# -----------------------
# Model Evaluation
# -----------------------
with st.expander("📊 Detailed Model Evaluation", expanded=False):
    # Generate predictions for test set
    model.eval()
    all_preds = []
    all_true = []

    test_loader = DataLoader(test_ds, batch_size=64)
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            outputs = model(x_batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y_batch.cpu().numpy())

    # Confusion Matrix
    cm = confusion_matrix(all_true, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Classification Report
    report = classification_report(all_true, all_preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.subheader("Classification Report")
    st.dataframe(report_df.round(3))

# -----------------------
# Tips and Information
# -----------------------
with st.expander("💡 Tips & Information", expanded=False):
    st.markdown(
        """
    ### Model Improvements Added:
    - **Batch Normalization**: Stabilizes training and improves convergence
    - **Dropout**: Reduces overfitting by randomly setting neurons to zero
    - **Learning Rate Scheduler**: Automatically reduces learning rate during training
    - **Multiple Optimizers**: Choose between Adam, SGD, and RMSprop
    - **Enhanced Metrics**: Track both training and validation metrics
    
    ### Interpretation Guide:
    - **Feature Maps**: Show what patterns each filter detects
    - **Training Curves**: Monitor for overfitting (gap between train/test)
    - **Confusion Matrix**: Identify which classes are commonly confused
    - **Filter Visualization**: Understand what the network has learned
    
    ### Experiment Ideas:
    1. Try different architectures (more/fewer filters)
    2. Adjust dropout rate to see effect on overfitting
    3. Compare different optimizers
    4. Observe how learning rate affects convergence
    """
    )

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, PyTorch, and Plotly")
