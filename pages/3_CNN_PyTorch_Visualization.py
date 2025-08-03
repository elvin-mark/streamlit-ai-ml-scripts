import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🧠 Configurable CNN Visualizer on Digits Dataset")

# -----------------------
# Load and prepare data
# -----------------------
digits = load_digits()
X = digits.images
y = digits.target

# Normalize and reshape
X = X / 16.0
X = np.expand_dims(X, 1)  # (n, 1, 8, 8)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
)
test_ds = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)
)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

# -----------------------
# User configuration
# -----------------------
with st.sidebar:
    st.header("🛠️ Model Config")
    n_filters1 = st.slider("Filters in Conv1", 4, 16, 8)
    n_filters2 = st.slider("Filters in Conv2", 4, 32, 16)
    fc_units = st.slider("Hidden units in FC1", 16, 128, 64)
    learning_rate = st.slider("Learning Rate", 0.0001, 0.05, 0.01, format="%.4f")
    num_epochs = st.slider("Epochs", 1, 20, 10)


# -----------------------
# Define CNN
# -----------------------
class ConfigurableCNN(nn.Module):
    def __init__(self, f1, f2, fc_units):
        super().__init__()
        self.conv1 = nn.Conv2d(1, f1, 3, padding=1)
        self.conv2 = nn.Conv2d(f1, f2, 3, padding=1)
        self.fc1 = nn.Linear(f2 * 2 * 2, fc_units)
        self.fc2 = nn.Linear(fc_units, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # -> [B, f1, 8, 8]
        x = F.max_pool2d(x, 2)  # -> [B, f1, 4, 4]
        x = F.relu(self.conv2(x))  # -> [B, f2, 4, 4]
        x = F.max_pool2d(x, 2)  # -> [B, f2, 2, 2]
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)


@st.cache_resource
def train_model(f1, f2, fc_units, lr, epochs):
    model = ConfigurableCNN(f1, f2, fc_units)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": []}
    for _ in range(epochs):
        model.train()
        correct, total, loss_total = 0, 0, 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()

            loss_total += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

        history["train_loss"].append(loss_total / total)
        history["train_acc"].append(correct / total)

    return model, history


# -----------------------
# Train Model
# -----------------------
with st.spinner("Training model..."):
    model, history = train_model(
        n_filters1, n_filters2, fc_units, learning_rate, num_epochs
    )
st.success("✅ Training Complete!")

# -----------------------
# Show Training History
# -----------------------
with st.expander("📈 Training Curves", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(history["train_loss"])
        st.caption("Loss over epochs")
    with col2:
        st.line_chart(history["train_acc"])
        st.caption("Accuracy over epochs")

# -----------------------
# Show Filters
# -----------------------
with st.expander("🔍 Conv1 Filters", expanded=False):
    filters = model.conv1.weight.data.numpy()
    fig, axes = plt.subplots(1, min(8, filters.shape[0]), figsize=(12, 2))
    for i, ax in enumerate(axes):
        ax.imshow(filters[i, 0], cmap="gray")
        ax.set_title(f"F{i}")
        ax.axis("off")
    st.pyplot(fig)

# -----------------------
# Feature Maps & Prediction
# -----------------------
st.subheader("🧪 Feature Maps & Prediction")

sample_idx = st.slider("Test Image Index", 0, len(X_test) - 1, 0)
sample_img = torch.tensor(X_test[sample_idx : sample_idx + 1], dtype=torch.float32)

st.image(X_test[sample_idx][0], width=150, caption=f"Label: {y_test[sample_idx]}")


def extract_feature_maps(model, x):
    with torch.no_grad():
        fmap1 = F.relu(model.conv1(x))
        pool1 = F.max_pool2d(fmap1, 2)
        fmap2 = F.relu(model.conv2(pool1))
        pool2 = F.max_pool2d(fmap2, 2)
    return fmap1, fmap2


fmap1, fmap2 = extract_feature_maps(model, sample_img)

with st.expander("🧬 Conv1 Feature Maps", expanded=True):
    fig, axes = plt.subplots(1, min(n_filters1, 8), figsize=(12, 2))
    for i, ax in enumerate(axes):
        ax.imshow(fmap1[0, i].cpu(), cmap="inferno")
        ax.set_title(f"Map {i}")
        ax.axis("off")
    st.pyplot(fig)

with st.expander("🧬 Conv2 Feature Maps", expanded=True):
    fig, axes = plt.subplots(1, min(n_filters2, 8), figsize=(12, 2))
    for i, ax in enumerate(axes):
        ax.imshow(fmap2[0, i].cpu(), cmap="inferno")
        ax.set_title(f"Map {i}")
        ax.axis("off")
    st.pyplot(fig)

# -----------------------
# Show Prediction
# -----------------------
model.eval()
with torch.no_grad():
    output = model(sample_img)
    pred = torch.argmax(output).item()
    prob = torch.softmax(output, dim=1).squeeze().numpy()

st.metric("🔢 Predicted Label", f"{pred}")
st.bar_chart(prob)
