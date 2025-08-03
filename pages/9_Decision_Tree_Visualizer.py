import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Decision Tree Visualizer", layout="wide")

st.title("🌳 Decision Tree Visualizer")
st.markdown(
    "Train a decision tree classifier and interactively explore how hyperparameters affect its behavior and decision boundaries."
)

# -----------------------------
# Dataset Selector
# -----------------------------
dataset_name = st.sidebar.selectbox("📊 Choose a dataset", ["Iris", "Titanic"])


@st.cache_data
def load_data(name):
    if name == "Iris":
        data = load_iris(as_frame=True)
        df = data.frame
        X = df.drop(columns="target")
        y = df["target"]
        feature_names = X.columns.tolist()
        class_names = data.target_names.tolist()
    elif name == "Titanic":
        df = pd.read_csv(
            "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        )
        df = df[["Pclass", "Sex", "Age", "Fare", "Survived"]].dropna()
        le = LabelEncoder()
        df["Sex"] = le.fit_transform(df["Sex"])
        X = df.drop(columns="Survived")
        y = df["Survived"]
        feature_names = X.columns.tolist()
        class_names = ["Died", "Survived"]
    return X, y, feature_names, class_names


X, y, feature_names, class_names = load_data(dataset_name)

# -----------------------------
# Model Controls
# -----------------------------
with st.sidebar:
    st.header("⚙️ Model Hyperparameters")
    max_depth = st.slider("Max Depth", 1, 10, 3)
    min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1)
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2, step=0.05)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

# Model Training
clf = DecisionTreeClassifier(
    max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# -----------------------------
# Tree Plot
# -----------------------------
with st.expander("🌲 Tree Visualization", expanded=True):
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        ax=ax,
    )
    st.pyplot(fig)

# -----------------------------
# Decision Boundary (2D)
# -----------------------------
if len(feature_names) >= 2:
    st.subheader("📊 2D Decision Boundary (Feature Pair)")

    f1, f2 = st.selectbox("Select Feature X", feature_names, index=0), st.selectbox(
        "Select Feature Y", feature_names, index=1
    )

    if f1 != f2:
        X_pair = X[[f1, f2]]
        X_train_pair, X_test_pair, y_train_pair, y_test_pair = train_test_split(
            X_pair, y, test_size=test_size, random_state=42, stratify=y
        )
        clf_2d = DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42
        )
        clf_2d.fit(X_train_pair, y_train_pair)

        x_min, x_max = X_pair[f1].min() - 1, X_pair[f1].max() + 1
        y_min, y_max = X_pair[f2].min() - 1, X_pair[f2].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300)
        )
        Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
        sns.scatterplot(data=X_pair, x=f1, y=f2, hue=y, palette="Set1", ax=ax2)
        ax2.set_title("Decision Boundary")
        st.pyplot(fig2)

# -----------------------------
# Performance Metrics
# -----------------------------
with st.expander("📈 Performance Metrics", expanded=False):
    st.metric("Accuracy", f"{acc:.3f}")
    fig3, ax3 = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax3,
    )
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    ax3.set_title("Confusion Matrix")
    st.pyplot(fig3)

# -----------------------------
# Data Preview
# -----------------------------
with st.expander("📄 Dataset Preview", expanded=False):
    st.dataframe(X.join(y.rename("Target")))
