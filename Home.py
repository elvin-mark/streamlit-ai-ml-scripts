import streamlit as st

st.set_page_config(page_title="Home", layout="wide")

st.title("Streamlit AI/ML Scripts")

st.markdown("""
This repository contains a collection of Streamlit applications that implement and visualize various machine learning and deep learning concepts.
""")

st.header("Scripts")

st.markdown("""
- **[MLP From Scratch](1_MLP_From_Scratch)**: Implements a simple Multi-Layer Perceptron (MLP) from scratch using NumPy to classify the Iris dataset.
- **[MLP Loss Landscape](2_MLP_Loss_Landscape)**: Visualizes the loss landscape of an MLP trained on the Iris dataset.
- **[CNN PyTorch Visualization](3_CNN_PyTorch_Visualization)**: A configurable Convolutional Neural Network (CNN) visualizer for the Digits dataset, built with PyTorch.
- **[PCA Digits Explorer](4_PCA_Digits_Explorer)**: An interactive explorer for Principal Component Analysis (PCA) on the handwritten digits dataset.
- **[KMeans Clustering Visualizer](5_KMeans_Clustering_Visualizer)**: A visualizer for the K-Means clustering algorithm on different datasets.
- **[Tokenizer Visualizer](6_Tokenizer_Visualizer)**: A tool to visualize the tokenization of text using a HuggingFace tokenizer.
- **[NGram Prediction](7_NGram_Prediction)**: An n-gram generator and token predictor.
- **[Attention Visualizer](8_Attention_Visualizer)**: A visualizer for the self-attention maps of the BERT model.
""")