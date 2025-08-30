import streamlit as st
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Paths
tokenizer_path = "saved_models/tokenizer.pkl"
model_path = "saved_models/bigru.h5"

# Google Drive file ID (from your shared link)
gdrive_id = "1D3K647VnDsN7XjoeBBFHGaGDrAbbOu0h"
gdrive_url = f"https://drive.google.com/uc?id={gdrive_id}"

# Make sure the directory exists
os.makedirs("saved_models", exist_ok=True)

# Load tokenizer
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

# Download model if not present
if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    gdown.download(gdrive_url, model_path, quiet=False)

# Load model
model = load_model(model_path)
print("Model loaded successfully!")
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
MAX_LEN = 128  # used during training

# Text preprocessing
def preprocess(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    return padded

# Prediction function
def predict_toxicity(text):
    processed = preprocess(text)
    probs = model.predict(processed)[0]
    return dict(zip(LABELS, probs))

# Bulk prediction
def bulk_predict(df, text_column):
    df['processed'] = df[text_column].astype(str).apply(preprocess)
    X = np.vstack(df['processed'].values)
    probs = model.predict(X)
    for i, label in enumerate(LABELS):
        df[label] = probs[:, i]
    return df.drop(columns='processed')

# Comment Classification    
def classify_comment(probs, threshold=0.5):
    triggered_labels = [label for label, score in probs.items() if score >= threshold]
    if not triggered_labels:
        return "Non-toxic"
    else:
        return ", ".join(triggered_labels)
# UI
st.set_page_config(page_title="Toxicity Detector", layout="wide")
st.title("Real-Time Toxicity Detection App")

tab1, tab2, tab3 = st.tabs(["Single Comment", "Bulk CSV Upload", "Model Insights"])

with tab1:
    st.subheader("Enter a comment to analyze:")
    user_input = st.text_area("Comment", height=150)
    if st.button("Predict Toxicity"):
        if user_input.strip():
            result = predict_toxicity(user_input)
            verdict = classify_comment(result)
            st.write(f"### Verdict: {verdict}")
            st.write("### Prediction Scores")
            for label, score in result.items():
                st.write(f"**{label}**: {score:.2f}")
            st.bar_chart(result)
        else:
            st.warning("Please enter a comment.")

with tab2:
    st.subheader("Upload a CSV file for bulk toxicity prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    text_column = st.text_input("Enter the column name containing comments", value="comment")
    if uploaded_file and text_column:
        df = pd.read_csv(uploaded_file)
        if text_column in df.columns:
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            if st.button("Run Bulk Prediction"):
                result_df = bulk_predict(df, text_column)
                st.success("Prediction complete!")
                st.dataframe(result_df.head())
                st.download_button("Download Results", result_df.to_csv(index=False), "toxicity_predictions.csv")
        else:
            st.error(f"Column '{text_column}' not found in uploaded file.")

with tab3:
    st.subheader("Model Performance Metrics")
    st.markdown("""
    - **Best Model**: GRU with GloVe embeddings + Focal Loss  
    - **Macro F1**: 0.59  
    - **Micro F1**: 0.73  
    - **Strong Labels**: `toxic`, `obscene`, `insult`  
    - **Improved Recall**: `threat`, `identity_hate`  
    """)

    st.write("### Confusion Matrices (from uploaded CSV)")

    if uploaded_file and "comment" in df.columns and all(label in df.columns for label in LABELS):
        # Predict again using bulk_predict
        result_df = bulk_predict(df, "comment")
        y_true = df[LABELS].values
        y_pred = (result_df[LABELS].values >= 0.5).astype(int)

        conf_matrices = multilabel_confusion_matrix(y_true, y_pred)

        for i, label in enumerate(LABELS):
            st.write(f"**{label}**")
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrices[i], annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
    else:
        st.info("Upload a labeled CSV with true toxicity columns to view confusion matrices.")

    st.write("### Sample Test Cases")
    sample_comments = [
        "You are a horrible person!",
        "I love this community.",
        "I will find you and hurt you.",
        "You're such an idiot.",
        "This is a great post!"
    ]
    for comment in sample_comments:
        st.markdown(f"**Comment**: {comment}")
        scores = predict_toxicity(comment)
        verdict = classify_comment(scores)
        st.write(f"**Predicted Labels**: {verdict}")
        st.bar_chart(scores)


