import streamlit as st
import numpy as np
import cv2
import pandas as pd
import json
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, preprocess_input
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D

# Inject CSS for advanced styling
st.markdown(
    """
    <style>
    /* Entire App Background and Text Color */
    .stApp {
        background: linear-gradient(135deg, #f7fafc, #e0e7e9);
        color: #1e2a38;  /* Dark slate blue */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        min-height: 100vh;
    }

    /* Sidebar Background, Text, and Widget Styling */
    [data-testid="stSidebar"] {
        background-color: #bfd8d2 !important;  /* Muted soft teal */
        color: #1e2a38 !important;  /* Dark cool text */
        padding-top: 1.5rem;
        border-radius: 0 15px 15px 0;
    }

    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label {
        color: #1e2a38 !important;
        font-weight: 600;
    }

    [data-testid="stSidebar"] .streamlit-expanderHeader {
        color: #417d7a !important;
    }
    [data-testid="stSidebar"] .stFileUploader > div > label {
        color: #355c59 !important;
    }

    /* Title Styling */
    h1 {
        text-align: center;
        color: #34675c !important;  /* Calm green-teal */
        font-weight: 700;
        font-size: 3rem !important;
        text-shadow: 1px 1px 3px #c7d7d5;
        margin-bottom: 5px;
    }
    p {
        text-align: center;
        font-size: 1.1rem;
        margin-top: 0;
        margin-bottom: 1rem;
        color: #F0F8FF;
        font-weight: 600;
    }

    /* Expanders Styling */
    .stExpander {
        background-color: rgba(191, 216, 210, 0.75);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 6px 20px rgba(76, 127, 119, 0.3);
        color: #1e2a38;
        margin-bottom: 25px;
        transition: box-shadow 0.3s ease-in-out;
    }
    .streamlit-expanderHeader {
        font-size: 1.2rem !important;
        color: #2f5d58 !important;
        font-weight: 700 !important;
    }
    .stExpander:hover {
        box-shadow: 0 10px 30px rgba(64, 110, 101, 0.5);
    }

    /* Download Buttons */
    button[kind="secondary"] {
        background-color: #34675c !important;  /* Dark calm teal */
        color: white !important; /* Very light text */
        font-weight: 700 !important;
        border-radius: 8px !important;
        padding: 10px 22px !important;
        margin-top: 12px;
        margin-bottom: 18px;
        box-shadow: 0 3px 12px rgba(52, 103, 92, 0.45);
        transition: background-color 0.3s ease;
    }
    button[kind="secondary"]:hover {
        background-color: #2a5850 !important;
        color: #d4ebe9 !important;
        box-shadow: 0 5px 15px rgba(42, 88, 80, 0.6);
    }

    /* Prediction Box */
    .prediction-box {
        background-color: rgba(191, 216, 210, 0.85);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 12px 30px rgba(76, 127, 119, 0.5);
        border: 3px solid #34675c;
        color: #1e2a38 !important;
        margin-bottom: 40px;
        text-align: center;
    }

    /* Captions under images */
    .image-caption {
        font-weight: 700;
        font-size: 0.95rem;
        text-align: center;
        color: #2f5d58;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    /* Divider lines */
    hr {
        border: 1.5px solid #75a49a;
        margin: 30px 0;
        border-radius: 4px;
    }

    /* Dataframe styles (tables) */
    .css-1d391kg, .css-10trblm, .css-1lcbmhc, .css-1v3fvcr {
        color: #1e2a38 !important;
        font-weight: 600;
    }

    .css-1v3fvcr th {
        background-color: #c2dbd8 !important;
        color: #1e2a38 !important;
        font-weight: 700;
    }

    .css-1v3fvcr td {
        background-color: rgba(178, 214, 209, 0.6) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Page config and header (title/subtitle already styled by CSS)
st.set_page_config(page_title="Skin Disease Classifier", layout="wide")
st.markdown("<h1> Skin Disease Classification App</h1>", unsafe_allow_html=True)
# st.markdown("<p>Upload an image to classify it into one of the trained categories.</p>", unsafe_allow_html=True)
st.write("<hr>", unsafe_allow_html=True)

# Sidebar for file upload
st.sidebar.header("Upload Input Image")
uploaded_file = st.sidebar.file_uploader("Choose a skin image (JPG/PNG)", type=["jpg", "jpeg", "png"])

# Load artifacts once with caching
@st.cache_resource
def load_artifacts():
    cnn_model = load_model("artifacts/rmcnn_lstm.h5")
    lda = joblib.load("artifacts/lda.joblib")
    pca = joblib.load("artifacts/pca.joblib")
    scaler = joblib.load("artifacts/scaler.joblib")
    with open("artifacts/class_mapping.json", "r") as f:
        class_mapping = json.load(f)
    base_model = EfficientNetV2S(include_top=False, weights="imagenet", input_shape=(300, 300, 3))
    base_model.trainable = False
    gap_model = tf.keras.Sequential([base_model, GlobalAveragePooling2D()])
    return cnn_model, lda, pca, scaler, class_mapping, gap_model

cnn_model, lda, pca, scaler, class_mapping, gap_model = load_artifacts()

def lda_projection(feature_vector):
    return np.array([np.dot(row[1:], feature_vector) + row[0] for row in lda])

def preprocess_image(img, target_size=(300, 300)):
    img_resized = cv2.resize(img, target_size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_equalized = cv2.equalizeHist(gray)
    img_equalized_rgb = cv2.cvtColor(img_equalized, cv2.COLOR_GRAY2RGB)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    img_clahe = clahe.apply(img_equalized)
    blurred = cv2.GaussianBlur(img_clahe, (0, 0), 10)
    img_sharpened = cv2.addWeighted(img_clahe, 1.5, blurred, -0.5, 0)
    img_sharpened_rgb = cv2.cvtColor(img_sharpened, cv2.COLOR_GRAY2RGB)
    img_efnet = preprocess_input(img_sharpened_rgb.astype(np.float32))
    return img_rgb, img_equalized_rgb, img_sharpened_rgb, img_efnet

def extract_features(img_array):
    features = gap_model.predict(np.expand_dims(img_array, axis=0), verbose=0)
    return features[0]

def run_pipeline(img):
    img_rgb, img_equalized_rgb, img_sharpened_rgb, img_efnet = preprocess_image(img)
    features = extract_features(img_efnet)
    lda_features = lda_projection(features)
    combined = np.hstack([features, lda_features]).reshape(1, -1)
    pca_features = pca.transform(combined)
    scaled_features = scaler.transform(pca_features)
    final_input = scaled_features.reshape(-1, 100, 1)
    return {
        "images": [img_rgb, img_equalized_rgb, img_sharpened_rgb],
        "features": features,
        "lda_features": lda_features,
        "pca_features": pca_features[0],
        "final_input": final_input
    }

def predict(final_input):
    preds = cnn_model.predict(final_input, verbose=0)[0]
    predicted_class = class_mapping[str(np.argmax(preds))]
    return predicted_class, preds

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    pipeline_out = run_pipeline(img)
    images = pipeline_out["images"]
    features = pipeline_out["features"]
    lda_features = pipeline_out["lda_features"]
    pca_features = pipeline_out["pca_features"]
    final_input = pipeline_out["final_input"]

    # 1. Preprocessing steps as columns with captions
    with st.expander("1. Image Preprocessing", expanded=True):
        captions = ["Resized (RGB)", "Histogram Equalized", "CLAHE & Sharpened"]
        cols = st.columns(len(images))
        for i, im in enumerate(images):
            with cols[i]:
                st.image(im, width='content')
                st.markdown(f"<p class='image-caption'>{captions[i]}</p>", unsafe_allow_html=True)

    # 2. EfficientNet Features
    with st.expander("2. EfficientNet Extracted Features (Preview & Download)", expanded=True):
        features_df = pd.DataFrame(features.reshape(1, -1))
        st.dataframe(features_df, use_container_width=True)
        st.download_button("Download EfficientNet Features CSV", features_df.to_csv(index=False), "efficientnet_features.csv", width='content')
    st.write("<hr>", unsafe_allow_html=True)

    # 3. LDA Features
    with st.expander("3. Linear Discriminate Analysis (Preview & Download)"):
        lda_df = pd.DataFrame(lda_features.reshape(1, -1))
        st.dataframe(lda_df, use_container_width=True)
        st.download_button("Download LDA Features CSV", lda_df.to_csv(index=False), "lda_features.csv", width='content')
    st.write("<hr>", unsafe_allow_html=True)

    # 4. PCA Features
    with st.expander("4. Principle Component Analysis Features (Preview & Download)"):
        pca_df = pd.DataFrame(pca_features.reshape(1, -1))
        st.dataframe(pca_df, use_container_width=True)
        st.download_button("Download PCA Features CSV", pca_df.to_csv(index=False), "pca_features.csv", width='content')
    st.write("<hr>", unsafe_allow_html=True)

    # 5. Prediction result with styled box
    predicted_class, probs = predict(final_input)
    st.subheader("5. Predicted Result")
    st.markdown(
        f"<div class='prediction-box'>"
        f"<h4 style='color:white;'>Predicted Class: <b>{predicted_class}</b></h4></div>",
        unsafe_allow_html=True,
    )
    st.write("#### Class Probabilities")
    prob_dict = {class_mapping[str(i)]: float(prob) for i, prob in enumerate(probs)}
    st.json(prob_dict)
else:
    st.info("Please upload a skin image from the sidebar to get started!")
