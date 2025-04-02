import os
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import gdown
import shap
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adamax
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image

# Set Streamlit Page Configuration
st.set_page_config(page_title="ANN Dashboard - Customer Spending Analysis", layout="wide")

# --- 1. Data Loading and Preprocessing ---

# 📥 Load Dataset
DATASET_FILE_ID = "1uy-mxgp4qqeUxOlzJWJLQsi4Le8qD61a"  # Your Google Drive file ID
MODEL_PATH = "spending_model.h5"

@st.cache_data
def load_data():
    if not os.path.exists("credit_card_data.csv"):
        gdown.download(f"https://drive.google.com/uc?id={DATASET_FILE_ID}", "credit_card_data.csv", quiet=False)

    df = pd.read_csv("credit_card_data.csv")

    # Data preprocessing
    df['Converted'] = (df['PURCHASES'] > df['PURCHASES'].median()).astype(int)
    df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].median(), inplace=True)
    df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median(), inplace=True)
    if 'CUST_ID' in df.columns:
        df.drop(columns=['CUST_ID'], inplace=True)

    return df

# 🎨 Custom CSS
st.markdown(
    """
    <style>
        .title {
            color: #4B8BBE;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sidebar-header {
            color: #306998;
            font-size: 1.5em;
            margin-bottom: 1rem;
        }
        .metric-label {
            font-size: 1.2em;
            color: #2C3E50;
        }
        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #4B8BBE;
        }
        .stButton > button {
            background-color: #306998;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Title
st.title("💳 ANN Customer Spending Analysis Dashboard")

# Sidebar Configuration
st.sidebar.header("🔧 Model Configuration")
epochs = st.sidebar.slider("Epochs", 5, 50, 10)
batch_size = st.sidebar.selectbox("Batch Size", [32, 64, 128, 256], index=2)
hidden_units = st.sidebar.slider("Hidden Units", 32, 256, 128, step=32)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.1, 0.5, 0.3, step=0.1)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.01, 0.001, 0.0001], index=1)
optimizer_choice = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"], index=0)

# Select Optimizer
optimizers = {
    "adam": Adam(learning_rate=learning_rate),
    "sgd": SGD(learning_rate=learning_rate),
    "rmsprop": RMSprop(learning_rate=learning_rate)
}
optimizer = optimizers[optimizer_choice]

# --- 2. Model Building ---
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_units, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(hidden_units//2, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- 3. Data Exploration ---
df = load_data()

st.header("📊 Data Exploration")
if st.checkbox("Show Raw Data"):
    st.dataframe(df.head())

st.subheader("Data Statistics")
st.dataframe(df.describe())

# Visualizations
st.subheader("Data Distributions")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.histplot(df['BALANCE'], bins=30, kde=True, ax=axes[0, 0])
sns.histplot(df['PURCHASES'], bins=30, kde=True, ax=axes[0, 1])
sns.histplot(df['CREDIT_LIMIT'], bins=30, kde=True, ax=axes[1, 0])
sns.histplot(df['PAYMENTS'], bins=30, kde=True, ax=axes[1, 1])
st.pyplot(fig)

# --- 4. Model Training ---
if st.button("🚀 Train Model"):
    with st.spinner("Training model..."):
        # Prepare data
        X = df.drop(columns=['Converted'])
        y = df['Converted']

        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

        # Build and train model
        model = build_model(X_train_scaled.shape[1])
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        history = model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_scaled, y_val),
            callbacks=[early_stopping],
            class_weight=class_weight_dict,
            verbose=0
        )

    st.success("🎉 Model training complete!")

    # Evaluation
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)

    # Display metrics
    st.subheader("📊 Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Test Accuracy", f"{accuracy:.4f}")
    with col2:
        st.metric("Test Loss", f"{loss:.4f}")

    # Training history plots
    st.subheader("📈 Training History")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()

    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    st.pyplot(fig)

    # Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low Spender', 'High Spender'],
                yticklabels=['Low Spender', 'High Spender'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    # Classification Report
    st.subheader("📜 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # Feature Importance
    st.subheader("🔍 Feature Importance")
    explainer = shap.Explainer(model, X_train_scaled[:100])
    shap_values = explainer(X_test_scaled[:100])
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test_scaled[:100], show=False)
    st.pyplot(fig)

# GitHub Follow Button
st.markdown(
    """
    <div style="text-align: center; margin-top: 2rem;">
        <a href="https://github.com/yourusername" target="_blank">
            <button style="background-color: #306998; color: white; padding: 10px 20px; border-radius: 5px;">
                ⭐ Follow Me on GitHub
            </button>
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)
