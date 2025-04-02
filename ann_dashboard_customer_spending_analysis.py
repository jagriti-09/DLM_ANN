import streamlit as st
import pandas as pd  # Fixed typo from 'panda'
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Fixed typo from 'scabcom'
from sklearn.preprocessing import StandardScaler  # Fixed capitalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report  # Fixed incomplete import
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # Fixed typo
from sklearn.utils.class_weight import compute_class_weight  # Fixed typo
import plotly.express as px  # Fixed underscore
import plotly.graph_objects as go  # Fixed dot notation
import requests
from io import StringIO  # Fixed capitalization

# Correct page config
st.set_page_config(page_title="Customer Spending ANN Dashboard")  # Fixed typo in "Dashboard"

# Title
st.title("Artificial Neural Network for Customer Spending Analysis")
st.markdown("""
This dashboard analyzes credit card customer data to classify customers as High Spenders or Low Spenders using an Artificial Neural Network.
""")

# Sidebar for user inputs
st.sidebar.header("Model Configuration")
epochs = st.sidebar.slider("Number of Epochs", 5, 50, 10)
batch_size = st.sidebar.selectbox("Batch Size", [32, 64, 128, 256], index=2)
hidden_units = st.sidebar.slider("Hidden Units in First Layer", 32, 256, 128, step=32)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.1, 0.5, 0.3, step=0.1)

# Load data function
@st.cache_data
def load_data():
    file_id = '1uy-mxgp4qqeUxOlzJWJLQsi4Le8qD61a'
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    response = requests.get(url)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text), encoding='utf-8')

    # Data preprocessing
    df['Converted'] = (df['PURCHASES'] > df['PURCHASES'].median()).astype(int)
    df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].median(), inplace=True)
    df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median(), inplace=True)
    if 'CUST_ID' in df.columns:
        df.drop(columns=['CUST_ID'], inplace=True)

    return df

df = load_data()

# Data Exploration Section
st.header("Data Exploration")

# Show raw data
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.write(df)

# Data statistics
st.subheader("Data Statistics")
st.write(df.describe())

# Visualizations
st.subheader("Data Visualizations")

# Create tabs for different visualizations
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Distributions",
    "Correlations",
    "Purchase Types",
    "Credit vs Balance",
    "Payment Behavior"
])

with tab1:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Distribution of Key Features', fontsize=16)
    sns.histplot(df['BALANCE'], bins=30, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Balance Distribution')
    sns.histplot(df['PURCHASES'], bins=30, kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Purchases Distribution')
    sns.histplot(df['CREDIT_LIMIT'], bins=30, kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Credit Limit Distribution')
    sns.histplot(df['PAYMENTS'], bins=30, kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Payments Distribution')
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    numerical_cols = ['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES',
                     'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS']
    plt.figure(figsize=(12, 8))
    corr_matrix = df[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap of Numerical Features')
    st.pyplot(plt)

with tab3:
    purchase_types = df[['ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE']]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=purchase_types)
    plt.title('Comparison of Purchase Types')
    plt.ylabel('Amount')
    plt.xlabel('Purchase Type')
    plt.xticks(rotation=45)
    st.pyplot(plt)

with tab4:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='CREDIT_LIMIT', y='BALANCE', data=df, alpha=0.6)
    plt.title('Credit Limit vs Balance')
    plt.xlabel('Credit Limit')
    plt.ylabel('Balance')
    st.pyplot(plt)

with tab5:
    df['PAYMENT_RATIO'] = df['PAYMENTS'] / (df['BALANCE'] + 0.0001)
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='BALANCE', y='PAYMENT_RATIO', data=df, hue='PRC_FULL_PAYMENT', palette='viridis')
    plt.title('Payment Behavior Analysis')
    plt.xlabel('Balance')
    plt.ylabel('Payment Ratio (Payments/Balance)')
    plt.yscale('log')
    plt.legend(title='Full Payment %')
    st.pyplot(plt)

# Model Training Section
st.header("Model Training")

# Split data
X = df.drop(columns=['Converted'])
y = df['Converted']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

# Build model
model = Sequential([
    Dense(hidden_units, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(dropout_rate),
    Dense(hidden_units//2, activation='relu'),
    Dropout(dropout_rate),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)

# Train model
if st.button("Train Model"):
    st.write("Training in progress...")

    history = model.fit(
        X_train_scaled, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_scaled, y_val),
        callbacks=[early_stopping, model_checkpoint],
        class_weight=class_weights_dict,
        verbose=1
    )

    # Plot training history
    st.subheader("Training History")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    # Loss plot
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    st.pyplot(fig)

    # Evaluate model
    st.subheader("Model Evaluation")
    y_pred = model.predict(X_test_scaled)
    y_pred_classes = (y_pred > 0.5).astype(int)

    # Metrics
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred_classes):.4f}")
    st.text(classification_report(y_test, y_pred_classes))

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred_classes)
    fig = px.imshow(cm,
                   labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=['Low Spender', 'High Spender'],
                   y=['Low Spender', 'High Spender'],
                   text_auto=True)
    fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig)

    # Feature importance (using permutation importance)
    st.subheader("Feature Importance")
    try:
        from sklearn.inspection import permutation_importance
        result = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=42)
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': result.importances_mean,
            'Std': result.importances_std
        }).sort_values('Importance', ascending=False)

        fig = px.bar(importance_df, x='Importance', y='Feature', error_x='Std',
                     title='Feature Importance (Permutation)')
        st.plotly_chart(fig)
    except ImportError:
        st.warning("Could not import permutation_importance from sklearn.inspection")

    # Save model
    model.save('customer_spending_model.h5')
    with open('customer_spending_model.h5', 'rb') as f:
        st.download_button(
            label="Download Model",
            data=f,
            file_name="customer_spending_model.h5",
            mime="application/octet-stream"
        )

# Prediction Section
st.header("Make Predictions")

# Create input form for predictions
with st.form("prediction_form"):
    st.subheader("Enter Customer Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        balance = st.number_input("Balance", min_value=0.0, value=1000.0)
        purchases = st.number_input("Purchases", min_value=0.0, value=500.0)
        credit_limit = st.number_input("Credit Limit", min_value=0.0, value=3000.0)

    with col2:
        oneoff_purchases = st.number_input("One-off Purchases", min_value=0.0, value=200.0)
        installments = st.number_input("Installment Purchases", min_value=0.0, value=300.0)
        cash_advance = st.number_input("Cash Advance", min_value=0.0, value=0.0)

    with col3:
        payments = st.number_input("Payments", min_value=0.0, value=800.0)
        minimum_payments = st.number_input("Minimum Payments", min_value=0.0, value=100.0)
        tenure = st.number_input("Tenure (months)", min_value=0, value=12)

    submitted = st.form_submit_button("Predict Spending Category")

    if submitted:
        # Create input array
        input_data = pd.DataFrame({
            'BALANCE': [balance],
            'PURCHASES': [purchases],
            'ONEOFF_PURCHASES': [oneoff_purchases],
            'INSTALLMENTS_PURCHASES': [installments],
            'CASH_ADVANCE': [cash_advance],
            'CREDIT_LIMIT': [credit_limit],
            'PAYMENTS': [payments],
            'MINIMUM_PAYMENTS': [minimum_payments],
            'TENURE': [tenure],
            # Add other features with default values
            'BALANCE_FREQUENCY': [df['BALANCE_FREQUENCY'].median()],
            'PURCHASES_FREQUENCY': [df['PURCHASES_FREQUENCY'].median()],
            'ONEOFF_PURCHASES_FREQUENCY': [df['ONEOFF_PURCHASES_FREQUENCY'].median()],
            'PURCHASES_INSTALLMENTS_FREQUENCY': [df['PURCHASES_INSTALLMENTS_FREQUENCY'].median()],
            'CASH_ADVANCE_FREQUENCY': [df['CASH_ADVANCE_FREQUENCY'].median()],
            'CASH_ADVANCE_TRX': [df['CASH_ADVANCE_TRX'].median()],
            'PURCHASES_TRX': [df['PURCHASES_TRX'].median()],
            'PRC_FULL_PAYMENT': [df['PRC_FULL_PAYMENT'].median()]
        })

        # Scale input data
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_class = "High Spender" if prediction[0][0] > 0.5 else "Low Spender"
        confidence = prediction[0][0] if prediction_class == "High Spender" else 1 - prediction[0][0]

        st.subheader("Prediction Result")
        st.success(f"Predicted Category: {prediction_class}")
        st.info(f"Confidence: {confidence:.2%}")

        # Show probability distribution
        fig = go.Figure(go.Bar(
            x=['Low Spender', 'High Spender'],
            y=[1 - prediction[0][0], prediction[0][0]],
            marker_color=['blue', 'green']
        ))
        fig.update_layout(
            title="Prediction Probabilities",
            xaxis_title="Category",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig)

# Footer
st.markdown("---")
st.markdown("""
**Note:** This dashboard uses an Artificial Neural Network to classify customers based on their spending behavior.
The model was trained on credit card customer data with the goal of predicting whether a customer is a high spender or low spender.
""")

