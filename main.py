import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score as ras
from sklearn.metrics import confusion_matrix

# Function to load data in chunks for large files
def load_data(file):
    chunk_size = 10000  # Adjust based on your memory capacity
    return pd.concat(pd.read_csv(file, chunksize=chunk_size))

# Function to optimize memory usage
def optimize_memory(data):
    for col in data.select_dtypes(include=['float']):
        data[col] = pd.to_numeric(data[col], downcast='float')
    for col in data.select_dtypes(include=['int']):
        data[col] = pd.to_numeric(data[col], downcast='integer')
    for col in data.select_dtypes(include=['object']):
        data[col] = data[col].astype('category')
    return data

# Define navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Welcome", "Credit Card Fraud Detection", "Online Payment Fraud Detection"])

# Welcome Page
if page == "Welcome":
    st.title("Welcome to FraudGuard")
    st.subheader("This application helps you detect and prevent fraud in real time.")
    st.write("Please use the sidebar to navigate between the tools.")

# Fraud Detection Page
elif page == "Credit Card Fraud Detection":
    st.title("Credit Card Fraud Detection")

    # Upload the dataset
    uploaded_file = st.file_uploader("Upload the Fraud Detection Dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        # Load the dataset
        data = pd.read_csv(uploaded_file)

        # Display the first few rows of the dataset
        st.subheader('Dataset Overview:')
        st.write(data.head())

        # Display basic stats of the dataset
        st.subheader('Dataset Summary:')
        st.write(data.describe())

        # Determine the number of fraud and valid cases
        fraud = data[data['Class'] == 1]
        valid = data[data['Class'] == 0]
        outlier_fraction = len(fraud) / float(len(valid))

        st.subheader("Fraud vs Valid Transactions:")
        st.write(f"Fraud Cases: {len(fraud)}")
        st.write(f"Valid Transactions: {len(valid)}")
        st.write(f"Outlier Fraction: {outlier_fraction:.4f}")

        # Display Amount details for fraud and valid transactions
        st.subheader("Amount Distribution of Fraudulent Transactions:")
        st.write(fraud['Amount'].describe())

        st.subheader("Amount Distribution of Valid Transactions:")
        st.write(valid['Amount'].describe())

        # Display the correlation matrix
        st.subheader("Correlation Matrix:")
        corrmat = data.corr()
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corrmat, vmax=.8, square=True, ax=ax)
        st.pyplot(fig)

        # Splitting the data into training and testing sets
        X = data.drop(columns=['Class'])
        Y = data['Class']
        xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Training the model
        if st.button('Train Random Forest Model'):
            st.write("Training the model...")
            rfc = RandomForestClassifier()
            rfc.fit(xTrain, yTrain)

            # Predictions
            yPred = rfc.predict(xTest)

            # Display performance metrics
            acc = accuracy_score(yTest, yPred)
            prec = precision_score(yTest, yPred)
            rec = recall_score(yTest, yPred)
            f1 = f1_score(yTest, yPred)
            MCC = matthews_corrcoef(yTest, yPred)

            st.subheader("Model Performance:")
            st.write(f"Accuracy: {acc:.4f}")
            st.write(f"Precision: {prec:.4f}")
            st.write(f"Recall: {rec:.4f}")
            st.write(f"F1 Score: {f1:.4f}")
            st.write(f"Matthews Correlation Coefficient: {MCC:.4f}")

            # Display confusion matrix
            st.subheader("Confusion Matrix:")
            conf_matrix = confusion_matrix(yTest, yPred)
            fig, ax = plt.subplots(figsize=(12, 9))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', ax=ax, xticklabels=['Valid', 'Fraud'], yticklabels=['Valid', 'Fraud'])
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            st.pyplot(fig)


# Online Fraud Detection Page
elif page == "Online Payment Fraud Detection":
    st.title("Online Payment Fraud Detection")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload the Online Fraud Dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        # Load and optimize dataset
        data = load_data(uploaded_file)
        data = optimize_memory(data)

        st.subheader("Dataset Overview")
        st.write(data.head())
        st.write(data.describe())

        # Sample a fraction of the dataset for visualization
        sample_data = data.sample(frac=0.1, random_state=42)

        # Data visualization
        st.subheader("Data Visualizations")
        fig, ax = plt.subplots()
        sns.countplot(x='type', data=sample_data, ax=ax)
        st.pyplot(fig)

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        numeric_data = data.select_dtypes(include=[float, int])
        corr_matrix = numeric_data.corr()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(corr_matrix, cmap='BrBG', fmt='.2f', linewidths=2, annot=True, ax=ax)
        st.pyplot(fig)

        # Data Preprocessing
        st.subheader("Preprocessing Data")
        type_dummies = pd.get_dummies(data['type'], drop_first=True)
        data_new = pd.concat([data, type_dummies], axis=1)
        X = data_new.drop(['isFraud', 'type', 'nameOrig', 'nameDest'], axis=1)
        y = data_new['isFraud']

        st.write(f"Features shape: {X.shape}")
        st.write(f"Labels shape: {y.shape}")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Model training
        st.subheader("Model Training")
        models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf', probability=True), RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=7)]

        if st.button("Train Models"):
            for model in models:
                model.fit(X_train, y_train)
                train_preds = model.predict_proba(X_train)[:, 1]
                val_preds = model.predict_proba(X_test)[:, 1]
                st.write(f'{model.__class__.__name__}:')
                st.write(f'Training AUC: {ras(y_train, train_preds):.4f}')
                st.write(f'Validation AUC: {ras(y_test, val_preds):.4f}')
                st.write("---")

        # Model Evaluation
        st.subheader("Model Evaluation")
        if st.button("Show Confusion Matrix for XGBoost"):
            confusion_matrix(models[1], X_test, y_test)
            plt.show()
            st.pyplot()

