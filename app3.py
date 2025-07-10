import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# Title
st.title("Early Disease Prediction App")
st.markdown("Upload a dataset and predict whether an individual is likely to have heart disease.")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Raw Data Preview")
    st.dataframe(df.head())

    # Drop unnecessary columns
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)
    if 'date' in df.columns:
        df.drop('date', axis=1, inplace=True)

    # Encode categorical variables
    if 'country' in df.columns:
        df['country'] = LabelEncoder().fit_transform(df['country'])
    if 'occupation' in df.columns:
        df['occupation'] = LabelEncoder().fit_transform(df['occupation'])

    # Define X and y
    X = df.drop('disease', axis=1)
    y = df['disease']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # EDA
    st.subheader("🔎 Exploratory Data Analysis")

    # Age distribution
    st.markdown("#### 📊 Age Distribution")
    fig1 = plt.figure(figsize=(8,4))
    sns.histplot(df['age'], kde=True, color='orange')
    st.pyplot(fig1)

    # Correlation heatmap
    st.markdown("#### 🔗 Correlation Matrix")
    fig2 = plt.figure(figsize=(12,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    st.pyplot(fig2)

    # Model Training and Evaluation
    st.subheader("🧠 Model Training & Evaluation")

    def evaluate_model(model, X_test, y_test, name):
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.markdown(f"**{name} Accuracy:** `{acc * 100:.2f}%`")
        st.text(classification_report(y_test, y_pred))

        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name} - Confusion Matrix")
        st.pyplot(fig)

    # Logistic Regression
    st.markdown("### 📌 Logistic Regression")
    log_model = LogisticRegression(random_state=42)
    log_model.fit(X_train, y_train)
    evaluate_model(log_model, X_test, y_test, "Logistic Regression")

    # Decision Tree
    st.markdown("### 🌳 Decision Tree")
    tree_model = DecisionTreeClassifier(random_state=42)
    tree_model.fit(X_train, y_train)
    evaluate_model(tree_model, X_test, y_test, "Decision Tree")

    # SVM
    st.markdown("### 🧮 Support Vector Machine")
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    evaluate_model(svm_model, X_test, y_test, "SVM")

    # Hyperparameter Tuning
    st.subheader("🎯 Hyperparameter Tuning (GridSearchCV)")

    with st.spinner("Running GridSearchCV..."):

        # Logistic Regression Tuning
        param_log = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'saga']}
        grid_log = GridSearchCV(LogisticRegression(random_state=42), param_log, cv=5, scoring='accuracy')
        grid_log.fit(X_train, y_train)

        st.markdown(f"**Best Logistic Regression Params:** `{grid_log.best_params_}`")
        st.markdown(f"**Best Logistic Regression CV Accuracy:** `{grid_log.best_score_ * 100:.2f}%`")

        # Decision Tree Tuning
        param_tree = {
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_tree = GridSearchCV(DecisionTreeClassifier(random_state=42), param_tree, cv=5, scoring='accuracy')
        grid_tree.fit(X_train, y_train)

        st.markdown(f"**Best Decision Tree Params:** `{grid_tree.best_params_}`")
        st.markdown(f"**Best Decision Tree CV Accuracy:** `{grid_tree.best_score_ * 100:.2f}%`")

        # SVM Tuning
        param_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        grid_svm = GridSearchCV(SVC(random_state=42), param_svm, cv=5, scoring='accuracy')
        grid_svm.fit(X_train, y_train)

        st.markdown(f"**Best SVM Params:** `{grid_svm.best_params_}`")
        st.markdown(f"**Best SVM CV Accuracy:** `{grid_svm.best_score_ * 100:.2f}%`")

else:
    st.info("Please upload a valid CSV file to begin.")