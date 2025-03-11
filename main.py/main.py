import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import streamlit as st

# Streamlit UI
def main():
    st.title('Titanic Survival Prediction App')

    # Load the dataset
    df_train = pd.read_csv('Titanic_train.csv')
    df_test = pd.read_csv('Titanic_test.csv')

    # Data Preprocessing
    for col in ['Age', 'Fare']:
        df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

    df_train.fillna(df_train.mean(), inplace=True)
    df_test.fillna(df_test.mean(), inplace=True)

    # Encode categorical variables
    df_train = pd.get_dummies(df_train, drop_first=True)
    df_test = pd.get_dummies(df_test, drop_first=True)

    # Align columns
    df_test = df_test.reindex(columns=df_train.columns.drop('Survived'), fill_value=0)

    # Train-test split
    X = df_train.drop('Survived', axis=1)
    y = df_train['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model Building
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Model Evaluation
    y_pred = model.predict(X_test)
    st.write("### Model Evaluation")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"Precision: {precision_score(y_test, y_pred):.2f}")
    st.write(f"Recall: {recall_score(y_test, y_pred):.2f}")
    st.write(f"F1 Score: {f1_score(y_test, y_pred):.2f}")

    # ROC-AUC curve
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.title('ROC-AUC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    st.pyplot(plt)

if __name__ == '__main__':
    main()
