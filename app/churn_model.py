# app/churn_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
def load_data():
    data = pd.read_csv('data/churn_data.csv')
    return data

# Preprocess data
def preprocess_data(data):
    data = data.dropna()
    data = pd.get_dummies(data, drop_first=True)
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    return X, y

# Train model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy:.2f}')

    # Save the model
    with open('model/churn_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    # Return the model and scaler
    return model, scaler

# Main function to train model
def main():
    data = load_data()
    X, y = preprocess_data(data)
    model, scaler = train_model(X, y)

if __name__ == '__main__':
    main()