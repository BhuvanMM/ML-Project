from flask import render_template, request
from app import app
import pickle
import numpy as np

# Load the model
model = pickle.load(open('model/churn_model.pkl', 'rb'))

# Define the prediction function
def predict_churn(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input data from the form and convert to float
        input_data = [float(x) for x in request.form.values()]
        
        # Check if the number of features matches the expected count (13)
        if len(input_data) != 13:
            error_message = f"Expected 13 features, but got {len(input_data)}"
            return render_template('result.html', result=error_message)

        # Predict churn
        prediction = predict_churn(input_data)

        # Prepare result
        result = 'Churn' if prediction == 1 else 'No Churn'
        return render_template('result.html', result=result)

    except Exception as e:
        # Handle errors (e.g., invalid input)
        error_message = f"Error processing input: {e}"
        return render_template('result.html', result=error_message)
