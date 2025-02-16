from flask import Flask, render_template, request
import joblib
import numpy as np
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model (ensure the pickle file is in your project folder)
model = joblib.load("eth_by_stud_perform.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    var_1 = float(request.form['math score'])
    var_2 = float(request.form['reading score'])
    var_3 = float(request.form['writing score'])

    # Create an array of input data
    input_data = np.array([[var_1, var_2, var_3]])

    # Make a prediction using the model
    prediction = model.predict(input_data)

    # Map prediction result to human-readable text
    result = prediction[0]

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    # Get the PORT environment variable, default to 5000 if not provided
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
