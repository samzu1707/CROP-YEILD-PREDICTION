import joblib
import numpy as np
from flask import Flask, render_template, request

try:
    model = joblib.load('random_forest_regressor_model.pkl')
except FileNotFoundError:
    model = None
app = Flask(__name__)

FEATURE_NAMES = [
    'Crop', 
    'Crop_Year', 
    'Season', 
    'State', 
    'Annual_Rainfall', 
    'Fertilizer', 
    'Pesticide'
]

@app.route('/')
def index():
    """Route to render the main input form."""

    return render_template('index.html', feature_names=FEATURE_NAMES, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    """Route to handle form submission and make predictions."""
    if not model:
        pass

    try:
        data = request.form.to_dict()

        input_values = [float(data[name]) for name in FEATURE_NAMES]

        final_features = np.array([input_values])
        prediction = model.predict(final_features)
        output = f"Predicted Yield (or Target Value): {prediction[0]:.2f}"

    except Exception as e:

        output = f"Prediction Error: Invalid input or data mismatch. Details: {e}"
        return render_template('index.html', feature_names=FEATURE_NAMES, prediction=output, is_error=True)

    return render_template('index.html', feature_names=FEATURE_NAMES, prediction=output)

if __name__ == '__main__':
    app.run(debug=True)