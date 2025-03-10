from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("furniture_placement_model.pkl")

# Define expected input features (should match what was used during training)
expected_features = ['Room Width', 'Room Height', 'Furniture Width', 'Furniture Height', 
                     'Obstacle X', 'Obstacle Y', 'Obstacle Width', 'Obstacle Height',
                     'Furniture Type_Chair','Furniture Type_Sofa','Furniture Type_Table',
                     'Obstacle Type_Pillar','Obstacle Type_Wall']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.json

        # Convert input to DataFrame
        input_data = pd.DataFrame([data])

        # Ensure input has correct columns
        missing_cols = set(expected_features) - set(input_data.columns)
        if missing_cols:
            return jsonify({"error": f"Missing columns: {missing_cols}"}), 400

        # Predict Furniture Placement (X, Y)
        prediction = model.predict(input_data)

        # Return response
        response = {
            "Furniture X": float(prediction[0][0]),
            "Furniture Y": float(prediction[0][1])
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
