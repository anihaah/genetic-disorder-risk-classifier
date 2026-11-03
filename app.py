from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load saved model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return "Genetic Disorder Risk Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[
        data['Age'],
        data['Blood_Pressure'],
        data['Gene_Mutation_Score'],
        data['Family_History']
    ]])
    
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0][1]
    
    return jsonify({
        'risk_prediction': int(prediction),
        'risk_probability': float(probability)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
