from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model_path='./models/decision_tree_obj.lb'
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('myhome.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form values
        pregnancies = float(request.form['Pregnancies'])
        glucose = float(request.form['Glucose'])
        bloodpressure = float(request.form['BloodPressure'])
        skinthickness = float(request.form['SkinThickness'])
        insulin = float(request.form['Insulin'])
        bmi =float(request.form['BMI'])
        diabetespedigreefunction = float(request.form['DiabetesPedigreeFunction'])
        age =float(request.form['Age'])

        # Create an array with the inputs
        data = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age]])

        # Predict the outcome
        prediction = model.predict(data)
        output = 'Diabetic' if prediction[0] == 1 else 'Non-diabetic'

        return render_template('myhome.html', prediction_text=f'Prediction: {output}')

if __name__ == '__main__':
    app.run(debug=True)

