from flask import Flask, render_template, request
import pickle
import pandas as pd


app = Flask(__name__)

# Load the machine learning model from the pickle file
model = pickle.load(open('model1.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        user_input = {
            'HighBP': int(request.form['HighBP']),
            'HighChol': int(request.form['HighChol']),
            'BMI': float(request.form['BMI']),
            'Smoker': int(request.form['Smoker']),
            'Stroke': int(request.form['Stroke']),
            'HeartDiseaseorAttack': int(request.form['HeartDiseaseorAttack']),
            'PhysActivity': int(request.form['PhysActivity']),
            'HvyAlcoholConsump': int(request.form['HvyAlcoholConsump']),
            'NoDocbcCost': int(request.form['NoDocbcCost']),
            'GenHlth': int(request.form['GenHlth']),
            'MentHlth': int(request.form['MentHlth']),
            'PhysHlth': int(request.form['PhysHlth']),
            'DiffWalk': int(request.form['DiffWalk']),
            'Age': int(request.form['Age']),
            'Education': int(request.form['Education']),
            'Income': int(request.form['Income'])
        }

        # Make a prediction using the trained model
        input_data = pd.DataFrame([user_input])
        prediction = model.predict(input_data)

        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
    app.run(debug=True)

