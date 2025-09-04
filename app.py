import warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

try:
    # Load the pipeline
    pipeline = joblib.load('model/pipeline.joblib')
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            # Get values from the form
            input_data = pd.DataFrame({
                'bedroom': [float(request.form['bedroom'])],
                'layout_type': [request.form['layout_type']],
                'property_type': [request.form['property_type']],
                'locality': [request.form['locality']],
                'area': [float(request.form['area'])],
                'furnish_type': [request.form['furnish_type']],
                'bathroom': [float(request.form['bathroom'])],
                'city': [request.form['city']]
            })

            # Make prediction using the pipeline
            prediction = pipeline.predict(input_data)[0]

            return render_template('result.html', 
                                prediction=f'₹{prediction:,.2f}',
                                success=True)
    except Exception as e:
        return render_template('result.html', 
                             error=f"Error making prediction: {str(e)}",
                             success=False)

if __name__ == '__main__':
    app.run(debug=True)