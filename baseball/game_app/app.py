import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from werkzeug.utils import secure_filename
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('home.html')


@app.route('/model')
def home():
    return render_template('model.html')

@app.route('/about')
def about():
    return render_template('about.html')
    

@app.route('/model/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    func = lambda prediction : "This team would qualify to the Playoffs" if prediction==[1] else "This team would not qualify to the Playoffs"
    prediction = model.predict(final_features)

    return render_template('model.html', prediction_text=func(prediction))


if __name__ == "__main__":
    app.run(debug = True)