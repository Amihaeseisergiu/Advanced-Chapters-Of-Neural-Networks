from flask import Flask, render_template, redirect, url_for, request
from flask import make_response, jsonify
from flask_cors import CORS
import numpy as np, pickle, gzip, os
from numpy.random import uniform as runi
from hw3 import DigitNetwork

digitNetwork = DigitNetwork(load=True)
digitNetwork.load()

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
   message = None
   if request.method == 'POST':
        global digitNetwork

        datafromjs = request.json
        digit = np.array(datafromjs)
        column_digit = digit.reshape(len(digit), 1)

        return jsonify(str(digitNetwork.classify(column_digit)))

if __name__ == "__main__":
    app.run(debug = True)