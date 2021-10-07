from flask import Flask, render_template, redirect, url_for, request
from flask import make_response, jsonify
from flask_cors import CORS
import numpy as np, pickle, gzip
from numpy.random import uniform as runi
from hw2 import DigitNetwork

digitNetwork = DigitNetwork(noParameters=True)
digitNetwork.load()

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
   message = None
   if request.method == 'POST':
        global digitNetwork

        datafromjs = request.json

        return jsonify(str(digitNetwork.classify(np.array(datafromjs))))

if __name__ == "__main__":
    app.run(debug = True)