from flask import Flask, render_template, redirect, url_for, request
from flask import make_response, jsonify
from flask_cors import CORS
import numpy as np, pickle, gzip
from numpy.random import uniform as runi

class DigitNetwork:
    def __init__(self):
        with open('./weights.npy', 'rb') as f:
            self.weights = np.load(f)
        
        with open('./biases.npy', 'rb') as f:
            self.biases = np.load(f)

    def classify(self, input):
        return np.argmax(np.dot(self.weights, input) + self.biases)

digitNetwork = DigitNetwork()

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