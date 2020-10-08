import flask
from flask import Flask, render_template, request, redirect, Response

from werkzeug.utils import secure_filename

import os

from skimage import color
from skimage import data
import cv2
import numpy as np

import pandas as pd

from io import StringIO
from keras.models import load_model

import matplotlib.pyplot as plt


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
UPLOAD_FOLDER = os.path.basename('data')

app = flask.Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
H = 28
W = 28

#load the model
model = load_model('clf.h5')
clf = model

#run model
def run_model(input_arr, rows = 1):
    input_arr = input_arr.reshape(rows, 28, 28, 1).astype('float') / 255
    return model.predict(input_arr).argmax(axis=1)

@app.route('/', methods=['GET'])
def index_page():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['fileupload']
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (28, 28))
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    out = run_model(img)
    #model.predict(img)
    print(out)
    print(np.argmax(out))
    response = np.array_str(out)
    return response
    
        
# Below model as api ( call with specific test data in request and get response )
# define a ping  function as an endpoint to check health of the service
@app.route("/ping", methods=["GET"])
def ping():
    #health check for container
    health_check_arr = pd.read_csv("health-check-data.csv").values
    
    try:
        result = run_model(health_check_arr)
        return Response(response='{"status": "ok"}', status=200, mimetype='application/json')
    except:
        return Response(response='{"status": "error"}', status=500, mimetype='application/json')    
    
    
#define the predict function to predict output   
@app.route("/invocations", methods=["POST"])
def predict_1():
    if flask.request.content_type == "text/csv":
        X_train = flask.request.data.decode('utf-8')
        X_train = pd.read_csv(StringIO(X_train), header=None).values
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')
    
    #run model
    results = run_model(X_train)
    
    #format into csv
    results_str = ",\n".join(results.astype('str'))
    
    return Response(response=results_str, status=200, mimetype='text/csv')

if __name__ == "__main__":
    # start the flask app, allow remote connections
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port = port)