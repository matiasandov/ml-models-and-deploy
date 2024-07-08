import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

#decir que url debe  trigger que function
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template("index.html")

#funcion para hacer prediction

def predictForm(predList):
    #convertir a uni dimensional
    toPred = np.array(predList).reshape(1,12)
    #rb = read binary
    model = pickle.load(open("model.pkl", "rb"))
    result = model.predict(toPred)
    return result[0]

#metodo post 
#direcciona a result
@app.route("/result", methods = ['POST'])
def result():
    if request.method == 'POST':
        #hacer la dicc
        pred = request.form.to_dict()
        #tomar solo los valores
        pred = list(pred.values())
        pred = list(map(int, pred))
        res = predictForm(pred)
        
        predRes = ""
        if int(res) == 0:
            predRes = "Income lower than 50k"
        else:
            predRes = "Income more than 50k"
        
        print(predRes)
        return render_template("result.html", prediction=predRes)
            
        
if __name__ == "__main__":
    app.run(debug=True)
