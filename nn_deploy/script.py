from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
app = Flask(__name__)

with open("neural.pkl", "rb") as file:
    model = file

@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    if request.method == "POST":
        #data = request.data
        data = request.form.to_dict().values()
        data = list(map(int, data))
        data = np.array(data).reshape(1,10)
        preds = (model.predict(data) > 0.5).astype("int32")
        return render_template("result.html", preds)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    




