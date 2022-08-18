from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open("./model/model.pkl","rb"))

app = Flask(__name__)

## flask 앱의 루트 디렉토리 초기화

@app.route('/')

def main():
    return render_template("home.html")

@app.route('/predict', methods=['POST']) 
def start():
    val1 = request.form['a']
    val2 = request.form['b']
    val3 = request.form['c']
    arr = np.array([[val1, val2, val3]])
    pred = model.predict(arr)
    print("start pred ", pred)
    return render_template('after.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)