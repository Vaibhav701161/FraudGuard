from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model1 = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return "Welcome to the Home Page!"

@app.route('/test')
def test():
    return {"members": ["Member 1", "Member 2"]}

@app.route('/predict', methods=['POST','GET']) #NEED TO SEND THE DATA HERE FROM REACT
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(int_features)
    print(final)
    
    prediction = model1.predict_proba(final)
    output = '{0:.{1}f}.format(prediction[0][1],2)'
    
    #NEED TO RETURE THE PREDICTION TO MAIN PAGE

if __name__ == "__main__":
    app.run(debug=True)
