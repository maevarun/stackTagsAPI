from flask import Flask, request, render_template

import pickle

import numpy as np
import pandas as pd

import Modules.Preprocessing.stopwords as sw
import Modules.Preprocessing.tokenAndLemmatiz as tal

app = Flask(__name__)

modelOVR = pickle.load(open('ModelsAPI/model_OVR.pkl', 'rb'))

@app.route("/")
def template():
    return "Bienvenue sur le projet : Catégoriser vos questions!"

@app.route("/onevsrest")
def homepage():
    return render_template("index.html")

def predict_tag():
    question = request.args.get('question')
    print(question)
    question_vec = vectorizer.transform(question)
    prediction = modelOVR.predict(question_vec)
    return prediction

# Désactiver cette fonction au moment de la mise en ligne
if __name__ == "__main__":
    app.run(debug=True)