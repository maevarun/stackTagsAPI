from flask import Flask, request, render_template

import pickle
import joblib

import numpy as np
import pandas as pd

import Preprocessing.stopwords as sw
import Preprocessing.tokenAndLemmatiz as tal
import Preprocessing.cvAndTfIdf as cvtf
# vectorizer = joblib.load("Preprocessing/vectorizer.pkl")
# vectorizer = pickle.load(open('Preprocessing/vectorizer.pkl', 'rb'))
vectorizer = pickle.load(open('Preprocessing/vectorizer.pkl', 'rb'))

modelOVR = pickle.load(open('ModelsAPI/model_OVR.pkl', 'rb'))

app = Flask(__name__)

@app.route("/")
def template():
    return "Bienvenue sur le projet : Catégoriser vos questions!"

@app.route("/onevsrest", methods=['POST'])
def predict_tag():
    request_data = request.get_json()

    question = None

    if 'question' in request_data:
        question = request_data['question']
        preprocess_punct = sw.kill_punctuation(question)
        question_clean = sw.Preprocess_listofSentence(preprocess_punct)
        question_list = list(cvtf.sent_to_words(question_clean))
        question_lemmatized = cvtf.lemmatization(question_list, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        vecto = vectorizer.transform(question_lemmatized)
        pred = modelOVR.predict(vecto)
    return ''' Le tag est taratata: {}'''.format(pred)

@app.route('/query-example')
def query_example():
    # if key doesn't exist, returns None
    language = request.args.get('language')
    print(language)
    # if key doesn't exist, returns a 400, bad request error
    framework = request.args['framework']
    print(framework)
    # if key doesn't exist, returns None
    website = request.args.get('website')
    return '''
                 <h1>The language value is: {}</h1>
                 <h1>The framework value is: {}</h1>
                 <h1>The website value is: {}'''.format(language, framework, website)

@app.route('/form-example', methods=['GET', 'POST'])
def form_example():
    # handle the POST request
    if request.method == 'POST':
        language = request.form.get('language')
        framework = request.form.get('framework')
        return '''
              <h1>The language value is: {}</h1>
              <h1>The framework value is: {}</h1>'''.format(language, framework)

    # otherwise handle the GET request
    return '''
           <form method="POST">
               <div><label>Language: <input type="text" name="language"></label></div>
               <div><label>Framework: <input type="text" name="framework"></label></div>
               <input type="submit" value="Submit">
           </form>'''

# @app.route('/json-example', methods=['POST'])
# def json_example():
#     request_data = request.get_json()
#
#     language = None
#     framework = None
#     python_version = None
#     example = None
#     boolean_test = None
#
#     if request_data:
#         if 'language' in request_data:
#             language = request_data['language']
#
#         if 'framework' in request_data:
#             framework = request_data['framework']
#
#         if 'version_info' in request_data:
#             if 'python' in request_data['version_info']:
#                 python_version = request_data['version_info']['python']
#
#         if 'examples' in request_data:
#             if (type(request_data['examples']) == list) and (len(request_data['examples']) > 0):
#                 example = request_data['examples'][0]
#
#         if 'boolean_test' in request_data:
#             boolean_test = request_data['boolean_test']
#
#     return '''
#                The language value is: {}
#                The framework value is: {}
#                The Python version is: {}
#                The item at index 0 in the example list is: {}
#                The boolean value is: {}'''.format(language, framework, python_version, example, boolean_test)

# Désactiver cette fonction au moment de la mise en ligne
if __name__ == "__main__":
    app.run(debug=True)