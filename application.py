from flask import Flask, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import json

application = Flask(__name__)

@application.route("/")
def index():
    return "Your Flask App Works! V1.0"

@application.route('/run', methods=['GET','POST'])
def load_model():
    loaded_model = None
    with open('basic_classifier.pk1','rb') as fid:
        loaded_model = pickle.load(fid)

    vectorizer=None
    with open('count_vectorizer.pk1','rb') as vd:
        vectorizer = pickle.load(vd)
    
    #prediction=loaded_model.predict(vectorizer.transform(['This is fake']))[0]

    if request.method == "POST":
        input = request.json.get('input')
        prediction = loaded_model.predict(vectorizer.transform([input]))[0]
        return prediction



if __name__ == '__main__':
    #application.run(host="127.0.0.1",port=5000, debug=True)
    application.run(port=5000, debug=True)