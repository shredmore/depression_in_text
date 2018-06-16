# To run use the following bash commands
# $ export FLASK_APP=hello.py
# $ flask run

import numpy as np
import pandas as pd
import string
from flask import Flask, render_template, request
from sklearn.externals import joblib
from common import utils, vocabulary
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import *

# # import nltk library
import nltk; nltk.download('punkt')
from nltk import sent_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize.treebank import TreebankWordTokenizer

# # import stopword libraries
nltk.download('stopwords'); from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words


#####
#APP
####
app = Flask(__name__)


#####
#PREPROCESSOR
#####
def tokenize_text(input_text):
    """
    Args: 
    input_text: a string representing an 
    individual review
        
    Returns:
    input_token: a list containing stemmed 
    tokens, with punctutations removed, for 
    an individual review
        
    """
    input_tokens=[]
        
    # Split sentence
    sents=sent_tokenize(input_text)
            
    # Split word
    for sent in sents:
        input_tokens+=TreebankWordTokenizer().tokenize(sent)
        
    return input_tokens


# # canonicalize
def canonicalize_tokens(input_tokens):
    """
    Args:
    input_tokens: a list containing tokenized 
    tokens for an individual review
    
    Returns:
    input_tokens: a list containing canonicalized 
    tokens for an individual review
    
    """
    input_tokens=utils.canonicalize_words(input_tokens)
    return input_tokens


# # preprocessor 
def preprocessor(raw_text):
    """
    Args:
    raw_text: a string representing an
    individual review
    
    Returns:
    preprocessed_text: a string representing 
    a preprocessed individual review
    
    """
    # tokenize
    tokens=tokenize_text(raw_text)
    
    # canonicalize
    canonical_tokens=canonicalize_tokens(tokens)
    
    # rejoin string
    preprocessed_text=(" ").join(canonical_tokens) 
    return preprocessed_text

#####
#STOPWORDS
####
# sklearn stopwords (frozenset)
sklearn_stopwords=stop_words.ENGLISH_STOP_WORDS
print("number of sklearn stopwords: %d" %(len(sklearn_stopwords)))
#print(sklearn_stopwords)

# nltk stopwords (list)
nltk_stopwords=stopwords.words("english")
print("number of nltk stopwords: %d" %(len(nltk_stopwords)))
#print(nltk_stopwords)

# combined sklearn, nltk, other stopwords (set)
total_stopwords=set(list(sklearn_stopwords.difference(set(nltk_stopwords)))+nltk_stopwords)

other_stopwords=["DG", "DGDG", "@", "rt", "'rt", "'", ":", "depression"]
for w in other_stopwords:
    total_stopwords.add(w)

@app.route('/')
def hello():
	return render_template('journal.html')
	# return "Hello"

@app.route('/submit', methods=['POST'])
def submit_textarea():
	# store the given text in a variable
	text = request.form.get("text")

	#trasform user input text to imported (pre-trained) tfidf matrix 
	export_test_example=loaded_tfidf.transform([text]) 

	#feed the tfidf matrix into pre-trained logistic regression
	#model and get scores
	score = loaded_lr.predict_proba(export_test_example)
	# string_1 = "Based on the language used in your journal entry here are the probabilities of no depression (left) to signs of depression (right): "

    #response
	# response = string_1 + str(score)

	#get words and weights from test journal
	#follow this tutorial - https://sarahleejane.github.io/learning/python/2015/08/09/simple-tables-in-webapps-using-flask-and-pandas-with-python.html
	coef_sq = loaded_lr.coef_
	word_idx = np.nonzero(export_test_example)[1]
	vocab = np.array(loaded_tfidf.get_feature_names())[word_idx]
	weights = coef_sq[:, word_idx]
	df = pd.DataFrame({'Weights of words in sample Journal': weights[0]}
                  , index=vocab)
	df = df.sort(columns=['Weights of words in sample Journal'], ascending=False, inplace=True)
	table = df.to_html()

	#return score. have to use format() otherwise
	#will throw an error. something specific to flask
	# return 'Your scores are: {}'.format(score)
	return render_template('results.html', results=score, table=table, text=text)


if __name__ == '__main__':
	#load models
	loaded_tfidf = joblib.load('tfidf_exported_model')
	loaded_lr = joblib.load('logistic_regression_model')
	#start app
	app.run(host='0.0.0.0', port=8080, debug=True)
