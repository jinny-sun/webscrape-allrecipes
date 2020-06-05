import streamlit as st
import numpy as np
import pandas as pd
import re

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    import string
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    wnl = WordNetLemmatizer()

    def lemmatize(string):
        for word in re.findall(r"[a-z]+", string):
            string = string.replace(word, wnl.lemmatize(word, 'n') if 's' in word[-3:] else word)
        return string

    # Remove anything in parenthesis
    mess = re.sub(r"\([^\)]+\)", '', mess)

    # Make everything lowercase
    mess = mess.lower()
    # Remove non-word punctuation
    mess =' '.join(re.findall(r"[-,''\w]+", mess)) # This leaves some commas as a character #
    mess = re.sub(r"\,", ' ', mess)
    # Remove punctuation and numbers
    #mess = ''.join([char for char in mess if char not in string.punctuation])
    mess = ''.join([i for i in mess if not i.isdigit()])
    # Remove plurals
    mess = lemmatize(mess)
    #clean excess whitespace
    mess = re.sub(r"\s+", ' ', mess).strip()
    # Remove stopwords
    return([word for word in mess.split() if word.lower() not in stopwords.words('english')])

st.title("What's 4 Dinner?")

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')
y_cal_train = pd.read_csv('y_cal_train.csv')
y_cal_test = pd.read_csv('y_cal_test.csv')

# This includes the list of ingredients before cleaning
df = pd.read_csv('clean_df.csv')

option = st.selectbox(
    'Which recipe would you like to cook for dinner?',
    y_cal_test['name'])

'You selected: ', option

test_key = y_cal_test[y_cal_test['name']==option]['key'][0]
test_ingredient = X_test[X_test['key']==test_key]['clean_ing']
test_label = y_cal_test[y_cal_test['name']==option]['label'][0]

# Calorie prediction v1 - same code as jupyter notebook
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer # Bag of Words
from sklearn.feature_extraction.text import TfidfTransformer # TF-IDF
from sklearn.linear_model import LogisticRegression
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
#    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', LogisticRegression()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

pipeline.fit(X_train['clean_ing'],y_cal_train['label']) # Fit Model using training data
# predictions = pipeline.predict(X_test['clean_ing']) # Predict using test data
predictions = pipeline.predict(test_ingredient) # Predict using test data

# Calorie Prediction v2
# import pickle
# from sklearn.linear_model import LogisticRegression
# ingredient_bow = pickle.load(open('ingredient_bow.sav','rb'))
# predictions = ingredient_bow.score(test_ingredient,test_label)
# predictions

"""
Calorie Predictor
"""
if predictions == 0:
    'This recipe is less than or equal to 600 calories per serving'
else:
    'This recipe is more than 600 calories per serving'

"""
Ingredients
"""
# Using original df_ing equation with get)units(x)
# df[df['name']==option][['quantity','unit','ingredient']]

# Using updated df_ing equation with get_quantity(x)
df[df['name']==option][['quantity','ingredient']]
