import sys
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('averaged_perceptron_tagger')

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath):
    """Function to load the data set from the database.

    Args:
        database_filepath: location of the database


    Returns:
        X: independent variable
        Y: target variable
        cols: column list

    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM Messages", engine)
    X = df.message.values
    Y = df.drop(['id','message','original','genre'],axis=1).values
    col = df.drop(['id','message','original','genre'],axis=1)
    cols = (col.columns).tolist()
    return X,Y,cols

def tokenize(text):
     detected_urls = re.findall(url_regex, text)
     for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

     tokens = word_tokenize(text)
     lemmatizer = WordNetLemmatizer()

     clean_tokens = []
     for tok in tokens:

         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
         clean_tokens.append(clean_tok)

     return clean_tokens



def build_model():
    """Function to build a machine learning pipeline.

    Args:
        none


    Returns:
        cv

    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

    parameters  = {
    'vect__max_df': (2.0,3.0),
    'vect__ngram_range': ((1, 1),(1, 2)),
     
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """Function to evaluate the machine learning pipeline.

    Args:
        model: pipeline
        X_test: test data independent variable
        Y_test: test data target variable
        category_names: column names

    Returns:
        Classification report

    """
    # predict on test data
    y_pred = model.predict(X_test)

    def display_results(y_test, y_pred):

        for index in range(len(cols)):


            labels = np.unique(y_pred)
            classification_rep = classification_report(y_test[:, index], y_pred[:, index], labels=labels)

            print("Labels:", labels)
            print("Classification Report:\n", classification_rep)

        # display results
        display_results(y_test, y_pred)


def save_model(model, model_filepath):
    """Function to save model in a pickle file.

    Args:
        model: model training results
        model_filepath: location of the save pickle file

    Returns:
        classifier.pkl file

    """

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
