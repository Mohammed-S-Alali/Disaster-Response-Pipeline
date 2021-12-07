import sys
import pandas as pd
import nltk
import re
import pickle

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,  GridSearchCV 
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from sqlalchemy import create_engine

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])

def load_data(database_filepath):
    """
    This function loads database file by filepath.

    Parameters
    ----------
    database_filepath:str, path object or file-like object for database\n
    Any valid string path is acceptable. The string could be a URL.

    Returns
    -------
    X
        A dataframe of message is returned as two-dimensional data structure with labeled axes.
    
    Y
        A dataframe of message categories are returned as two-dimensional data structure with labeled axes.
    
    category_names
        A list of category names are returned as one-dimensional data structure.
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    
    category_names = Y.columns

    return X,Y,category_names


def tokenize(text):
    """
    This function generates a cleaned list of tokens from text of message.

    Parameters
    ----------
    text:str\n
    Any valid string path is acceptable. The string could be a URL.

    Returns
    -------
    clean_tokens
        A list of tokens is returned as one-dimensional data structure.
    """

    # regex for URLs to be replaced with a placeholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex,text)

    for url in detected_urls:
        text = text.replace(url,"urlplaceholder")
    
    # the words in the text input to then be split, tokenised and lemmatized, removing stop words
    words = word_tokenize(text)

    # remove stop words
    stopwords_en = stopwords.words("english")
    words = [word for word in words if word not in stopwords_en]

    # lemmatization
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(word) for word in words]

    return clean_tokens


def build_model():
    """
    This function generates a model after doing a pipeline for GridSerachCV.

    Parameters
    ----------
    No parameters

    Returns
    -------
    model
        A GridSearchCV object is returned with defined pipeline, cv, param_grid and verbose.
    """
    
    # create a pipeline
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer = tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))    
    ])

    # parameters of Grid search
    parameters = {
                'vect__min_df': [1, 5],
                'tfidf__use_idf': [True, False],
                'clf__estimator__n_estimators':[10,50],
                'clf__estimator__min_samples_split': [2, 3]
                }

    model = GridSearchCV(pipeline,cv=3, param_grid=parameters,verbose=2)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function prints a classification report after predicting the category of messages.

    Parameters
    ----------
    model:GridSearchCV Object\n
    A GridSearchCV object with defined pipeline, cv, param_grid and verbose.

    X_test:Dataframe\n
    A splited dataframe of message for test as two-dimensional data structure with labeled axes.

    Y_test:Dataframe\n
    A splited dataframe of message categories for test as two-dimensional data structure with labeled axes.

    category_names:list\n
    A list of category names as one-dimensional data structure.

    Returns
    -------
    No returns
    """

    # predict the categroy of message
    Y_pred = model.predict(X_test)
    
    # convert ndarray to dataframe
    Y_pred = pd.DataFrame(Y_pred)

    # rename the columns of dataframe
    Y_pred.columns = category_names

    print(classification_report(Y_test,Y_pred, target_names = category_names))


def save_model(model, model_filepath):
    pickle.dump(model,open(model_filepath,'wb'))



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