from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np

# LOAD DATA
categories = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]

# returns bunch: dictionary-like obj with data(list of str), target(ndarray), target_names(list), full description,
# filenames(ndarray)
twenty_train = load_files(container_path='C:\\20news-bydate-train', description=None, categories=categories,
                          load_content=True, shuffle=True, encoding='latin-1', decode_error='strict', random_state=42,
                          allowed_extensions=None)
twenty_test = load_files(container_path='C:\\20news-bydate-test', description=None, categories=categories,
                         load_content=True, shuffle=True, encoding='latin-1', decode_error='strict', random_state=42,
                         allowed_extensions=None)

# BUILD A PIPELINE (make the vectorizer => transformer => classifier easier to work with)
# Each step in the pipeline is a tuple consisting of a name (string) and a transformer or estimator (object).
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3, random_state=42, max_iter=5, tol=None))
])

# PARAMETER TUNING (grid search)
# Find the best combination of parameters to optimize the model's performance.
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3)
}

gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)

# CountVectorizer is fitted to twenty_train.data.
# TfidfTransformer is fitted to the output of CountVectorizer.
# MultinomialNB is fitted to the output of TfidfTransformer.
gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)

docs_test = twenty_test.data
prediction = gs_clf.predict(docs_test)


