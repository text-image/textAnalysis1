from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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

text_clf.fit(twenty_train.data, twenty_train.target)
prediction = text_clf.predict(twenty_test.data)

# EVALUATION
accurate = np.mean(prediction == twenty_test.target)

report = classification_report(twenty_test.target, prediction, target_names=twenty_test.target_names)
print('Classification Report:\n', report)

conf_matrix = confusion_matrix(twenty_test.target, prediction)
print('Confusion Matrix:\n', conf_matrix)

