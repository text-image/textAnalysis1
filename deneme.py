from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import MultiComparison

# LOAD DATA
# categories = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]

# returns bunch: dictionary-like obj with data(list of str), target(ndarray), target_names(list), full description,
# filenames(ndarray)
twenty_train = load_files(container_path='C:\\20news-bydate-train', description=None, load_content=True, shuffle=True,
                          encoding='latin-1', decode_error='strict', random_state=42, allowed_extensions=None)
twenty_test = load_files(container_path='C:\\20news-bydate-test', description=None, load_content=True, shuffle=True,
                         encoding='latin-1', decode_error='strict', random_state=42, allowed_extensions=None)

# BUILD A PIPELINE (make the vectorizer => transformer => classifier easier to work with)
# Each step in the pipeline is a tuple consisting of a name (string) and a transformer or estimator (object).
text_clf_idf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3, random_state=42, max_iter=5, tol=None))
])

text_clf_count = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3, random_state=42, max_iter=5, tol=None))
])

# CountVectorizer is fitted to twenty_train.data.
# TfidfTransformer is fitted to the output of CountVectorizer.
# MultinomialNB is fitted to the output of TfidfTransformer.
text_clf_idf.fit(twenty_train.data, twenty_train.target)
text_clf_count.fit(twenty_train.data, twenty_train.target)

# PCA and LDA
pca = PCA(n_components=2)
twenty_pca = pca.fit_transform(twenty_train.data)

lda = LDA(n_components=2)
twenty_lda = lda.fit_transform(twenty_train.data, twenty_train.target)