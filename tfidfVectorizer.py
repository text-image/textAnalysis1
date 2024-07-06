from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_rel

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

# Prevent division by zero
epsilon = 1e-10
prediction_idf = text_clf_idf.predict(twenty_test.data)
prediction_count = text_clf_count.predict(twenty_test.data)

print(f"Prediction of Tfidf: {prediction_idf}")
print(f"Prediction of CountVectorizer: {prediction_count}")


def safe_accuracy_score(y_true, y_predict):
    try:
        return accuracy_score(y_true, y_predict)
    except ZeroDivisionError:
        return 0.0


# EVALUATION
accurate_idf = accuracy_score(twenty_test.target, prediction_idf)
accurate_count = accuracy_score(twenty_test.target, prediction_count)

print(f"Accuracy score with TfidfTransformer: {accurate_idf: .4f}")
print(f"Accuracy score with CountVectorizer: {accurate_count: .4f}")

# PAIRED T-TEST


def safe_t_test_rel(data1, data2):
    try:
        t_stat, p_val = ttest_rel(data1, data2, nan_policy="omit")
        return t_stat, p_val
    except ValueError as e:
        print(f"Error performing t-test: {e}")


t_statistic, p_value = safe_t_test_rel(prediction_idf, prediction_count)
print(f"t-stat: {t_statistic: .4f}\np-val: {p_value: .4f}")
# INTERPRETATION
alpha_value = 0.05
if p_value < alpha_value:
    print("Models are significantly different", end=" ")
    if accurate_count < accurate_idf:
        print("and the model with TfidfTransformer is better.")
    else:
        print("and the model with only CountVectorizer is better.")
else:
    print("There is no significant difference.")
