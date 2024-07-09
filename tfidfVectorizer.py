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

prediction_idf = text_clf_idf.predict(twenty_test.data)
prediction_count = text_clf_count.predict(twenty_test.data)
idf_pca = text_clf_idf.predict(twenty_pca)
idf_lda = text_clf_idf.predict(twenty_lda)
count_pca = text_clf_count.predict(twenty_pca)
count_lda = text_clf_count.predict(twenty_lda)

print(f"Prediction of Tfidf: {prediction_idf}")
print(f"Prediction of CountVectorizer: {prediction_count}")
print(f"Prediction of Tfidf compressed by PCA: {idf_pca}")
print(f"Prediction of Tfidf compressed by LDA: {idf_lda}")
print(f"Prediction of CountVectorizer compressed by PCA: {count_pca}")
print(f"Prediction of CountVectorizer compressed by LDA: {count_lda}")

# EVALUATION
accurate_idf = accuracy_score(twenty_test.target, prediction_idf)
accurate_count = accuracy_score(twenty_test.target, prediction_count)
accurate_idf_pca = accuracy_score(twenty_test.target, idf_pca)
accurate_idf_lda = accuracy_score(twenty_test.target, idf_lda)
accurate_count_pca = accuracy_score(twenty_test.target, count_pca)
accurate_count_lda = accuracy_score(twenty_test.target, count_lda)

print(f"Accuracy score with TfidfTransformer: {accurate_idf: .4f}")
print(f"Accuracy score with CountVectorizer: {accurate_count: .4f}")
print(f"Accuracy score with compressed by PCA: {accurate_idf_pca: .4f}")
print(f"Accuracy score with compressed by LDA: {accurate_idf_lda: .4f}")
print(f"Accuracy score with CountVectorizer compressed by PCA: {accurate_count_pca: .4f}")
print(f"Accuracy score with CountVectorizer compressed by LDA: {accurate_count_lda: .4f}")
'''
# PAIRED T-TEST


def safe_t_test_rel(data1, data2):
    try:
        t_stat, p_val = ttest_rel(data1, data2, nan_policy="omit")
        return t_stat, p_val
    except ValueError as e:
        print(f"Error performing t-test: {e}")
'''
predictions = [prediction_idf, prediction_count, idf_pca, idf_lda, count_pca, count_lda]
data = []
for i, j in enumerate(predictions):
    for i_text, predict in enumerate(j):
        data.append([i_text, predict, f"Model_{i+1}"])
        print(data)
# data = [[0, 17, 'Model_1'], [1, 8, 'Model_1'], ...], i=model index, j= predictions of a model

# DATA FRAME
df = pd.DataFrame(data, columns=["Sample", "Prediction", "Model"])

# ANOVA
# unpack list=> f_oneway(*(groups)) = f_oneway(group1, group2, ...)
f_stats, p_value = f_oneway(*(df[df["Model"] == f"Model_{i+1}"]["Prediction"] for i in range(len(predictions))))
print(f"ANOVA F-statistic: {f_stats:.4f}, p-value: {p_value:.4f}")

# INTERPRETATION
if p_value < 0.05:
    mc = MultiComparison(df["Prediction"], df["Model"])
    results = mc.tukeyhsd()
    print(results)
else:
    print("No significant differences between models.")
