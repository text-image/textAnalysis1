from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
import numpy as np
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


def fit_in_batches(classifier, x_train, y_train, batch_size):
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]
        classifier.partial_fit(x_batch, y_batch, classes=np.unique(y_train))
    return classifier


def predict_in_batches(classifier, x_test, batch_size):
    predictions = []
    for i in range(0, len(x_test), batch_size):
        x_batch = x_test[i:i + batch_size]
        predictions.extend(classifier.predict(x_batch))
    return np.array(predictions)


# CountVectorizer
vectorizer = CountVectorizer()
# model learns the principal components based on the training data distribution
twenty_train_count_float64 = vectorizer.fit_transform(twenty_train.data)
# transform is used on the test data to apply the same transformation learned from the training data
twenty_test_count_float64 = vectorizer.transform(twenty_test.data)

# Convert to float32
twenty_train_count = twenty_train_count_float64.astype('float32')
twenty_test_count = twenty_test_count_float64.astype('float32')

# train classifier
classifier_count = SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3, random_state=42, max_iter=5, tol=None)
classifier_count.fit(twenty_train_count, twenty_train.target)
# The trained classifier to make predictions on the transformed test data
prediction_count = classifier_count.predict(twenty_test_count)

# TfidfVectorizer
transformer = TfidfTransformer()
twenty_train_idf = transformer.fit_transform(twenty_train_count)
twenty_test_idf = transformer.transform(twenty_test_count)

classifier_idf = SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3, random_state=42, max_iter=5, tol=None)
classifier_idf.fit(twenty_train_idf, twenty_train.target)
prediction_idf = classifier_idf.predict(twenty_test_idf)

# Incremental PCA
# batch_size to define the number of samples processed at a time
ipca = IncrementalPCA(n_components=5, batch_size=1000)

twenty_train_count_pca = ipca.fit_transform(twenty_train_count.toarray())
twenty_test_count_pca = ipca.transform(twenty_test_count.toarray())
classifier_ipca_count = SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3, random_state=42, max_iter=5, tol=None)
classifier_ipca_count.fit(twenty_train_count_pca, twenty_train.target)
prediction_count_pca = classifier_ipca_count.predict(twenty_test_count_pca)

twenty_train_idf_pca = ipca.fit_transform(twenty_train_idf)
twenty_test_idf_pca = ipca.transform(twenty_test_idf)
classifier_ipca_idf = SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3, random_state=42, max_iter=5, tol=None)
classifier_ipca_idf.fit(twenty_train_idf_pca, twenty_train.target)
prediction_idf_pca = classifier_ipca_idf.predict(twenty_test_idf_pca)

# LDA
lda = LDA(n_components=2)

twenty_train_count_lda = lda.fit_transform(twenty_train_count.toarray(), twenty_train.target)
twenty_test_count_lda = lda.transform(twenty_test_count.toarray())
classifier_lda_count = SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3, random_state=42, max_iter=5, tol=None)
classifier_lda_count.fit(twenty_train_count_lda, twenty_train.target)
prediction_count_lda = classifier_lda_count.predict(twenty_test_count_lda)

twenty_train_idf_lda = lda.fit_transform(twenty_train_idf, twenty_train.target)
twenty_test_idf_lda = lda.transform(twenty_test_idf)
classifier_lda_idf = SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3, random_state=42, max_iter=5, tol=None)
classifier_lda_idf.fit(twenty_train_idf_lda, twenty_train.target)
prediction_idf_lda = classifier_lda_idf.predict(twenty_test_idf_lda)

print(f"Prediction of CountVectorizer: {prediction_count}")
print(f"Prediction of Tfidf: {prediction_idf}")
print(f"Prediction of CountVectorizer compressed by PCA: {prediction_count_pca}")
print(f"Prediction of CountVectorizer compressed by LDA: {prediction_count_lda}")
print(f"Prediction of Tfidf compressed by PCA: {prediction_idf_pca}")
print(f"Prediction of Tfidf compressed by LDA: {prediction_idf_lda}")

# EVALUATION
accurate_count = accuracy_score(twenty_test.target, prediction_count)
accurate_idf = accuracy_score(twenty_test.target, prediction_idf)
accurate_count_pca = accuracy_score(twenty_test.target, prediction_count_pca)
accurate_count_lda = accuracy_score(twenty_test.target, prediction_count_lda)
accurate_idf_pca = accuracy_score(twenty_test.target, prediction_idf_pca)
accurate_idf_lda = accuracy_score(twenty_test.target, prediction_idf_lda)

print(f"Accuracy score with CountVectorizer: {accurate_count: .4f}")
print(f"Accuracy score with TfidfTransformer: {accurate_idf: .4f}")
print(f"Accuracy score with CountVectorizer compressed by PCA: {accurate_count_pca: .4f}")
print(f"Accuracy score with CountVectorizer compressed by LDA: {accurate_count_lda: .4f}")
print(f"Accuracy score with compressed by PCA: {accurate_idf_pca: .4f}")
print(f"Accuracy score with compressed by LDA: {accurate_idf_lda: .4f}")

'''
# PAIRED T-TEST


def safe_t_test_rel(data1, data2):
    try:
        t_stat, p_val = ttest_rel(data1, data2, nan_policy="omit")
        return t_stat, p_val
    except ValueError as e:
        print(f"Error performing t-test: {e}")
'''
predictions = [prediction_count, prediction_idf, prediction_count_pca, prediction_count_lda, prediction_idf_pca,
               prediction_idf_lda]
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
