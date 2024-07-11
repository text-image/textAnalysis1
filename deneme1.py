from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import MultiComparison
import pandas as pd
import numpy as np
import openpyxl

categories = ["alt.atheism", "sci.med", "talk.politics.mideast", "rec.motorcycles"]
twenty_news = load_files(container_path='C:\\20_newsgroups', categories=categories, description=None, load_content=True,
                         shuffle=True, encoding='latin-1', decode_error='strict', random_state=42,
                         allowed_extensions=None)


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


def pre_steps(data, vectorizer, compressor, classifier):
    data_vector = vectorizer.fit_transform(data.data).astype('float32').toarray()
    if compressor is not None:
        if isinstance(compressor, LDA):
            data_vector = compressor.fit_transform(data_vector, data.target)
        else:
            data_vector = compressor.fit_transform(data_vector)

    data_train, data_test, data_train_target, data_test_target = train_test_split(data_vector, data.target, test_size=0.2,
                                                                                  random_state=42)
    data_train_target = np.array(data_train_target)
    data_test_target = np.array(data_test_target)

    fitted_classifier = fit_in_batches(classifier, data_train, data_train_target, batch_size=500)

    return data_train, data_test, data_train_target, data_test_target, fitted_classifier


# CountVectorizer
count_result = pre_steps(twenty_news, CountVectorizer(), None, SGDClassifier())
# TfidfVectorizer
tfidf_result = pre_steps(twenty_news, TfidfVectorizer(), None, SGDClassifier())
# Incremental PCA
# batch_size to define the number of samples processed at a time
count_pca_result = pre_steps(twenty_news, CountVectorizer(), IncrementalPCA(n_components=50, batch_size=1000),
                             SGDClassifier())
tfidf_pca_result = pre_steps(twenty_news, TfidfVectorizer(), IncrementalPCA(n_components=50, batch_size=1000),
                             SGDClassifier())
# LDA
count_lda_result = pre_steps(twenty_news, CountVectorizer(), LDA(n_components=2), SGDClassifier())
tfidf_lda_result = pre_steps(twenty_news, TfidfVectorizer(), LDA(n_components=2), SGDClassifier())

# PREDICT
all = [count_result, tfidf_result, count_pca_result, tfidf_pca_result, count_lda_result, count_lda_result]
predictions = []
for j in all:
    for i, classifier in enumerate(j[4]):
        prediction = predict_in_batches(classifier, j[1][i], batch_size=500)
        predictions.append(prediction)
for i, prediction in enumerate(predictions):
    print(f"Prediction of Classifier {i+1}: {prediction}")
# EVALUATE
accuracies = []
for j in all:
    for i, prediction in enumerate(predictions):
        accuracy = accuracy_score(all[3][i], prediction)
        accuracies.append(accuracy)
        print(f"Accuracy score of Classifier {i+1}: {accuracy:.4f}:")
data = []
for i, j in enumerate(predictions):
    for i_text, predict in enumerate(j):
        data.append([i_text, predict, f"Model_{i+1}"])
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

'''
# EXCEL
workbook = openpyxl.Workbook()
sheet = workbook.active
# Add data to the Excel sheet
data = [
    ["CountVectorizer", "TfidfTransformer", "CounterVec with PCA", "Tfidf with PCA", "CounterVec with LDA",
     "Tfidf with LDA", "f-statistic", "p-value"],
    accuracies
]
for row in data:
    sheet.append(row)
workbook.save("sheet.xlsx")
print("Excel file is created.")
'''
