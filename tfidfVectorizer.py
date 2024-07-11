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


# CountVectorizer
count_vectorizer = CountVectorizer()
twenty_news_count_float64 = count_vectorizer.fit_transform(twenty_news.data)
twenty_news_count = twenty_news_count_float64.astype('float32').toarray()
# TfidfVectorizer
idf_vectorizer = TfidfVectorizer()
twenty_news_idf = idf_vectorizer.fit_transform(twenty_news.data).astype('float32').toarray()
# Incremental PCA
# batch_size to define the number of samples processed at a time
ipca = IncrementalPCA(n_components=50, batch_size=1000)
twenty_news_count_pca = ipca.fit_transform(twenty_news_count)
twenty_news_idf_pca = ipca.fit_transform(twenty_news_idf)
# LDA
lda = LDA(n_components=2)
twenty_news_count_lda = lda.fit_transform(twenty_news_count, twenty_news.target)
twenty_news_idf_lda = lda.fit_transform(twenty_news_idf, twenty_news.target)

to_split = [twenty_news_count, twenty_news_idf, twenty_news_count_pca, twenty_news_idf_pca, twenty_news_count_lda,
            twenty_news_idf_lda]
twenty_news_trains = []
twenty_news_tests = []
target_trains = []
target_tests = []
for i in to_split:
    twenty_train, twenty_test, twenty_train_target, twenty_test_target = train_test_split(i, twenty_news.target,
                                                                                          test_size=0.2,
                                                                                          random_state=42)
    twenty_news_train_target = np.array(twenty_train_target)
    twenty_news_test_target = np.array(twenty_test_target)
    twenty_news_trains.append(twenty_train)
    twenty_news_tests.append(twenty_test)
    target_trains.append(twenty_train_target)
    target_tests.append(twenty_test_target)

classifiers = [SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3, random_state=42, max_iter=5, tol=None) for i in
               range(6)]
batch_size = 500
# TRAIN THE CLASSIFIER
fitted = []
for i, classifier in enumerate(classifiers):
    fitted_classifier = fit_in_batches(classifier, twenty_news_trains[i], target_trains[i], batch_size)
    fitted.append(fitted_classifier)
# PREDICT
predictions = []
for i, classifier in enumerate(fitted):
    prediction = predict_in_batches(classifier, twenty_news_tests[i], batch_size)
    predictions.append(prediction)
for i, prediction in enumerate(predictions):
    print(f"Prediction of Classifier {i+1}: {prediction}")
# EVALUATE
accuracies = []
for i, prediction in enumerate(predictions):
    accuracy = accuracy_score(target_tests[i], prediction)
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
