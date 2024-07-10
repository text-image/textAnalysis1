from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import MultiComparison
import openpyxl

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
count_vectorizer = CountVectorizer()
# model learns the principal components based on the training data distribution
twenty_train_count_float64 = count_vectorizer.fit_transform(twenty_train.data)
# transform is used on the test data to apply the same transformation learned from the training data
twenty_test_count_float64 = count_vectorizer.transform(twenty_test.data)
# Convert to float32
twenty_train_count = twenty_train_count_float64.astype('float32').toarray()
twenty_test_count = twenty_test_count_float64.astype('float32').toarray()

# TfidfVectorizer
idf_vectorizer = TfidfVectorizer()
twenty_train_idf = idf_vectorizer.fit_transform(twenty_train.data).astype('float32').toarray()
twenty_test_idf = idf_vectorizer.transform(twenty_test.data).astype('float32').toarray()

# Incremental PCA
# batch_size to define the number of samples processed at a time
ipca = IncrementalPCA(n_components=50, batch_size=1000)
twenty_train_count_pca = ipca.fit_transform(twenty_train_count)
twenty_test_count_pca = ipca.transform(twenty_test_count)
twenty_train_idf_pca = ipca.fit_transform(twenty_train_idf)
twenty_test_idf_pca = ipca.transform(twenty_test_idf)


# LDA
lda = LDA(n_components=2)
twenty_train_count_lda = lda.fit_transform(twenty_train_count, twenty_train.target)
twenty_test_count_lda = lda.transform(twenty_test_count)
twenty_train_idf_lda = lda.fit_transform(twenty_train_idf, twenty_train.target)
twenty_test_idf_lda = lda.transform(twenty_test_idf)

twenty_trains = [twenty_train_count, twenty_train_idf, twenty_train_count_pca, twenty_train_idf_pca,
                 twenty_train_count_lda, twenty_train_idf_lda]
twenty_tests = [twenty_test_count, twenty_test_idf, twenty_test_count_pca, twenty_test_idf_pca,
                twenty_test_count_lda, twenty_test_idf_lda]
target_train = np.array(twenty_train.target)

classifiers = [SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3, random_state=42, max_iter=5, tol=None)] * 6
batch_size = 500

# TRAIN THE CLASSIFIER
fitted = []
for i, classifier in enumerate(classifiers):
    fitted_classifier = fit_in_batches(classifier, twenty_trains[i], target_train, batch_size)
    fitted.append(fitted_classifier)

# PREDICT
predictions = []
for i, classifier in enumerate(fitted):
    prediction = predict_in_batches(classifier, twenty_tests[i], batch_size)
    predictions.append(prediction)

for i, prediction in enumerate(predictions):
    print(f"Prediction of Classifier {i+1}: {prediction}")

# EVALUATE
accuracies = []
for i, prediction in enumerate(predictions):
    accuracy = accuracy_score(twenty_test.targey, prediction)
    accuracies.append(accuracy)
    print(f"Accuracy score of Classifier {i+1}: {accuracy}: .4f")

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
