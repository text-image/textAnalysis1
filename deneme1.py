from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import MultiComparison
import time
import pandas as pd
import numpy as np
import openpyxl

categories = ["alt.atheism", "sci.med", "talk.politics.mideast", "rec.motorcycles"]
twenty_news = load_files(container_path='C:\\20_newsgroups', categories=categories, description=None, load_content=True,
                         shuffle=True, encoding='latin-1', decode_error='strict', random_state=42,
                         allowed_extensions=None)


def fit_in_batches(classifier, x_train, y_train, batch_size):
    if hasattr(classifier, 'partial_fit'):
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            classifier.partial_fit(x_batch, y_batch, classes=np.unique(y_train))
    else:
        classifier.fit(x_train, y_train)
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

    data_train, data_test, data_train_target, data_test_target = train_test_split(data_vector, data.target,
                                                                                  test_size=0.2, random_state=42)

    data_train_target = np.array(data_train_target)
    data_test_target = np.array(data_test_target)

    fitted_classifier = fit_in_batches(classifier, data_train, data_train_target, batch_size=500)

    return data_train, data_test, data_train_target, data_test_target, fitted_classifier


# Vectorizer and compressor combinations
vectorizer = [CountVectorizer(), TfidfVectorizer()]
compressor = [None, IncrementalPCA(n_components=50, batch_size=1000), LDA(n_components=2)]
classifier = [SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3, random_state=42, max_iter=5, tol=None),
              LogisticRegression(max_iter=1000),
              SVC(kernel='linear', random_state=42)]
combinations = [(vec, comp, classify) for vec in vectorizer for comp in compressor for classify in classifier]

# Re-initialize classifier for each combination
all_results = []
model_no = 1
for vec, comp, classify in combinations:
    start_time = time.time()
    if isinstance(classify, SGDClassifier):
        classifier_instance = SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3, random_state=42, max_iter=5,
                                            tol=None)
    elif isinstance(classify, LogisticRegression):
        classifier_instance = LogisticRegression(max_iter=1000)
    elif isinstance(classify, SVC):
        classifier_instance = SVC(kernel='linear', random_state=42)

    result = pre_steps(twenty_news, vec, comp, classifier_instance)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time of model {model_no}: {execution_time: .6f} seconds")
    model_no += 1
    all_results.append(result)

# PREDICT AND EVALUATE
predictions = []
accuracies = []
for i, j in enumerate(all_results):
    data_train, data_test, data_train_target, data_test_target, fitted_classifier = j
    prediction = predict_in_batches(fitted_classifier, data_test, batch_size=500)
    predictions.append(prediction)

    accuracy = accuracy_score(data_test_target, prediction)
    accuracies.append(accuracy)
    print(f"Accuracy score of Model {i+1}: {accuracy:.4f}:")
# for i, prediction in enumerate(predictions):
#    print(f"Prediction of Classifier {i+1}: {prediction}")

data = []
for i, j in enumerate(predictions):
    for i_text, predict in enumerate(j):
        data.append([i_text, predict, f"Model_{i+1}"])
# data = [[0, 17, 'Model_1'], [1, 8, 'Model_1'], ...], i=model index, j= predictions of a model

# DATA FRAME
df = pd.DataFrame(data, columns=["Sample", "Prediction", "Model"])

# ANOVA
# unpack list=> f_oneway(*(groups)) = f_oneway(group1, group2, ...)
# Check for unique values in each group before performing ANOVA
unique_values_check = [len(df[df["Model"] == f"Model_{i+1}"]["Prediction"].unique()) > 1 for i in
                       range(len(predictions))]
if all(unique_values_check):
    # Perform ANOVA
    f_stats, p_value = f_oneway(*(df[df["Model"] == f"Model_{i+1}"]["Prediction"] for i in range(len(predictions))))
    print(f"ANOVA F-statistic: {f_stats:.4f}, p-value: {p_value:.4f}")

    # Interpret results
    if p_value < 0.05:
        mc = MultiComparison(df["Prediction"], df["Model"])
        results = mc.tukeyhsd()
        print(results)
    else:
        print("No significant differences between models.")
else:
    print("ANOVA cannot be performed because one or more models have only one unique prediction.")

# EXCEL
workbook = openpyxl.Workbook()
sheet = workbook.active
model_names = []
for i in range(18):
    model_names.append(f"Model {i+1}")
model_names.extend(["f statistics", "p value"])
# Add data to the Excel sheet
data = [model_names, accuracies + [f_stats, p_value]]

for row in data:
    sheet.append(row)
workbook.save("sheet.xlsx")
print("Excel file is created.")
