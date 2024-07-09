import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import MultiComparison

# Load data
twenty_train = load_files(container_path='C:\\20news-bydate-train', encoding='latin-1')
twenty_test = load_files(container_path='C:\\20news-bydate-test', encoding='latin-1')

# CountVectorizer
vectorizer = CountVectorizer()
# model learns the principal components based on the training data distribution
twenty_train_count_float64 = vectorizer.fit_transform(twenty_train.data)
# transform is used on the test data to apply the same transformation learned from the training data
twenty_test_count_float64 = vectorizer.transform(twenty_test.data)
# Convert to float32
X_train_count = twenty_train_count_float64.astype('float32').toarray()
X_test_count = twenty_test_count_float64.astype('float32').toarray()

# TfidfVectorizer
transformer = TfidfTransformer()
X_train_idf = transformer.fit_transform(X_train_count).astype('float32').toarray()
X_test_idf = transformer.transform(X_test_count).astype('float32').toarray()

# Incremental PCA
ipca = IncrementalPCA(n_components=50, batch_size=1000)
X_train_count_pca = ipca.fit_transform(X_train_count)
X_test_count_pca = ipca.transform(X_test_count)
X_train_idf_pca = ipca.fit_transform(X_train_idf)
X_test_idf_pca = ipca.transform(X_test_idf)

# LDA
lda = LDA(n_components=2)
X_train_count_lda = lda.fit_transform(X_train_count, twenty_train.target)
X_test_count_lda = lda.transform(X_test_count)
X_train_idf_lda = lda.fit_transform(X_train_idf, twenty_train.target)
X_test_idf_lda = lda.transform(X_test_idf)

# Define classifiers
classifiers = [
    SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3, random_state=42, max_iter=5, tol=None),
    SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3, random_state=42, max_iter=5, tol=None),
]


# Fit classifiers in TensorFlow with GPU support
def fit_classifier_in_batches_tf(classifier, X_train, y_train, batch_size):
    # Convert to TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)

    # Compile model
    classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Fit model
    classifier.fit(dataset, epochs=5)  # Adjust epochs as needed
    return classifier


# Predict in batches
def predict_in_batches_tf(classifier, X_test, batch_size):
    # Convert to TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices(X_test).batch(batch_size)

    # Predict
    predictions = classifier.predict(dataset)
    return np.argmax(predictions, axis=1)


# Convert classifiers to TensorFlow models
classifiers_tf = [tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=lambda: model) for model in classifiers]

# Fit and predict for each dataset
predictions_tf = []
for i, classifier in enumerate(classifiers_tf):
    # Fit classifier
    classifier = fit_classifier_in_batches_tf(classifier, X_train_count, twenty_train.target, batch_size=500)

    # Predict
    prediction_count = predict_in_batches_tf(classifier, X_test_count, batch_size=500)
    predictions_tf.append(prediction_count)

    # Optionally, fit and predict with TF-IDF vectors as well
    classifier = fit_classifier_in_batches_tf(classifier, X_train_idf, twenty_train.target, batch_size=500)
    prediction_idf = predict_in_batches_tf(classifier, X_test_idf, batch_size=500)
    predictions_tf.append(prediction_idf)

# Process predictions
for i, prediction in enumerate(predictions_tf):
    print(f"Prediction of Classifier {i + 1}: {prediction}")

# Evaluate accuracy
accuracies_tf = []
for i, prediction in enumerate(predictions_tf):
    accuracy = accuracy_score(twenty_test.target, prediction)
    accuracies_tf.append(accuracy)
    print(f"Accuracy score of Classifier {i + 1}: {accuracy:.4f}")

# ANOVA analysis
data = []
for i, preds in enumerate(predictions_tf):
    for sample_idx, pred in enumerate(preds):
        data.append([sample_idx, pred, f"Model_{i + 1}"])

df = pd.DataFrame(data, columns=["Sample", "Prediction", "Model"])

f_stats, p_value = f_oneway(*(df[df["Model"] == f"Model_{i + 1}"]["Prediction"] for i in range(len(predictions_tf))))
print(f"ANOVA F-statistic: {f_stats:.4f}, p-value: {p_value:.4f}")

# Interpretation of ANOVA results
if p_value < 0.05:
    mc = MultiComparison(df["Prediction"], df["Model"])
    results = mc.tukeyhsd()
    print(results)
else:
    print("No significant differences between models.")
