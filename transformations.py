from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

# LOAD DATA
container_path = 'C:\\20news-bydate-train'
categories = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]
twenty_train = load_files(container_path=container_path, description=None, categories=categories, load_content=True,
                          shuffle=True, encoding='latin-1', decode_error='strict', random_state=42,
                          allowed_extensions=None)

# By setting the correct encoding, you ensure that the text data is correctly interpreted and read into Python as
# strings. This is especially important for non-ASCII characters.

# returns bunch: dictionary-like obj with data(list of str), target(ndarray), target_names(list), full description,
# filenames(ndarray)


# EXTRACT FEATURES
# CountVectorizer builds a dictionary of features and transforms documents to feature vectors
count_vector = CountVectorizer()

# Get document-term sparse matrix where rows correspond to documents and columns correspond to terms (words).
x_count_of_train = count_vector.fit_transform(twenty_train.data)

# Term Frequency and Term Frequency times Inverse Document Frequency
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_count_of_train)

# TRAINING A CLASSIFIER
# naive Bayes classifier, multinomial variant
clf = MultinomialNB()
clf.fit(x_train_tfidf, twenty_train.target)

new_doc = ["Atheists do not have a God", "Water boils at 100 degrees celcius"]
# We use only transform since they have already been fit to the training set
new_doc_train_counts = count_vector.transform(new_doc)
new_doc_tfidf = tfidf_transformer.transform(new_doc_train_counts)
# Make predictions
prediction = clf.predict(new_doc_tfidf)

# %r, raw data, is replaced by repr(doc), which shows the document with quotes.
# %s is replaced by str(twenty_train.target_names[category]), which shows the category name as a regular string.
for doc, category in zip(new_doc, prediction):
    print('%r => %s' % (doc, twenty_train.target_names[category]))