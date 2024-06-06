from sklearn.datasets import load_files

container_path = 'C:\\20news-bydate-train'
categories = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]
twenty_train = load_files(container_path= container_path, description=None, categories=categories, load_content=True,
                  shuffle=True, encoding='latin-1', decode_error='strict', random_state=42, allowed_extensions=None)

# By setting the correct encoding, you ensure that the text data is correctly interpreted and read into Python as
# strings. This is especially important for non-ASCII characters.

# returns bunch: dictionary-like obj with data(list of str), target(ndarray), target_names(list), full description,
# filenames(ndarray)

print("\n".join(twenty_train.data[0].split("\n")[:3]))
arr = twenty_train.target[:10]
print(arr)
for t in arr:
    print(twenty_train.target_names[t])

# loads the target attribute as an array of integers that corresponds to the index of the category name in the
# target_names list. The category integer id of each sample is stored in the target attribute
# maps the category integer IDs to their corresponding category names