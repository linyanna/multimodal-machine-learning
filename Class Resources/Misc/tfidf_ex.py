from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    'good movie',
    'not a good movie',
    'did not like',
]

tf_idf_model = TfidfVectorizer()
tf_idf_vector = tf_idf_model.fit_transform(corpus)

tf_idf_array = tf_idf_vector.toarray()

print(tf_idf_array)