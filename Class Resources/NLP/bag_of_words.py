import nltk
import numpy

dataset = [
    'good movie',
    'not a good movie',
    'did not like',
]

word2count = {}
for data in dataset:
    words = nltk.word_tokenize(data)
    for word in words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1

unique_words = list(word2count.keys())
bag_of_words = []
for data in dataset:
    words = nltk.word_tokenize(data)
    bag_vector = numpy.zeros(len(unique_words))
    for w in words:
        for i, word in enumerate(unique_words):
            if word == w:
                bag_vector[i] += 1
    bag_of_words.append(bag_vector)

print(bag_of_words)