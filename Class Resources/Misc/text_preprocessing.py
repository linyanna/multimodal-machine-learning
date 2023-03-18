
"""
Common preprocessing steps for text data
"""

raw_text = "This is a sample text. It's not very long! It's not very short either. I'd say it's just right! :)"

# Step 1: sentence segmentation
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
corpus = sent_tokenize(raw_text)
print("\n Result for step 1 - sentence segmentation: \n", corpus)

# Step 2: lowercasing
print("\n Result for step 2 - lowercasing: \n")
step_2_corpus = []
for sentence in corpus:
    print(sentence.lower())
    step_2_corpus.append(sentence.lower())

# Step 3: remove digital numbers
import re
step_3_corpus = []
print("\n Result for step 3 - remove digital numbers: \n")
for sentence in step_2_corpus:
    print(sentence)
    sentence = re.sub('[0-9]', '', sentence)
    print(sentence)
    step_3_corpus.append(sentence)

# Step 4: decontraction
step_4_corpus = []
print("\n Result for step 4 - decontraction: \n")
for sentence in step_3_corpus:
    print(sentence)
    sentence = re.sub(r"won\'t", "will not", sentence)
    sentence = re.sub(r"can\'t", "can not", sentence)
    sentence = re.sub(r"n\'t", " not", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'s", " is", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'t", " not", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'m", " am", sentence)
    print(sentence)
    step_4_corpus.append(sentence)

# Step 5: remove punctuation
step_5_corpus = []
print("\n Result for step 5 - remove punctuation: \n")
for sentence in step_4_corpus:
    print(sentence)
    sentence = re.sub(r'[^a-z0-9<>]', ' ', sentence)
    # sentence = re.sub(r'[^\w\s]', '', sentence)
    print(sentence)
    step_5_corpus.append(sentence)

# Step 6: remove stop words
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

step_6_corpus = []
print("\n Result for step 6 - remove stop words: \n")
for sentence in step_5_corpus:
    print(sentence)
    sentence = [word for word in sentence.split() if word not in stop_words]
    print(sentence)
    step_6_corpus.append(sentence)

# Step 7: stemming
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

step_7_corpus = []
print("\n Result for step 7 - stemming: \n")
for sentence in step_6_corpus:
    print(sentence)
    temp_sentence = []
    for word in sentence:
        stemmed_word = stemmer.stem(word)
        temp_sentence.append(stemmed_word)
    print(temp_sentence)
    step_7_corpus.append(temp_sentence)

# Final result
print("\n Final result after text preprocessing: \n")
for sentence in step_7_corpus:
    print(" ".join(word for word in sentence))

print("\n Raw text: \n")
for sentence in corpus:
    print(sentence)


