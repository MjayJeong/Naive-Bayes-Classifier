import csv
import re
import numpy as np
import matplotlib.pyplot as plt

def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        stopwords = file.read().splitlines()
    return stopwords

def preprocess_text(text, stopwords):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stopwords]
    return words

def calculate_frequency(words):
    word_freq = {}
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    return word_freq

def classify_review(review, positive_prior, negative_prior, positive_word_probs, negative_word_probs, vocabulary,
                    total_positive_reviews, total_negative_reviews, stopwords):
    words = preprocess_text(review, stopwords)
    positive_score = np.log(positive_prior)
    negative_score = np.log(negative_prior)

    for word in words:
        if word in vocabulary:
            positive_score += np.log(positive_word_probs.get(word, 1 / (total_positive_reviews + len(vocabulary))))
            negative_score += np.log(negative_word_probs.get(word, 1 / (total_negative_reviews + len(vocabulary))))

    return 1 if positive_score > negative_score else 0

def laplace_training(train_data, test_data, stopwords, train_size):
    train_subset = train_data[:int(len(train_data) * train_size)]

    words_list = []
    for review, _ in train_subset:
        words = preprocess_text(review, stopwords)
        words_list.extend(words)

    word_frequency = calculate_frequency(words_list)
    sorted_word = sorted(word_frequency.items(), key=lambda item: item[1], reverse=True)
    top_1000_words = [word for word, _ in sorted_word[:1000]]

    positive_reviews = []
    negative_reviews = []
    for review, stars in train_subset:
        words = preprocess_text(review, stopwords)
        if stars == 5:
            positive_reviews.extend([word for word in words if word in top_1000_words])
        else:
            negative_reviews.extend([word for word in words if word in top_1000_words])

    positive_freq = calculate_frequency(positive_reviews)
    negative_freq = calculate_frequency(negative_reviews)

    vocabulary = top_1000_words
    vocabulary_size = len(vocabulary)
    total_positive_words = len(positive_reviews)
    total_negative_words = len(negative_reviews)

    positive_word_prob = {word: (positive_freq.get(word, 0) + 1) / (total_positive_words + vocabulary_size) for word in vocabulary}
    negative_word_prob = {word: (negative_freq.get(word, 0) + 1) / (total_negative_words + vocabulary_size) for word in vocabulary}

    total_reviews = len(train_subset)
    positive_prob = sum(stars == 5 for _, stars in train_subset) / total_reviews
    negative_prob = 1 - positive_prob

    correct_predictions = 0
    for review, stars in test_data:
        prediction = classify_review(review, positive_prob, negative_prob, positive_word_prob, negative_word_prob, vocabulary, total_positive_words, total_negative_words, stopwords)
        is_positive = 1 if stars == 5 else 0
        if prediction == is_positive:
            correct_predictions += 1

    accuracy = correct_predictions / len(test_data)
    return accuracy


stopwords = load_stopwords('stopwords.txt')

with open('train.csv', mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    train_data = [(row['text'], int(row['stars'])) for row in reader]

with open('test.csv', mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    test_data = [(row['text'], int(row['stars'])) for row in reader]


train_sizes = [0.1, 0.3, 0.5, 0.7, 1.0]
accuracies = []

for size in train_sizes:
    accuracy = laplace_training(train_data, test_data, stopwords, size)
    accuracies.append(accuracy)

plt.plot(train_sizes, accuracies, marker='o')
plt.title('Learning Curve')
plt.xlabel('Training Set Size (Proportion)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

for size, accuracy in zip(train_sizes, accuracies):
    print(f"Training Size: {size*100}%, Accuracy: {accuracy:.4f}")