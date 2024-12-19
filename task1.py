import csv
import re

def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        stopwords = file.read().splitlines()
    return stopwords


def preprocess_word(text, stopwords):
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



stopwords = load_stopwords('stopwords.txt')

with open('train.csv', mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    train_data = [row['text'] for row in reader]


words_list = []
for review in train_data:
    words = preprocess_word(review, stopwords)
    words_list.extend(words)


word_frequency = calculate_frequency(words_list)
sorted_word = sorted(word_frequency.items(), key=lambda item: item[1], reverse=True)
top_1000_words = [word for word, _ in sorted_word[:1000]]


print("Top 20-50 words:")
for i, word in enumerate(top_1000_words[20:51], start=20):
    print(f"{i}. {word}")