# Naive Bayes Classifier

This project implements a **Naive Bayes Classifier (NBC)** for sentiment analysis on a dataset containing reviews and their respective star ratings.<br>
The classifier is trained to identify positive and negative sentiments based on a simplified approach to text processing and classification.

---

## Project Overview

1. **Dataset**: 
   - `train.csv`: Training data for building the model.
   - `test.csv`: Testing data for evaluating the model's performance.
   - `stopwords.txt`: A list of words to exclude during preprocessing.

2. **Tasks**:
   - **Task 1**: Feature selection and preprocessing.
   - **Task 2**: Model training and evaluation using Laplace Smoothing.
   - **Task 3**: Learning curve analysis.

3. **Evaluation Metric**:
   - Model accuracy

---

## How to Execute

First of all, I saved the stopwords word in a txt file. It's the way that the txt file is fetched from the code.<br>
And the test.csv and train.csv files are fetched as well.<br>
The required library is numpy, matplotlib.
