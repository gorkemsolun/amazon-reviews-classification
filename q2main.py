"""
Project Summary - Part 2: Amazon Reviews Classification

In this part of the project, the goal is to develop a classification model for Amazon reviews, categorizing them into three distinct classes: Negative, Neutral, and Positive reviews. 
The project follows the Bag-of-Words representation and focuses on applying the Multinomial Naive Bayes and Bernoulli Naive Bayes models.

Dataset:  
The dataset provided contains 3000 reviews, where each review is preprocessed into word occurrence counts. 
There are 2300 reviews for training and 700 reviews for validation. Each review is represented by a vector of word counts, 
where the j-th element indicates how often the j-th word from the vocabulary appears in the review. The labels (negative, neutral, positive) are in separate files. 
The task is to train a model using the training set and validate it on the given validation set.

Bag-of-Words Model:  
The dataset is represented in the Bag-of-Words format, where the presence of each word in the document is conditionally independent of its position. 
The Naive Bayes model assumes that the probability of a review belonging to a specific class is determined by the individual word frequencies, rather than their position.

Questions and Tasks:

1. Multinomial Naive Bayes Model (Coding):  
   - Implement a Multinomial Naive Bayes Model to classify reviews.
   - The model is based on word occurrence frequencies and assumes that words follow a multinomial distribution within each class.
   - You will compute the probability of each document being in a class using the following estimator:
     P(Y = y_k | D_i) ∝ P(Y = y_k) ∏ P(X_j | Y = y_k)^t_wj,i
     where P(X_j | Y = y_k) is the probability of the j-th word occurring given the class, and t_wj,i is the count of the j-th word in the i-th document.
   - Use logarithmic probabilities to avoid underflow.

2. Additive Smoothing with Dirichlet Prior:  
   - Extend the Multinomial Naive Bayes model to include additive smoothing (Dirichlet prior with α = 1) to deal with zero probabilities for unseen words. 
     This involves "hallucinating" that each word appears α times in the dataset, smoothing the parameter estimates.
   - Train and evaluate the performance after smoothing, and compare the accuracy and confusion matrix with the non-smoothed version.

3. Bernoulli Naive Bayes Model:  
   - Implement a Bernoulli Naive Bayes Model where features are binary (indicating the presence or absence of words, rather than their frequency).
   - The model uses the following formula to predict the class:
     y_i = arg max ( log P(Y = y_k) + log ( ∏ t_j P(X_j | Y = y_k) + (1 - t_j)(1 - P(X_j | Y = y_k)) ) )
   - Compare the results of this binary model with the frequency-based Multinomial Naive Bayes model.

4. Class Imbalance and Evaluation:  
   - Analyze the class distribution (percentages of negative, neutral, and positive reviews) in the training and validation datasets. 
     Determine if the data is balanced or skewed, and discuss how class imbalance may affect the model’s performance.
   - Compute the number of occurrences of specific words (e.g., "good" and "bad") in positive reviews, and compute the log-ratio of their probabilities.

Evaluation:  
- The performance of both models (Multinomial and Bernoulli Naive Bayes) is evaluated using metrics like accuracy and the confusion matrix on the validation set.
- For each task, report the accuracy in three decimal points and discuss how smoothing (in the Multinomial Naive Bayes model) and using binary features (in the Bernoulli model) affect the results.

In summary, this part focuses on building and comparing two different Naive Bayes models (Multinomial and Bernoulli) for document classification, 
handling class imbalance, and implementing additive smoothing for better generalization.
"""

# Machine learning libraries will not be used in this part of the project. The focus is on implementing the models from scratch.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the training and validation data given in CSV format
# The labels are in separate files
# Instance numbers are not shown in the data
# NOTE: CHANGE THE FILE PATHS ACCORDING TO YOUR SYSTEM
train_data = pd.read_csv("dataset/x_train.csv")  # 2300 x 3000
train_label_data = pd.read_csv(
    "dataset/y_train.csv", header=None
)  # 2300 x 1 {0: Negative, 1: Neutral, 2: Positive}

test_data = pd.read_csv("dataset/x_test.csv")  # 700 x 3000
test_label_data = pd.read_csv(
    "dataset/y_test.csv", header=None
)  # 700 x 1 {0: Negative, 1: Neutral, 2: Positive}

# What are the percentages of each category in the y_train.csv y_test.csv? Draw a pie chart showing percentages.
# Count the number of occurrences of each class in the training and validation labels
train_label_counts = train_label_data[0].value_counts()
test_label_counts = test_label_data[0].value_counts()

# Calculate the percentages of each class in the training and validation labels
train_label_percentages = train_label_counts / train_label_counts.sum() * 100
test_label_percentages = test_label_counts / test_label_counts.sum() * 100

# Plot the pie chart showing the percentages of each class in the training and validation labels
""" fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].pie(
    train_label_percentages,
    labels=["Negative", "Neutral", "Positive"],
    autopct="%1.1f%%",
    startangle=90,
)
ax[0].set_title("Training Label Distribution")
ax[1].pie(
    test_label_percentages,
    labels=["Negative", "Neutral", "Positive"],
    autopct="%1.1f%%",
    startangle=90,
)
ax[1].set_title("Validation Label Distribution")
plt.show() """

# Train a Multinomial Naive Bayes model on the training set and evaluate your model on the test set given. Find and report the accuracy in three decimal points and
# report the confusion matrix for the test set.


# Implement a Multinomial Naive Bayes Model to classify reviews
class MultinomialNaiveBayes:
    def __init__(self, alpha=0):
        self.class_priors = None
        self.word_likelihoods = None
        self.alpha = alpha
        self.log_class_priors = None
        self.log_class_word_likelihoods = None
        print(f"Multinomial Naive Bayes Model with Alpha: {alpha}")

    def fit(self, X, y):
        """
        Fit the Multinomial Naive Bayes model to the training data.
        Parameters:
        - X: Training data (n_samples, n_features)
        - y: Training labels (n_samples,)
        """

        # Compute the class priors
        self.class_priors = np.bincount(y) / len(y)

        # Compute the word likelihoods
        self.word_likelihoods = np.zeros(
            (len(np.unique(y)), X.shape[1])
        )  # (n_classes, n_features)
        self.word_likelihoods += self.alpha  # Additive smoothing
        # Calculate the total word counts for each class
        for c in np.unique(y):
            # Get the word counts for the current class
            class_word_counts = X[y == c].sum(axis=0)

            # Add the class word counts to the likelihoods and apply additive smoothing
            self.word_likelihoods[c] += class_word_counts + self.alpha

            # Normalize the word likelihoods
            self.word_likelihoods[c] /= (
                self.word_likelihoods[c].sum()
                + self.alpha * X.shape[1]  # Additive smoothing
            )

        # Compute the class log probabilities
        # First, calculate the log probabilities for each class
        self.log_class_priors = np.log(self.class_priors)
        # Second, calculate the log probabilities for each word given the class
        self.log_class_word_likelihoods = np.log(self.word_likelihoods)

    def predict(self, X):
        """
        Predict the class labels for the input data.
        Parameters:
        - X: Input data (n_samples, n_features)
        Returns:
        - y_pred: Predicted class labels (n_samples,)
        """

        # Initialize the predicted class labels
        y_pred = np.zeros(X.shape[0])

        # Compute the log probabilities for each class
        log_probabilities = np.zeros((X.shape[0], len(self.class_priors)))
        for c in range(len(self.class_priors)):
            # Compute the log probabilities for each class
            log_probabilities[:, c] = self.log_class_priors[c] + (
                X @ self.log_class_word_likelihoods[c]
            )

        # Assign the class with the highest log probability to each sample
        y_pred = np.argmax(log_probabilities, axis=1)

        return y_pred


# Train a Multinomial Naive Bayes model on the training set
mnb = MultinomialNaiveBayes(alpha=0)

# Convert the data to numpy arrays
X_train = train_data.values
y_train = train_label_data.values.ravel()

# Fit the model on the training data
mnb.fit(X_train, y_train)

# Evaluate the model on the test set
X_test = test_data.values
y_test = test_label_data.values.ravel()

# Predict the labels for the test set
y_pred = mnb.predict(X_test)

# Calculate the accuracy of the model
accuracy = (y_pred == y_test).mean()
print(f"Accuracy of Multinomial Naive Bayes Model: {accuracy:.3f}")

# Calculate the confusion matrix
confusion_matrix = np.zeros((3, 3))
for i in range(len(y_test)):
    confusion_matrix[y_test[i], y_pred[i]] += 1

print("Confusion Matrix:")
print(confusion_matrix)

# Extend your classifier so that it can compute an estimate of alpha word using a fair Dirichlet prior. Set α = 1.
# Train and evaluate your model on the test set given. Find and report the accuracy in three decimal points and report the confusion matrix for the test set.

# Train a Multinomial Naive Bayes model with additive smoothing on the training set
mnb_smoothed = MultinomialNaiveBayes(alpha=1)

# Fit the model on the training data
mnb_smoothed.fit(X_train, y_train)

# Evaluate the smoothed model on the test set
# Predict the labels for the test set
y_pred_smoothed = mnb_smoothed.predict(X_test)

# Calculate the accuracy of the smoothed model
accuracy_smoothed = (y_pred_smoothed == y_test).mean()
print(
    f"Accuracy of Multinomial Naive Bayes Model with Additive Smoothing: {accuracy_smoothed:.3f}"
)

# Calculate the confusion matrix for the smoothed model
confusion_matrix_smoothed = np.zeros((3, 3))
for i in range(len(y_test)):
    confusion_matrix_smoothed[y_test[i], y_pred_smoothed[i]] += 1

print("Confusion Matrix with Additive Smoothing:")
print(confusion_matrix_smoothed)


# Implement a Bernoulli Naive Bayes Model to classify reviews
class BernoulliNaiveBayes:
    def __init__(self, alpha=0):
        self.class_priors = None
        self.word_likelihoods = None
        self.log_class_priors = None
        self.log_class_word_likelihoods = None
        self.alpha = alpha
        print(f"Bernoulli Naive Bayes Model with Alpha: {alpha}")

    def fit(self, X, y):
        """
        Fit the Bernoulli Naive Bayes model to the training data.
        Parameters:
        - X: Training data (n_samples, n_features)
        - y: Training labels (n_samples,)
        """

        # Compute the class priors
        bincount = np.bincount(y)
        self.class_priors = bincount / len(y)

        # Compute the word likelihoods
        self.word_likelihoods = np.zeros(
            (len(np.unique(y)), X.shape[1])
        )  # (n_classes, n_features)
        self.word_likelihoods += self.alpha  # Additive smoothing
        # Calculate the total word counts for each class
        for c in np.unique(y):
            # Get the word counts for the current class (binary features)
            class_word_counts = (X[y == c] > 0).sum(axis=0)

            # Add the class word counts to the likelihoods and apply additive smoothing
            self.word_likelihoods[c] += class_word_counts + self.alpha

            # Normalize the word likelihoods
            self.word_likelihoods[c] /= (
                bincount[c] + self.alpha * 2  # Additive smoothing for binary features
            )

        # Compute the class log probabilities
        # First, calculate the log probabilities for each class
        self.log_class_priors = np.log(self.class_priors)

        # Second, calculate the log probabilities for each word given the class but using binary features
        self.log_class_word_likelihoods = np.log(self.word_likelihoods)

        # Third, compute the log probabilities for the complement of each word given the class
        self.log_class_word_likelihoods_complement = np.log(1 - self.word_likelihoods)

    def predict(self, X):
        """
        Predict the class labels for the input data.
        Parameters:
        - X: Input data (n_samples, n_features)
        Returns:
        - y_pred: Predicted class labels (n_samples,)
        """

        # Initialize the predicted class labels
        y_pred = np.zeros(X.shape[0])

        # Convert the features to binary
        X = X > 0

        # Compute the log probabilities for each class
        log_probabilities = np.zeros((X.shape[0], len(self.class_priors)))
        for c in range(len(self.class_priors)):
            # Compute the log probabilities for each class using binary features
            log_probabilities[:, c] = self.log_class_priors[c] + (
                X @ self.log_class_word_likelihoods[c]
                + (1 - X) @ self.log_class_word_likelihoods_complement[c]
            )

        # Assign the class with the highest log probability to each sample
        y_pred = np.argmax(log_probabilities, axis=1)

        return y_pred


# Train a Bernoulli Naive Bayes model on the training set
bnb = BernoulliNaiveBayes(alpha=1)

# Fit the model on the training data
bnb.fit(X_train, y_train)

# Evaluate the model on the test set
# Predict the labels for the test set
y_pred_bnb = bnb.predict(X_test)

# Calculate the accuracy of the model
accuracy_bnb = (y_pred_bnb == y_test).mean()
print(f"Accuracy of Bernoulli Naive Bayes Model: {accuracy_bnb:.3f}")

# Calculate the confusion matrix
confusion_matrix_bnb = np.zeros((3, 3))
for i in range(len(y_test)):
    confusion_matrix_bnb[y_test[i], y_pred_bnb[i]] += 1

print("Confusion Matrix for Bernoulli Naive Bayes Model:")
print(confusion_matrix_bnb)
