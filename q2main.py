# Görkem Kadir Solun 22003214

# Machine learning libraries will not be used in this part of the project. The focus is on implementing the models from scratch.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the training and test data given in CSV format
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
# Count the number of occurrences of each class in the training and test labels
train_label_counts = train_label_data[0].value_counts()
test_label_counts = test_label_data[0].value_counts()

# Calculate the percentages of each class in the training and test labels
train_label_percentages = train_label_counts / train_label_counts.sum() * 100
test_label_percentages = test_label_counts / test_label_counts.sum() * 100

# Plot the pie chart showing the percentages of each class in the training and test labels
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].pie(
    train_label_percentages,
    labels=["Positive", "Neutral", "Negative"],
    autopct="%1.1f%%",
    startangle=90,
)
ax[0].set_title("Training Label Distribution")
ax[1].pie(
    test_label_percentages,
    labels=["Positive", "Neutral", "Negative"],
    autopct="%1.1f%%",
    startangle=90,
)
ax[1].set_title("Test Label Distribution")
plt.show()

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

        # Calculate the total word counts for each class
        for c in np.unique(y):
            # Get the word counts for the current class
            class_word_counts = X[y == c].sum(axis=0)

            # Add the class word counts to the likelihoods and apply additive smoothing
            self.word_likelihoods[c] = class_word_counts + self.alpha

            # Normalize the word likelihoods
            self.word_likelihoods[c] /= (
                class_word_counts.sum() + self.alpha * X.shape[1]  # Additive smoothing
            )

        """ for i in range(len(self.word_likelihoods)): # If you want to get 0.583 accuracy, you can use this code block
            for j in range(len(self.word_likelihoods[i])):
                if self.word_likelihoods[i][j] == 0:
                    self.word_likelihoods[i][j] = 1e-12 """

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

        # Calculate the total word counts for each class
        for c in np.unique(y):
            # Get the word counts for the current class (binary features)
            class_word_counts = (X[y == c] > 0).sum(axis=0)

            # Add the class word counts to the likelihoods and apply additive smoothing
            self.word_likelihoods[c] = class_word_counts + self.alpha

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
