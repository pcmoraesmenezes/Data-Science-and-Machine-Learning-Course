# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data       # Data is the features
y = iris.target     # Target is the labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# Create a Gaussian Naive Bayes classifier
nb_classifier = GaussianNB()

# Train the classifier on the training set
nb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb_classifier.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

"""
Output:

Accuracy: 1.00 - 100% accuracy
Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        11
           1       1.00      1.00      1.00         9
           2       1.00      1.00      1.00        10

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
    weighted avg       1.00      1.00      1.00        30

precision = TP / (TP + FP) TP = True Positive, FP = False Positive
precision is the ratio of correctly predicted positive observations
to the total predicted positive observations

recall = TP / (TP + FN) TP = True Positive, FN = False Negative
recall is the ratio of correctly predicted positive observations

f1-score = 2 * (precision * recall) / (precision + recall)
f1-score is the weighted average of precision and recall

support is the number of actual occurrences of the class in the specified
dataset

macro avg is the average precision, recall and f1-score between classes
weighted avg is the weighted average of precision, recall and f1-score
between classes


"""
