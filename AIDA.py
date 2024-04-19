# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Sample email dataset (features: email content, labels: 0 for non-spam, 1 for spam)
emails = [
    ("Get free money now!", 1),
    ("Important meeting tomorrow", 0),
    ("Claim your prize today", 1),
    ("Reminder: Project deadline approaching", 0)
]

# Split data into features (X) and labels (y)
X, y = zip(*emails)

# Convert text data into numerical features using Bag-of-Words representation
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Evaluate the classifier
accuracy = classifier.score(X_test, y_test)
print("Accuracy:", accuracy)

# Example email to classify
new_email = ["Congratulations! You've won a million dollars!"]
new_email_vectorized = vectorizer.transform(new_email)

# Predict label for the new email
prediction = classifier.predict(new_email_vectorized)
if prediction[0] == 1:
    print("This email is classified as spam.")
else:
    print("This email is classified as non-spam.")
