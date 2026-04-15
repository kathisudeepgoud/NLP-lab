#11. Implement Maximum Entropy Classifier? (Movie Reviews)

import nltk 
from nltk.corpus import movie_reviews 
nltk.download('movie_reviews') 
docs = [(list(movie_reviews.words(f)), c) 
for c in movie_reviews.categories() 
for f in movie_reviews.fileids(c)] 
features = lambda w: {word: True for word in w[:100]} 
train = [(features(w), c) for w, c in docs[:1500]] 
test = [(features(w), c) for w, c in docs[1500:]] 
clf = nltk.MaxentClassifier.train(train, max_iter=5) 
print("Accuracy:", nltk.classify.accuracy(clf, test))

#-------------------OR-------------------

# Step 1: Import required libraries
import nltk
import random
from nltk.classify import MaxentClassifier
from nltk.corpus import movie_reviews

# Step 2: Download dataset (run once)
nltk.download('movie_reviews')

# Step 3: Define feature extraction function
def extract_features(words):
    return {word: True for word in words}

# Step 4: Load dataset
documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        words = list(movie_reviews.words(fileid))
        documents.append((words, category))

# Step 5: Shuffle dataset
random.shuffle(documents)

# Step 6: Convert documents into feature sets
featuresets = [(extract_features(d), c) for (d, c) in documents]

# Step 7: Split into training and testing data
train_set = featuresets[:1500]
test_set = featuresets[1500:]

# Step 8: Train Maximum Entropy Classifier
classifier = MaxentClassifier.train(train_set, max_iter=10)

# Step 9: Evaluate the model
accuracy = nltk.classify.accuracy(classifier, test_set)
print("Accuracy:", accuracy)

# Step 10: Test with custom sentence
test_sentence = "This movie was fantastic and full of suspense"
test_words = test_sentence.split()
test_features = extract_features(test_words)

prediction = classifier.classify(test_features)
print("Prediction for custom sentence:", prediction)
