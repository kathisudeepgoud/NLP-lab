#19. Implement a text classification application?

# Step 1: Import required libraries
import nltk
import random
from nltk.corpus import movie_reviews

# Step 2: Download dataset (run once)
nltk.download('movie_reviews')

# Step 3: Prepare documents
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Step 4: Shuffle documents
random.shuffle(documents)

# Step 5: Extract features
def extract_features(words):
    return {word: True for word in words}

# Step 6: Create feature sets
featuresets = [(extract_features(d), c) for (d, c) in documents]

# Step 7: Split into training and testing data
train_set = featuresets[:1500]
test_set = featuresets[1500:]

# Step 8: Train classifier (Naive Bayes)
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Step 9: Evaluate accuracy
accuracy = nltk.classify.accuracy(classifier, test_set)
print("Accuracy:", accuracy)

# Step 10: Test with custom input
sentence = "This movie is amazing and wonderful"
words = sentence.split()
features = extract_features(words)

prediction = classifier.classify(features)
print("Prediction:", prediction)

# Step 11: Show most informative features
classifier.show_most_informative_features(5)
