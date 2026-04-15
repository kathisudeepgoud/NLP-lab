#20. Implement a text clustering application?
# Step 1: Import required libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Step 2: Download resources
nltk.download('punkt')
nltk.download('stopwords')

# Step 3: Sample documents
documents = [
    "I love playing football",
    "Football is a great sport",
    "Python is used for machine learning",
    "Machine learning and AI are related fields",
    "I enjoy watching movies",
    "Movies are a great source of entertainment"
]
# Step 4: Text preprocessing function
def preprocess(text):
    tokens = word_tokenize(text.lower())   # convert to lowercase & tokenize
    tokens = [word for word in tokens if word.isalnum()]  # remove punctuation
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # remove stopwords
    return " ".join(tokens)

# Step 5: Apply preprocessing
processed_docs = [preprocess(doc) for doc in documents]

# Step 6: Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_docs)

# Step 7: Apply K-Means clustering
k = 2  # number of clusters
model = KMeans(n_clusters=k, random_state=42)
model.fit(X)

# Step 8: Print cluster labels
for i, doc in enumerate(documents):
    print(f"Document: {doc}")
    print(f"Cluster: {model.labels_[i]}")
    print()
