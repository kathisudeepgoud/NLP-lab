# 6. Remove the stopwords and rarewords in the document?

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
nltk.download('punkt')
nltk.download('stopwords')
text = "NLP is a branch of Artificial Intelligence. NLP helps computers understand human language."
tokens = word_tokenize(text.lower())
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in tokens if word.isalnum() and word not in stop_words]
print("After Stopword Removal:", filtered_words)
freq = FreqDist(filtered_words)
final_words = [word for word in filtered_words if freq[word] > 1]
print("After Removing Rare Words:", final_words)
