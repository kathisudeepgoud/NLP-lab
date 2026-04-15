# 5. Perform tokenizing and stemming by reading the input string?
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('punkt')
text = input("Enter a sentence: ")
tokens = word_tokenize(text)
print("Tokens:", tokens)
ps = PorterStemmer()
stems = [ps.stem(word) for word in tokens]
print("Stemmed Words:", stems)
