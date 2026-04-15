# 4. Write a program to slit sentences in a document?
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
text = "NLP is fun. It is used in Artificial Intelligence. NLP helps computers understand human language."
sentences = sent_tokenize(text)
print("Sentences in the document:")
for s in sentences:
    print(s)
