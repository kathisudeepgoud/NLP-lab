import nltk 
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.probability import FreqDist 
nltk.download('punkt')   
nltk.download('punkt_tab')                 
text = "NLP is fun. It is used in AI. NLP helps computers understand language." 
print("Sentences:", sent_tokenize(text))           
words = [w.lower() for w in word_tokenize(text) if w.isalnum()] 
freq = FreqDist(words)                     
print(freq) 
