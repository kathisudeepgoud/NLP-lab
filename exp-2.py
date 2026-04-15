import nltk, re, requests 
from nltk.probability import FreqDist 
import matplotlib.pyplot as plt 
url = "https://www.example.com" 
text = requests.get(url).text 
tokens = re.findall(r"[A-Za-z0-9]+", text.lower()) 
freq = FreqDist(tokens) 		
print(freq.most_common(10))     
freq.plot(10)
