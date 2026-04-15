#Write a tagger that tags Date and Money expressions?

import re 
from nltk.tokenize import word_tokenize 
text = "I paid $50 on 12/10/2024" 
for w in word_tokenize(text): 
    if re.match(r"\$\d+", w): print(w, "-> MONEY") 
    elif re.match(r"\d{1,2}/\d{1,2}/\d{4}", w): print(w, "-> DATE")
