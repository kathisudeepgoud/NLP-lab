# 7. Identify the parts of speech in the document?
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
nltk.download('punkt')	#  Downloads the Punkt tokenizer model. Required for splitting text into words and sentences. Punkt is a pre-trained model used for tokenization.
nltk.download('averaged_perceptron_tagger')	#// Averaged perceptron tagger is used for POS tagging in NLTK.

text = "NLP helps computers understand human language."
tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)
print("Word\tPOS Tag")
for word, tag in pos_tags:
    print(word, "\t", tag)
