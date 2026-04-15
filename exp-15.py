#15. Implement Regex parser?
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.chunk import RegexpParser
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
sentence = "The quick brown fox jumps over the lazy dog"
words = word_tokenize(sentence)
tagged_words = pos_tag(words)
grammar = "NP: {<DT>?<JJ>*<NN>}"
parser = RegexpParser(grammar)
tree = parser.parse(tagged_words)
print(tree)
tree.draw()
