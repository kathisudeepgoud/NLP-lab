#16. Implement chunking using Shallow parsing?

# Step 1: Import required libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.chunk import RegexpParser

# Step 2: Download required resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Step 3: Input sentence
sentence = "The little boy saw a big dog in the park"

# Step 4: Tokenization
tokens = word_tokenize(sentence)

# Step 5: POS Tagging
pos_tags = pos_tag(tokens)

# Step 6: Define grammar for chunking
grammar = "NP: {<DT>?<JJ>*<NN>}"

# Step 7: Create chunk parser
chunk_parser = RegexpParser(grammar)

# Step 8: Perform chunking
chunk_tree = chunk_parser.parse(pos_tags)

# Step 9: Display result
print(chunk_tree)
