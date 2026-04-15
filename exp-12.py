#12. Implement NER tagger?
# Step 1: Import required libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
# Step 2: Download required resources (run once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
# Step 3: Input sentence
sentence = "Sachin Tendulkar played cricket in India for BCCI."
# Step 4: Tokenization
tokens = word_tokenize(sentence)
# Step 5: POS Tagging
pos_tags = pos_tag(tokens)
# Step 6: Named Entity Recognition
ner_tree = ne_chunk(pos_tags)
# Step 7: Print output
print(ner_tree)

