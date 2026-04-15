#8. Implement the N-gram tagger?
import nltk
from nltk.corpus import treebank
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger
from nltk.tokenize import word_tokenize
nltk.download('treebank')
nltk.download('punkt')
train_data = treebank.tagged_sents()
unigram_tagger = UnigramTagger(train_data)
bigram_tagger = BigramTagger(train_data, backoff=unigram_tagger)
trigram_tagger = TrigramTagger(train_data, backoff=bigram_tagger)
text = "NLP helps computers understand language"
tokens = word_tokenize(text)
tagged_output = trigram_tagger.tag(tokens)
print(tagged_output)
