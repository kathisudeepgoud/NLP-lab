#10. Implement Brill tagger?
import nltk 
from nltk.corpus import treebank 
from nltk.tag import UnigramTagger 
from nltk.tag.brill_trainer import BrillTaggerTrainer 
nltk.download('treebank') 
train = treebank.tagged_sents()[:2000] 
test = treebank.tagged_sents()[2000:] 
base = UnigramTagger(train) 
trainer = BrillTaggerTrainer(base, []) 
brill = trainer.train(train) 
print("Accuracy:", brill.evaluate(test))
