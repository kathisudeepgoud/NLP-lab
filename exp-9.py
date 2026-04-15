#9. Implement Regex tagger?

import nltk
from nltk.tag import RegexpTagger
from nltk.tokenize import word_tokenize
patterns = [
    (r'.*ing$', 'VBG'),
    (r'.*ed$', 'VBD'),
    (r'.*es$', 'VBZ'),
    (r'.*ould$', 'MD'),
    (r'.*\'s$', 'NN$'),
    (r'.*s$', 'NNS'),
    (r'^-?[0-9]+(\.[0-9]+)?$', 'CD'),
    (r'.*', 'NN')
]
regex_tagger = RegexpTagger(patterns)
sentence = "The boy is playing and watched 3 movies"
tokens = word_tokenize(sentence)
print(regex_tagger.tag(tokens))

