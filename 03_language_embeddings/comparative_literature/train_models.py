#trains one word2vec model per cleaned text (run clean_texts.py first)
#based on 03_gensim_train.py, with a few changes for small corpora (see comments)

import logging
import os
import re

from gensim.models import Word2Vec
from multiprocessing import cpu_count

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
this_dir = os.path.dirname(os.path.abspath(__file__))

#TODO_______________________parameters to set (only edit here)
#pairs of (input text file, output model file)
corpora = [
    ('sherlock_holmes_clean.txt', 'sherlock_holmes.kv'),
    ('people_of_the_abyss_clean.txt', 'people_of_the_abyss.kv'),
]
#smaller than the 150 of the course script, as both corpora are small
vector_size = 100
#words appearing fewer times than this are ignored; their vectors would be mostly noise
min_count = 5
#number of words on each side of a word that count as its context
window = 5
#passes over the text; small corpora need many passes
epochs = 100
#_______________________end of parameters to set


def read_sentences(path):
    #each line of the cleaned files is one paragraph
    sentences = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            l = line.strip().lower()
            if len(l) != 0:
                #words, keeping inner apostrophes so "don't" stays one token rather than "don" + "t"
                tokens = re.findall(r"[a-z]+(?:'[a-z]+)*", l)
                sentences.append(tokens)
    return sentences


for input_text_file, output_file in corpora:
    print(f'training on {input_text_file}')
    sentences = read_sentences(os.path.join(this_dir, input_text_file))

    model = Word2Vec(
        sentences,
        min_count=min_count,
        workers=cpu_count(),
        vector_size=vector_size,
        window=window,
        sg=1,               #skip-gram: usually better than the default CBOW for small corpora
        epochs=epochs,
        seed=42             #makes results repeatable (with workers=1 for full determinism)
        )
    model.wv.save(os.path.join(this_dir, output_file))
    print(f'saved {output_file}: {len(model.wv)} words')
