import argparse
import sys

import gensim.downloader

#run using python .\src\01_python_intro\03_console_app\query.py architecture -n 5     

parser = argparse.ArgumentParser(description="Print the words most similar to a given word.")
parser.add_argument("word", help="the word to look up")
parser.add_argument("-n", "--top-n", type=int, default=10, help="how many similar words to print (default: 10)")
args = parser.parse_args()

word = args.word.strip().lower()
top_n = args.top_n

model = gensim.downloader.load("glove-wiki-gigaword-50")

if word not in model:
    print(f"'{word}' is not in the model's vocabulary.", file=sys.stderr)
    raise SystemExit(1)

print(f"Top {top_n} words most similar to '{word}':")
for rank, (similar_word, similarity) in enumerate(model.most_similar(word, topn=top_n), start=1):
    print(f"{rank:2}. {similar_word:<15} {similarity:.4f}")
