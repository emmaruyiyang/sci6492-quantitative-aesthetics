# Two Londons — AI-assisted assignment 2 (comparative literature)

The same words drawn in two word2vec models, side by side, in the style of assignment 1 (Semantic Constellation).

- **The Complete Sherlock Holmes**, Arthur Conan Doyle, 1887–1927: London seen by a middle-class detective
- **The People of the Abyss**, Jack London, 1903: London's East End, written after living there as a pauper

Both books describe the same city in the same years from opposite sides.

## Files

| file | what it is |
|---|---|
| `raw/cano_raw.txt` | complete Sherlock Holmes canon, already plain text: https://sherlock-holm.es/stories/plain-text/cano.txt |
| `raw/abyss_raw.txt` | Project Gutenberg #1688: https://www.gutenberg.org/ebooks/1688 |
| `clean_texts.py` | removes contents, headings, footnotes, captions, tables, licence text; one paragraph per line |
| `*_clean.txt` | cleaned texts (Holmes ≈ 655k words, Abyss ≈ 62k words) |
| `train_models.py` | word2vec skip-gram, 100 dims, window 5, min_count 5, 100 epochs |
| `*.kv` | trained models (Holmes 6969 words, Abyss 1518 words) |
| `app.py`, `web/` | Flask + SVG comparison app |

## Run

From the `sci6492_code` folder (the `.venv` there is Python 3.12 with flask, gensim, scikit-learn):

```bash
.venv/bin/python 03_language_embeddings/comparative_literature/clean_texts.py    # only if the txt files change
.venv/bin/python 03_language_embeddings/comparative_literature/train_models.py   # only if the txt files change
.venv/bin/python 03_language_embeddings/comparative_literature/app.py
```

Open http://127.0.0.1:5051. URL parameters make screenshots repeatable, e.g.
`/?word=street&mode=shared`, `/?word=night&source=list&mode=color`, `/?word=house&mode=axis&axis=clean-dirty`.

## How the words are chosen

- **Neighbours** (default): the N nearest words to the seed in each book (stop words skipped), merged into one list.
- **Gehl list**: the vocabulary of Jan Gehl's *Life Between Buildings* (street, court, door, crowd, walk, sat, talked …),
  kept to the 19 words that occur at least 15 times in both books, so their vectors are reliable.

Each panel draws every chosen word its book contains. Words a book does not contain (fewer than 5 uses, so no vector)
are drawn in grey italics beyond a dashed rim, in the direction they have in the other book.

## How the map is built

- radius = similarity to the centre word, on **one scale shared by both panels**, so the rings mean the same on both sides
- angle = 2D PCA of the neighbours after removing their component along the centre word (as in assignment 1)
- hovering a word highlights the same word in the other book; clicking re-centres both books on it

## Encodings

- *Similarity*: size and gray level
- *Shared*: whether the word is a neighbour in both books (bold white), this book only (gray) or the other book only (italic)
- *Color words*: dot in the nearest color word, using **only the color words each book contains**.
  Holmes has 14 (red, yellow, brown, silver …); the Abyss has 7 (grey, white, black and a little red, green, blue, gold)
- *Axis*: projection on a semantic axis built from two words in each book (day → night, clean → dirty, warm → cold, inside → outside)
