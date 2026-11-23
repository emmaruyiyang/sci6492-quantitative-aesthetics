import gensim.downloader

top_n = 10

model = gensim.downloader.load("glove-wiki-gigaword-50")

while True:
    word = input("\nWord (empty to quit): ").strip().lower()
    if not word:
        break

    if word not in model:
        print(f"'{word}' is not in the model's vocabulary.")
        continue

    print(f"Top {top_n} words most similar to '{word}':")
    for rank, (similar_word, similarity) in enumerate(model.most_similar(word, topn=top_n), start=1):
        print(f"{rank:2}. {similar_word:<15} {similarity:.4f}")
