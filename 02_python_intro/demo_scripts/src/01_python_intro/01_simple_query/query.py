import gensim.downloader

word = "architecture"
top_n = 10


model_name = "glove-wiki-gigaword-50"
print(f"Loading model '{model_name}'...")
model : gensim.models.KeyedVectors = gensim.downloader.load(model_name)
print(f"Model '{model_name}' loaded.")


print(f"Start query for top {top_n} words most similar to '{word}':")
words = model.most_similar(word, topn=top_n)
print("query end")


for rank, (similar_word, similarity) in enumerate(words):
    print(f"{rank:2}. {similar_word:<15} {similarity:.4f}")
