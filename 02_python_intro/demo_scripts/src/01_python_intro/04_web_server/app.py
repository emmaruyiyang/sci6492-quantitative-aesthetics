from pathlib import Path

import gensim.downloader
from flask import Flask, jsonify, redirect, request, send_from_directory

# Only files inside WEB are ever served; app.py sits outside it, so it stays private.
WEB = Path(__file__).resolve().parent / "web"

app = Flask(__name__)

# Loaded once, when the server starts, not once per request.
model = gensim.downloader.load("glove-wiki-gigaword-50")


@app.get("/")
def index():
    """The page lives at /web, so send the bare root there."""
    return redirect("/web")


@app.get("/web")
def web():
    return send_from_directory(WEB, "index.html")


@app.get("/web/<path:filename>")
def web_file(filename):
    """Serve style.css and app.js, which sit next to index.html."""
    return send_from_directory(WEB, filename)


@app.get("/api/similar")
def similar():
    word = request.args.get("word", "").strip().lower()
    top_n = request.args.get("top_n", 10, type=int)

    if not word:
        return jsonify(error="Please enter a word."), 400
    if word not in model:
        return jsonify(error=f"'{word}' is not in the model's vocabulary."), 404

    results = [
        {"word": similar_word, "similarity": float(similarity)}
        for similar_word, similarity in model.most_similar(word, topn=top_n)
    ]
    return jsonify(word=word, results=results)


if __name__ == "__main__":
    print("Open http://127.0.0.1:5000/web")
    app.run(debug=True, use_reloader=False)
