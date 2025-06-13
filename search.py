from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np 
import faiss
import threading
import re
import json
import os
from flask_cors import CORS
from pymorphy2 import MorphAnalyzer
from ruwordnet import RuWordNet

app = Flask(__name__)
CORS(app)
model = SentenceTransformer('deepvk/USER-bge-m3')

faiss_index = None
nomenclature_map = {} 

index_lock = threading.Lock()

DIM = 1024



def load_faiss_index():
    global faiss_index
    if os.path.exists("nomenclature.index"):
        print("Загрузка FAISS индекса...")
        faiss_index = faiss.read_index("nomenclature.index")
        print("Индекс загружен!")
    else:
        print("Файл индекса не найден!")

def load_nomenclature_map():
    global nomenclature_map
    if os.path.exists("nomenclature_map.json"):
        print("Загрузка карты номенклатуры...")
        with open("nomenclature_map.json", "r", encoding="utf-8") as f:
            nomenclature_map.update(json.load(f))
        print(f"Загружено {len(nomenclature_map)} записей.")
    else:
        print("Файл карты номенклатуры не найден!")

morph = MorphAnalyzer()

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.replace('ё', 'е')
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
    return " ".join(lemmatized_words)

def get_synonyms(word: str) -> set:
    ruwordnet = RuWordNet()
    synonyms = set()
    try:
        synsets = ruwordnet.synsets(word)
        for synset in synsets:
            for lemma in synset.lemmas:
                synonyms.add(lemma.name)
        if synonyms:
            print(f"Синонимы для '{word}': {', '.join(synonyms)}")
        else:
            print(f"Синонимов не найдено для '{word}'")
    except Exception as e:
        print(f"Ошибка при получении синонимов для слова '{word}': {e}")
    return synonyms

def expand_query_with_synonyms(query: str) -> str:
    words = query.split()
    expanded_query = []
    
    for word in words:
        normalized_word = normalize(word)
        synonyms = get_synonyms(normalized_word)
        if synonyms:
            expanded_query.extend(synonyms)
        else:
            expanded_query.append(normalized_word)

    final_query = " ".join(expanded_query)
    print(f"Расширенный запрос: {final_query}")
    return final_query

@app.route("/faiss/search", methods=["POST"])
def search_similar():
    data = request.get_json()
    query = data.get("query", "")
    top_k = data.get("top_k", 10)
    
    if not query or faiss_index is None:
        return jsonify({"error": "Ошибка запроса или индекс не найден"}), 400

    # expanded_query = expand_query_with_synonyms(query)
    normalize_query = normalize(query)

    print(f"Получен запрос на поиск: {normalize_query}")
    embedding = model.encode([normalize_query])[0].astype("float32")

    with index_lock:
        D, I = faiss_index.search(np.array([embedding]), top_k)

    results = []
    for idx, dist in zip(I[0], D[0]):
        print(f"Индекс: {idx}, Расстояние: {dist}")
        
        idx_str = str(idx)
        
        meta = nomenclature_map.get(idx_str)
        if meta:
            print(f"Найдено соответствие для индекса {idx_str}: {meta}")
            results.append({
                "nomenclatureId": meta["nomenclatureId"],
                "nomenclature": meta["nomenclature"],
                "distance": float(dist)
            })
        else:
            print(f"Нет соответствия для индекса {idx_str}")

    if not results:
        print("Не найдено ни одного соответствия")
        
    return jsonify({"results": results})

if __name__ == "__main__": 
    load_faiss_index()
    load_nomenclature_map()
    app.run(host="127.0.0.1", port=5005)