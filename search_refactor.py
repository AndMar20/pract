from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from pymorphy2 import MorphAnalyzer

import numpy as np
import faiss
import threading
import re
import json
import os
import logging

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Константы
DIM = 1024
FAISS_INDEX_PATH = "nomenclature.index"
MAP_PATH = "nomenclature_map.json"

# Инициализация приложения
app = Flask(__name__)
CORS(app)

# Модель и морфологический анализатор
model = SentenceTransformer("deepvk/USER-bge-m3")
morph = MorphAnalyzer()

# Общие ресурсы 
faiss_index = None
nomenclature_map = {}
index_lock = threading.Lock()


# Утилиты 
def normalize(text: str) -> str:
    """Приводит текст к нормализованному лемматизированному виду."""
    text = text.lower().strip().replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    words = text.split()
    lemmatized = [morph.parse(word)[0].normal_form for word in words]
    return " ".join(lemmatized)


def load_json(path: str) -> dict:
    """Загружает JSON-файл в словарь."""
    if not os.path.exists(path):
        logger.warning(f"Файл не найден: {path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        logger.info(f"Загружено {len(data)} записей из {path}")
        return data


def load_faiss_index(path: str) -> faiss.Index:
    """Загружает FAISS-индекс из файла."""
    if not os.path.exists(path):
        logger.warning(f"Индекс не найден: {path}")
        return None

    logger.info("Загрузка FAISS индекса...")
    index = faiss.read_index(path)
    logger.info("FAISS индекс загружен")
    return index


# Основной маршрут 
@app.route("/faiss/search", methods=["POST"])
def search_similar():
    payload = request.get_json()
    query = payload.get("query")
    top_k = payload.get("top_k", 10)

    if not query or faiss_index is None:
        logger.warning("Некорректный запрос или индекс не загружен")
        return jsonify({"error": "Некорректный запрос или индекс не загружен"}), 400

    normalized = normalize(query)
    logger.info(f"Запрос: '{normalized}'")

    try:
        embedding = model.encode([normalized])[0].astype("float32")
    except Exception as e:
        logger.error(f"Ошибка при кодировании: {e}")
        return jsonify({"error": "Ошибка при получении эмбеддинга"}), 500

    with index_lock:
        D, I = faiss_index.search(np.array([embedding]), top_k)

    results = []
    for idx, dist in zip(I[0], D[0]):
        meta = nomenclature_map.get(str(idx))
        if meta:
            results.append({
                "nomenclatureId": meta["nomenclatureId"],
                "nomenclature": meta["nomenclature"],
                "distance": float(dist)
            })
        else:
            logger.debug(f"Нет данных для индекса {idx}")

    if not results:
        logger.info("Совпадений не найдено")

    return jsonify({"results": results})


# --- Точка входа ---
def initialize():
    global faiss_index, nomenclature_map
    faiss_index = load_faiss_index(FAISS_INDEX_PATH)
    nomenclature_map = load_json(MAP_PATH)

if __name__ == "__main__":
    initialize()
    app.run(host="127.0.0.1", port=5005)
