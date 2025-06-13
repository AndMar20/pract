import pyodbc
import json
import numpy as np
import faiss
import logging
import re
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
from pymorphy2 import MorphAnalyzer

connection_string = r'Driver={ODBC Driver 17 for SQL Server};Server=(localdb)\mssqllocaldb;Database=test;Trusted_Connection=yes;'

index_file_path = 'nomenclature.index'
index_map_path = 'nomenclature_map.json'
embedding_dim = 1024

model = SentenceTransformer('deepvk/USER-bge-m3')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

morph = MorphAnalyzer()

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.replace('ё', 'е')
    # text = re.sub(r'\d+\.\d+', '', text)
    # text = re.sub(r'\bсч\b', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    words = text.split()
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
    return " ".join(lemmatized_words)

def fetch_nomenclatures(batch_size=1000):
    logger.info("Подключение к базе данных и извлечение номенклатур...")
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()

    offset = 0
    nomenclatures = []
    nomenclature_ids = []

    while True:
        cursor.execute("""
            SELECT 
                Nomenclatures.NomenclatureId, 
                Nomenclatures.Nomenclature
            FROM 
                Nomenclatures
            ORDER BY Nomenclatures.NomenclatureId
            OFFSET ? ROWS FETCH NEXT ? ROWS ONLY
        """, (offset, batch_size))

        rows = cursor.fetchall()

        if not rows:
            logger.info("Все номенклатуры извлечены.")
            break

        for row in rows:
            nomenclature_ids.append(row.NomenclatureId)
            nomenclatures.append(row.Nomenclature)

        offset += batch_size

    conn.close()
    return nomenclature_ids, nomenclatures

def generate_embeddings_parallel(nomenclatures, batch_size=1000):
    logger.info(f"Генерация эмбеддингов для {len(nomenclatures)} номенклатур с использованием многозадачности...")

    embeddings = []
    total_batches = (len(nomenclatures) // batch_size) + (1 if len(nomenclatures) % batch_size > 0 else 0)

    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(total_batches):
            batch = nomenclatures[i * batch_size: (i + 1) * batch_size]
            futures.append(executor.submit(model.encode, batch))

        for future in futures:
            batch_embeddings = future.result()
            embeddings.extend(batch_embeddings)
            
    embeddings = [embedding / np.linalg.norm(embedding) for embedding in embeddings]

    return embeddings

def build_faiss_index(ids, embeddings):
    logger.info("Построение FAISS-индекса...")
    vectors = np.stack(embeddings)
    num_vectors = len(vectors)

    nlist = max(1, int(np.sqrt(num_vectors)))
    logger.info(f"Количество эмбеддингов: {num_vectors}, выбран nlist = {nlist}")
    quantizer = faiss.IndexFlatIP(embedding_dim)
    index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    7666
    index.nprobe = min(40, max(8, nlist // 10))
    logger.info(f"Установлен nprobe = {index.nprobe}")

    index_id_map = faiss.IndexIDMap(index)
    index_id_map.add_with_ids(vectors, np.array(range(len(ids)), dtype=np.int64))

    return index_id_map

def main():
    try:
        logger.info("Загрузка номенклатур из базы данных...")
        nomenclature_ids, nomenclatures = fetch_nomenclatures()

        if not nomenclatures:
            logger.warning("Нет номенклатур для обработки.")
            return
        
        logger.info("Нормализация и приведение к единому виду...")
        normalized_nomenclatures = [(normalize(name)) for name in nomenclatures]

        logger.info(f"Получено {len(normalized_nomenclatures)} номенклатур. Генерация эмбеддингов...")
        embeddings = generate_embeddings_parallel(normalized_nomenclatures)

        faiss_ids = list(range(len(normalized_nomenclatures)))
        logger.info(f"Построение FAISS-индекса...")
        index = build_faiss_index(faiss_ids, embeddings)

        logger.info(f"Сохранение индекса в файл: {index_file_path}")
        faiss.write_index(index, index_file_path)

        indexed_map = {
            str(idx): {
                "nomenclatureId": nomenclature_ids[idx],
                "nomenclature": nomenclatures[idx]
            }
            for idx in range(len(normalized_nomenclatures))
        }

        with open('nomenclature_map.json', 'w', encoding='utf-8') as f:
            json.dump(indexed_map, f, ensure_ascii=False, indent=4)

        logger.info("Процесс завершён успешно.")

    except Exception as e:
        logger.error(f"Произошла ошибка: {e}", exc_info=True)

if __name__ == "__main__":
    main()
