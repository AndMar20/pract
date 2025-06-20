import unittest
import requests

BASE_URL = "http://127.0.0.1:5005"

class TestFaissSearchAPI(unittest.TestCase):
    # Тест поиска по запросу
    def test_successful_search(self):
        response = requests.post(f"{BASE_URL}/faiss/search", json={
            "query": "болгарка",
            "top_k": 5
        })

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("results", data)
        self.assertGreater(len(data["results"]), 0)
        for item in data["results"]:
            self.assertIn("nomenclatureId", item)
            self.assertIn("nomenclature", item)
            self.assertIn("distance", item)

    # Тест обработки пустого запроса
    def test_empty_query(self):
        response = requests.post(f"{BASE_URL}/faiss/search", json={
            "query": ""
        })
        self.assertEqual(response.status_code, 400)

    # Тест отсутствия поля query
    def test_missing_query_field(self):
        response = requests.post(f"{BASE_URL}/faiss/search", json={})
        self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()