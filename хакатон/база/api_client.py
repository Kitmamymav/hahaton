import requests
from typing import List, Dict, Any, Optional

class OilAPIClient:
    def __init__(self, base_url: str):
        # Удаляем лишний слеш на конце URL для корректной конкатенации
        self.base_url = base_url.strip().rstrip('/')

    def check_health(self) -> bool:
        """Проверка доступности сервера (эндпоинт /models)."""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            return response.status_code == 200
        except:
            return False

    def get_models(self) -> List[Any]:
        """Получение списка всех обученных моделей."""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            if response.status_code == 200:
                return response.json()
            return []
        except:
            return []

    def train_model(self, data: List[Dict[str, Any]]) -> Optional[str]:
        """Отправка обучающей выборки на сервер. Возвращает уникальный ID модели."""
        url = f"{self.base_url}/train"
        
        # ДОБАВИЛИ КЛЮЧ "model", КОТОРЫЙ ТРЕБОВАЛ СЕРВЕР
        payload = {
            "model": "LinearRegression", # Указываем название модели
            "data": data
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json().get("model_id")
            else:
                print(f"\n[ОШИБКА API] Статус: {response.status_code}")
                print(f"[ОШИБКА API] Ответ сервера: {response.text}\n")
                return None
        except Exception as e:
            print(f"\n[СИСТЕМНАЯ ОШИБКА] {e}\n")
            return None


    def predict(self, model_id: str, data: List[Dict[str, Any]]) -> List[float]:
        """Получение прогнозов качества нефти для тестовой выборки."""
        url = f"{self.base_url}/predict"
        payload = {"model_id": model_id, "data": data}
        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json().get("predictions", [])
            return []
        except:
            return []

    def get_metrics(self, y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
        """Вычисление метрик (MAE, RMSE, R2) через внешний API."""
        url = f"{self.base_url}/metrics"
        payload = {"y_true": y_true, "y_pred": y_pred}
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                return response.json()
            return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
        except:
            return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
