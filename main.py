from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# 1. Загружаем твою модель (делаем это 1 раз при старте)
model = joblib.load('iris_model.joblib')
scaler = joblib.load('scaler.joblib') # <-- ЗАГРУЖАЕМ SCALER

app = FastAPI()

SPECIES_NAMES = ['setosa', 'versicolor', 'virginica']

# 2. ВОТ ОНО: Описываем, какие данные мы ждем
# Названия должны совпадать с тем, на чем училась модель
class IrisFeatures(BaseModel):
    sepal_length: float  # sepal length (cm)
    sepal_width: float   # sepal width (cm)
    petal_length: float  # petal length (cm)
    petal_width: float   # petal width (cm)

# 3. Создаем endpoint для ПРЕДСКАЗАНИЯ
# Он будет доступен по адресу http://127.0.0.1:8000/predict
@app.post("/predict")
def predict_species(features: IrisFeatures):
    # 4. Превращаем данные из Pydantic в список или 2D-массив,
    # который "понимает" твоя scikit-learn модель
    data_for_model = [
        [
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]
    ]

    data_scaled = scaler.transform(data_for_model)
    # 5. Делаем предсказание
    prediction_raw = model.predict(data_scaled)

    # 6.
    prediction_index = int(prediction_raw[0])

    # 3. ИСПОЛЬЗУЕМ ИНДЕКС, ЧТОБЫ ПОЛУЧИТЬ ИМЯ
    species_name = SPECIES_NAMES[prediction_index]


    # 7. Возвращаем JSON-ответ
    return {
        "predicted_species": species_name,
        "predicted_species_id": prediction_index,
    }