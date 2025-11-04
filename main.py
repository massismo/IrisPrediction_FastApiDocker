from fastapi import FastAPI
from pydantic import BaseModel
import joblib

model = joblib.load('iris_model.joblib') # загружаем файл модели и файл scaler
scaler = joblib.load('scaler.joblib')

app = FastAPI() # создаём веб объект

SPECIES_NAMES = ['setosa', 'versicolor', 'virginica'] # также записываем названия видов

class IrisFeatures(BaseModel): # через BaseModel описываем класс нужных нам данных на вход
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict") # создаём endpoint, адрес http://127.0.0.1:8000/predict
def predict_species(features: IrisFeatures): # создаём список
    data_for_model = [
        [
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]
    ]

    data_scaled = scaler.transform(data_for_model) # нормализуем данные, потом предсказываем результат
    prediction_raw = model.predict(data_scaled)

    prediction_index = int(prediction_raw[0]) # получаем индекс и возвращаем имя, соответствующее индексу
    species_name = SPECIES_NAMES[prediction_index]

    return { # ответ
        "predicted_species": species_name,
        "predicted_species_id": prediction_index,
    }