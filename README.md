# Это тестовый проект, демонстрирующий цикл MLOps. 

## Краткое описание файлов.
- IRIS.py: Тут модель обучается на данных IRIS и сохраняет данные в .joblib.
- main.py: Код REST API на FastAPI. Загружает .joblib файлы и предоставляет эндпоинт.
- Dockerfile: Инструкция для Docker по сборке образа, на котором запускается main.py.
- requirements.txt: Список Python-зависимостей, необходимых для установки.
- iris_model.joblib: Сохраненный файл обученной ML-модели.
- scaler.joblib: Сохраненный файл StandardScaler. Он обязателен для корректной предобработки данных, поступающих в API.


## Инструкция по тестированию.
Чтобы сделать запрос на предсказание цветка ириса:

1. Вам понадобится Docker desktop на устройстве.

2. Далее перейдите по ссылке https://hub.docker.com/r/massismo/iris-fastapi-service, установите docker контейнер и запустите его.
Установку и запуск можно произвести командой: ```docker run -d -p 8000:8000 massismo/iris-fastapi-service:latest```
После запуска, сервис будет активен по адресу http://127.0.0.1:8000/docs.

3. Направить запрос на предсказание можно через терминал вашего устройства. Откройте PowerShell любым способом и напишите команду (для windows):
```
$body = @{
  sepal_length = 5.1
  sepal_width  = 3.5
  petal_length = 1.4
  petal_width  = 0.2
} | ConvertTo-Json

Invoke-RestMethod -Uri 'http://127.0.0.1:8000/predict' -Method Post -Body $body -ContentType 'application/json' 
```

Либо команду для Bash (Linux):

```curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'```

По желанию в любом текстовом редакторе меняйте основные 4 параметра цветка.

Также запросить предсказание можно через интерактивную документацию Swagger UI.
В браузере откройте http://127.0.0.1:8000/docs, далее нажмите на POST/predict, потом на кнопку Try it out,
в появивишемся поле введите параметры цветка и намите execute, ниже в поле Response body появится индекс и название цветка.
