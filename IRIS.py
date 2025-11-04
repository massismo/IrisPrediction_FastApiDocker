import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

iris = load_iris()
X = iris.data            # shape (4 параметра цветка)
y = iris.target          # index (индекс по виду: 0, 1, 2)
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df['species'] = [target_names[i] for i in y]
print(df.head()) # выводим табличку чтобы посмотреть как выглядят данные

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
) # распределяем сколько данных пойдёт на обучение, а сколько на тестирвание
# stratify=y сохраняет соотношение классов в сплитах

scaler = StandardScaler() # нормализуем данные от 0 до 1
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5) # модель K-NN
logreg = LogisticRegression(max_iter=200, multi_class='auto') # логическая регрессия

knn.fit(X_train_scaled, y_train) # обучаем модели через .fit на нормализованных данных
logreg.fit(X_train_scaled, y_train)

for name, model in [('k-NN', knn), ('LogisticRegression', logreg)]: # тут результаты по предсказанию на тестовых данных
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f'\n{name} accuracy: {acc:.3f}')
    print(classification_report(y_test, y_pred, target_names=target_names)) # precision, recall, f1-score, support
    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred)) # таблица, которая визуально показывает, где именно модель ошибается
                                            # по диагонали кол-во правильных предсказаний, вне диагонали ошибки

scores = cross_val_score(knn, scaler.fit_transform(X), y, cv=5) # кросс-валидация для K-NN и logreg
print('\nk-NN 5-fold CV accuracy:', np.round(scores, 3), 'mean =', scores.mean())
scores = cross_val_score(logreg, scaler.fit_transform(X), y, cv=5)
print('\nlogreg 5-fold CV accuracy:', np.round(scores, 3), 'mean =', scores.mean())

sns.pairplot(df, hue='species', vars=feature_names) # дополнительно визуализируем графиками распределения данных для наглядности
plt.suptitle('Pairplot of Iris features', y=1.02)
plt.show()

joblib.dump(knn, 'iris_model.joblib') # сохраняем joblib файлы с данными по модели (в данном случае K-NN) и нормализации
joblib.dump(scaler, 'scaler.joblib')
