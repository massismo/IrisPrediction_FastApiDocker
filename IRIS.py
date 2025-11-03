
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

# 1. Загрузка данных
iris = load_iris()
X = iris.data            # shape (150, 4)
y = iris.target          # метки 0,1,2
feature_names = iris.feature_names
target_names = iris.target_names

# 2. Быстрый взгляд на данные
df = pd.DataFrame(X, columns=feature_names)
df['species'] = [target_names[i] for i in y]
print(df.head())

# 3. Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# stratify=y сохраняет соотношение классов в сплитах

# 4. Масштабирование (нужно для многих алгоритмов)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Выбор модели — попробуем k-NN и логистическую регрессию
knn = KNeighborsClassifier(n_neighbors=5)
logreg = LogisticRegression(max_iter=200, multi_class='auto')

# 6. Обучение
knn.fit(X_train_scaled, y_train)
logreg.fit(X_train_scaled, y_train)

# 7. Предсказания и оценка
for name, model in [('k-NN', knn), ('LogisticRegression', logreg)]:
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f'\n{name} accuracy: {acc:.3f}')
    print(classification_report(y_test, y_pred, target_names=target_names))
    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred))

# 8. Простая кросс-валидация (оценка стабильности)
scores = cross_val_score(knn, scaler.fit_transform(X), y, cv=5)
print('\nk-NN 5-fold CV accuracy:', np.round(scores, 3), 'mean =', scores.mean())

"""
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # длина/ширина чашелистика, лепестка
new_sample_scaled = scaler.transform(new_sample)
pred = knn.predict(new_sample_scaled)
print('\nPredicted species for', new_sample, '->', target_names[pred[0]])
"""
# 10. Визуализация (пара признаков)
sns.pairplot(df, hue='species', vars=feature_names)
plt.suptitle('Pairplot of Iris features', y=1.02)
plt.show()

joblib.dump(knn, 'iris_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
