# 1. Импорт библиотек
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Загружаем предобработанные данные
df = pd.read_csv('preprocessed.csv')

# 3. Разбиваем на признаки X и целевую переменную y
X = df['clean_text']
y = df['Tag']

# 4. Делим на train и test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. TF-IDF векторизация
vectorizer_path = "tfidf_vectorizer.pkl"
if os.path.exists(vectorizer_path):
    vectorizer = joblib.load(vectorizer_path)
    x_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
else:
    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        stop_words=None
    )
    x_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    joblib.dump(vectorizer, vectorizer_path)

print("Размерность обучающей матрицы:", x_train_vec.shape)

# 6. Обучение или загрузка модели
model_path = "logreg_model.pkl"
if os.path.exists(model_path):
    print("Загружаем обученную модель...")
    lr = joblib.load(model_path)
else:
    print("Обучаем модель...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(x_train_vec, y_train)
    joblib.dump(lr, model_path)

# 7. Оценка модели
labels = lr.classes_.tolist()
y_pred = lr.predict(X_test_vec)

print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=lr.classes_)
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=labels,
            yticklabels=labels,
            cmap="Blues")
plt.xlabel("Предсказано")
plt.ylabel("Истинный класс")
plt.title("Матрица ошибок")
plt.show()

print("Точность модели:", accuracy_score(y_test, y_pred))