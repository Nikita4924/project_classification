# 1. Импорт библиотек
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")  # чтобы не мешали предупреждения

# 2. Загружаем предобработанные данные
df = pd.read_csv('recreated.csv')

# 3. Разбиваем на признаки X и целевую переменную y
X = df['clean_text']
y = df['Tag']

# 4. Инициализируем TF-IDF векторизатор
vectorizer = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 2),
    stop_words=None
)

# 5. Повторяем обучение и оценку 10 раз
accuracies = []
f1_scores = []

for i in range(10):
    print(f"\n🔁 Повтор {i+1}")
    
    # Делим на train и test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=i,
        stratify=y
    )
    
    # Векторизация
    x_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Обучение модели
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(x_train_vec, y_train)
    
    # Предсказание
    y_pred = lr.predict(X_test_vec)
    
    # Метрики
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    f1_macro = report['macro avg']['f1-score']
    
    print(f"Точность: {acc:.2f}, F1-макро: {f1_macro:.2f}")
    
    accuracies.append(acc)
    f1_scores.append(f1_macro)

# 6. Итоговые средние значения
print("\n📊 Средние результаты по 10 запускам:")
print(f"Средняя точность: {np.mean(accuracies):.2f}")
print(f"Средний F1-макро: {np.mean(f1_scores):.2f}")