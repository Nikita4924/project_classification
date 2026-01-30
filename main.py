from model_revmod import save_model, load_model

# ======= 1. Исходные данные =======
texts = [
    "ужасная доставка",
    "очень доволен покупкой",
    "отличный товар",
    "не рекомендую"
]
labels = [0, 1, 1, 0]  # 0 — негатив, 1 — позитив

# ======= 2. Векторизация вручную =======
vocab = {}
idf = []
features = []

# Строим словарь
for text in texts:
    for word in text.lower().split():
        if word not in vocab:
            vocab[word] = len(vocab)

# Строим признаки
for text in texts:
    vec = [0] * len(vocab)
    for word in text.lower().split():
        if word in vocab:
            vec[vocab[word]] += 1
    features.append(vec)

# Простейший IDF (заглушка)
idf = [1.0 for _ in vocab]

# ======= 3. Обучение модели вручную =======
# Простейшая логика: веса = сумма признаков по классу
weights = [0.0] * len(vocab)
for vec, label in zip(features, labels):
    for i, val in enumerate(vec):
        weights[i] += val if label == 1 else -val

# ======= 4. Сохранение модели =======
model_data = {
    "header": "REV_MOD_V1",
    "version": "1.0",
    "vocab": vocab,
    "idf": idf,
    "weights": weights,
    "features": features,
    "labels": labels
}

save_model(model_data, "model.revmod")
print("Модель сохранена.")

# ======= 5. Загрузка и предсказание =======
loaded = load_model("model.revmod")
print("Модель загружена.")

def predict(text, model):
    vec = [0] * len(model["vocab"])
    for word in text.lower().split():
        if word in model["vocab"]:
            idx = model["vocab"][word]
            vec[idx] += 1
    score = sum(w * x for w, x in zip(model["weights"], vec))
    return 1 if score >= 0 else 0

# Пример предсказания
new_text = "доставка отличная"
result = predict(new_text, loaded)
print(f"Текст: '{new_text}' → {'Позитивный' if result == 1 else 'Негативный'}")