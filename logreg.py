import numpy as np

class ManualLogisticRegression:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

def predict_proba(self, X):
    if self.weights is None:
        raise ValueError("Модель не обучена. Сначала вызови .fit()")
    return self._sigmoid(np.dot(X, self.weights))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.weights = np.zeros(X.shape[1])

        for _ in range(self.epochs):
            y_pred = self._sigmoid(np.dot(X, self.weights))
            grad = np.dot(X.T, (y_pred - y)) / len(y)
            self.weights -= self.lr * grad

    def predict_proba(self, X):
        X = np.array(X)
        return self._sigmoid(np.dot(X, self.weights))

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)

    def get_weights(self):
        return self.weights