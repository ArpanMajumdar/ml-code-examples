import numpy as np


class Perceptron:
    def __init__(
        self, learning_rate: float, n_iter: int = 50, random_state: int = 1
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.rng = np.random.RandomState(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X: (n_examples, n_features)
        y: (n_examples, )
        """
        n_examples, n_features = X.shape
        self.w = self.rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.b = np.float32(0.0)
        self.errors = []

        for i in range(self.n_iter):
            error = 0
            for xi, yi in zip(X, y):
                yi_hat = self.predict(yi)
                update = self.learning_rate * (yi - yi_hat)
                dw = update * xi
                db = update
                self.w += dw
                self.b += db
                error += int(update != 0.0)
            self.errors.append(error)

    def net_input(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w) + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.net_input(X) > 0.0, 1, 0)
