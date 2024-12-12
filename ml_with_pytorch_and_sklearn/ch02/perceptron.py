from typing import Protocol

import numpy as np


class Classifier(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray): ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


class Perceptron:
    def __init__(
        self, learning_rate: float, n_iter: int = 50, random_state: int = 1
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X: (n_examples, n_features)
        y: (n_examples, )
        """
        _, n_features = X.shape
        rng = np.random.RandomState(self.random_state)
        self.w = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.b = 0.0
        self.errors = []

        for i in range(self.n_iter):
            error = 0
            for xi, yi in zip(X, y):
                yi_hat = self.predict(xi)
                update = self.learning_rate * (yi - yi_hat)
                # dw = learning_rate * (yi - yi_hat) * xi
                # db = learning_rate * (yi - yi_hat)
                dw = update * xi
                db = update
                self.w += dw
                self.b += db
                # print(
                #     f"xi = {xi}, yi = {yi}, yi_hat = {yi_hat}, update = {update}, w={self.w}, b={self.b}",
                # )
                error += int(update != 0.0)
            self.errors.append(error)

    def net_input(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w) + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.net_input(X) >= 0.0, 1, 0)
