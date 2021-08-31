import numpy as np

class Ridge_regression:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.W = None

    def fit(self, X, y):# X (n, m) y(n, 1)
        X, y = np.array(X), np.array(y) # その場限りだから
        X = np.insert(X, 0, 1, axis=1)
        # self.w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)+np.dot(self.alpha, np.identity(X.shape[1]))), X.T), y)
        self.w = np.linalg.inv(X.T @ X + self.alpha * np.identity(X.shape[1])) @ X.T @ y
        print("finish fit")

    def predict(self, X):
        X = np.array(X)
        X = np.insert(X, 0, 1, axis=1)
        return np.dot(X, self.w)