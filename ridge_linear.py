import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

class RidgeLinearClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.w = None

    def fit(self, X, y):
        n_features = X.shape[1]
        I = np.eye(n_features)
        I[0, 0] = 0
        self.w = np.linalg.inv(X.T @ X + self.alpha * I) @ X.T @ y

    def predict(self, X):
        return np.sign(X @ self.w)

    def decision_function(self, X):
        return X @ self.w
