import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

class LinearClassifierGD:
    def __init__(self, loss_type='logistic', lr=0.01, lambda1=0.0, lambda2=0.0, epochs=1000):
        self.loss_type = loss_type
        self.lr = lr
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.epochs = epochs
        self.w = None
        self.history = {'loss': [], 'acc': []}

    def _loss_and_grad(self, X, y):
        margins = y * (X @ self.w)
        n = X.shape[0]

        if self.loss_type == 'logistic':
            # L = log(1 + exp(-M))
            loss = np.mean(np.log1p(np.exp(-margins)))
            # dL/dw = -y * x / (1 + exp(M))
            grad = - (X.T @ (y / (1 + np.exp(margins)))) / n
        elif self.loss_type == 'hinge':
            # L = max(0, 1 - M)
            loss = np.mean(np.maximum(0, 1 - margins))
            # dL/dw = -y * x, если M < 1, иначе 0
            mask = (margins < 1).astype(float)
            grad = - (X.T @ (y * mask)) / n
        elif self.loss_type == 'exponential':
            # L = exp(-M)
            loss = np.mean(np.exp(-margins))
            # dL/dw = -y * x * exp(-M)
            grad = - (X.T @ (y * np.exp(-margins))) / n
        else:
            raise ValueError("Unknown loss type")

        reg_grad = np.zeros_like(self.w)
        reg_grad[1:] = self.lambda1 * np.sign(self.w[1:]) + 2 * self.lambda2 * self.w[1:]

        return loss, grad + reg_grad

    def fit(self, X, y):
        n_features = X.shape[1]
        self.w = np.zeros(n_features)

        for epoch in range(self.epochs):
            loss, grad = self._loss_and_grad(X, y)
            self.w -= self.lr * grad
            if epoch % 10 == 0:
                self.history['loss'].append(loss)
                preds = np.sign(X @ self.w)
                self.history['acc'].append(accuracy(y, preds))

    def predict(self, X):
        return np.sign(X @ self.w)