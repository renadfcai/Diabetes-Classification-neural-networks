import numpy as np


class HebbianDiabetesClassifier:
    def __init__(self):
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])

        for i in range(X.shape[0]):
            # Ensure y[i] is either +1 or -1
            target = 1 if y[i] > 0 else -1
            # Hebbian rule
            self.weights += target * X[i]
            self.bias += target

    def predict(self, X):
        prediction = np.dot(X, self.weights) + self.bias
        return np.sign(prediction)
# This is what we do
