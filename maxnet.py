import numpy as np


class Maxnet:
    def __init__(self, epsilon=0.01, max_iterations=1000):
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.weights = None

    def binary_activation(self, x):
        return np.where(x >= 0, 1, 0)

    def train(self, inputs):
        m = len(inputs)  # Total number of nodes (competitors)
        activations_old = self.binary_activation(inputs)
        self.weights = np.eye(m) - self.epsilon  # Initialize weights

        for _ in range(self.max_iterations):
            activations_new = self.binary_activation(
                activations_old - self.epsilon * np.sum(activations_old, axis=0)
            )

            # Save activations for the next iteration
            activations_old = np.copy(activations_new)

            # Test stopping condition
            if np.count_nonzero(activations_new) <= 1:
                break

        return activations_new

    def predict(self, inputs):
        if self.weights is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        activations = self.binary_activation(inputs)
        final_activations = self.binary_activation(
            activations - self.epsilon * np.sum(activations, axis=0)
        )
        predicted_labels = np.where(
            final_activations[:, 1] > 0, 1, 0
        )  # Assuming the second node represents the diabetic class (1)
        return predicted_labels
# edit this code 