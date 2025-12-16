import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


class Perceptron:
    def __init__(self, input_dim, lr=0.001):
        self.W = np.random.randn(input_dim) * 0.01
        self.b = 0.0
        self.lr = lr

    def predict(self, X):
        return np.dot(X, self.W) + self.b

    def compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def train(self, X, y, epochs=2000):
        n = len(X)
        X_mean = np.mean(X, axis=0)
        X_centered = X - X_mean

        for epoch in range(epochs):
            y_pred = self.predict(X_centered)
            dL_dy = 2 * (y_pred - y) / n
            dW = np.dot(X_centered.T, dL_dy)
            dB = np.sum(dL_dy)

            self.W -= self.lr * dW
            self.b -= self.lr * dB

        # Adjust bias back to original X space
        self.b = self.b - self.W * X_mean

        # Store X mean in case you need it later
        self.X_mean = X_mean
        return

# -------------------------------------------------------
# Experiment parameters
# -------------------------------------------------------
num_runs = 10
n_train = 100
n_test = 30

weights, biases, mses, r2s = [], [], [], []

# -------------------------------------------------------
# Run the code 10 times
# -------------------------------------------------------

for run in range(num_runs):
    # Generate a new random dataset each run
    X_train = np.random.uniform(0, 20, size=(n_train, 1))
    y_train = 2 * X_train.squeeze() + 1

    X_test = np.random.uniform(21, 30, size=(n_test, 1))
    y_test = 2 * X_test.squeeze() + 1

    # Train model
    model = Perceptron(input_dim=1, lr=0.001)
    model.train(X_train, y_train, epochs=5000)

    # Predict on test data
    y_test_pred = model.predict(X_test)

    # Evaluate
    test_mse = np.mean((y_test_pred - y_test) ** 2)
    ss_res = np.sum((y_test - y_test_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - ss_res / ss_tot

    # Store results
    weights.append(model.W[0])
    biases.append(model.b)
    mses.append(test_mse)
    r2s.append(r2)

    print(f"Run {run + 1}:")
    print("  Learned weight:", model.W[0])
    print("  Learned bias:", model.b)
    print("  Test MSE:", test_mse)
    print("  Test R²:", r2)
    print("-" * 30)

# -------------------------------------------------------
# Determine overall averages
# -------------------------------------------------------
print("Averages over 10 runs:")
print("  Avg weight:", np.mean(weights))
print("  Avg bias:", np.mean(biases))
print("  Avg Test MSE:", np.mean(mses))
print("  Avg Test R²:", np.mean(r2s))
