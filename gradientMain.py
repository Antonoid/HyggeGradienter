import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


class Perceptron:
    def __init__(self, input_dim, lr=0.1):
        self.W = np.random.randn(input_dim) * 0.01
        self.b = 0.0
        self.lr = lr

    def predict(self, X):
        return np.dot(X, self.W) + self.b

    def compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def train(self, X, y, epochs=100):
        loss_history = []
        n = len(X)

        for epoch in range(epochs):
            y_pred = self.predict(X)

            # Gradients (MSE)
            dL_dy = 2 * (y_pred - y) / n
            dW = np.dot(X.T, dL_dy)
            dB = np.sum(dL_dy)

            # Update
            self.W -= self.lr * dW
            self.b -= self.lr * dB

            loss_history.append(self.compute_loss(y_pred, y))

        return loss_history


# -------------------------------------------------------
# Generate a larger dataset
# -------------------------------------------------------

# np.random.seed(42)
n_train = 100
n_test = 30

# X_train = np.linspace(0, 20, 100).reshape(-1, 1)
# y_train = 2 * X_train.squeeze() + 1
X_train = np.random.uniform(0, 20, size=(n_train, 1))
y_train = 2 * X_train.squeeze() + 1


# X_test = np.linspace(21, 30, 30).reshape(-1, 1)
# y_test = 2 * X_test.squeeze() + 1
X_test = np.random.uniform(21, 30, size=(n_test, 1))
y_test = 2 * X_test.squeeze() + 1


# -------------------------------------------------------
# Train model
# -------------------------------------------------------

model = Perceptron(input_dim=1, lr=0.001)
loss_history = model.train(X_train, y_train, epochs=100)


# -------------------------------------------------------
# Evaluate on unseen data
# -------------------------------------------------------

y_test_pred = model.predict(X_test)

test_mse = np.mean((y_test_pred - y_test) ** 2)

# R² score
ss_res = np.sum((y_test - y_test_pred) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r2 = 1 - ss_res / ss_tot

# -------------------------------------------------------
# Results
# -------------------------------------------------------

print("Learned weight:", model.W[0])
print("Learned bias:", model.b)
print("Test MSE:", test_mse)
print("Test R²:", r2)

# -------------------------------------------------------
# Plots
# -------------------------------------------------------
"""
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, label="True Values")
plt.scatter(X_test, y_test_pred, label="Predictions")
plt.title("Model Predictions on Unseen Test Data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
"""
