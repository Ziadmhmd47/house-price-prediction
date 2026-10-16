
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. Load Dataset

file = pd.read_csv(r"e:\Stud\3-ML\بيت\house_price.csv")

x1 = file["area"].to_numpy(dtype=float)
x2 = file["rooms"].to_numpy(dtype=float)
y = file["price"].to_numpy(dtype=float)


# 2. Check Data
if not np.isfinite(np.column_stack((x1, x2, y))).all():
    raise ValueError("Dataset contains missing or invalid values.")


# 3. Split Data into Training and Testing
np.random.seed(42)

indices = np.random.permutation(len(y))
split_index = int(0.8 * len(y))

train_indices = indices[:split_index]
test_indices = indices[split_index:]

x1_train = x1[train_indices]
x2_train = x2[train_indices]
y_train = y[train_indices]

x1_test = x1[test_indices]
x2_test = x2[test_indices]
y_test = y[test_indices]

print("Training samples:", len(y_train))
print("Testing samples:", len(y_test))


# 4. Feature Scaling
x1_mean = x1_train.mean()
x1_std = x1_train.std()

x2_mean = x2_train.mean()
x2_std = x2_train.std()

y_mean = y_train.mean()
y_std = y_train.std()

if x1_std == 0 or x2_std == 0 or y_std == 0:
    raise ValueError("A column has zero standard deviation.")

x1_train_scaled = (x1_train - x1_mean) / x1_std
x2_train_scaled = (x2_train - x2_mean) / x2_std

y_train_scaled = (y_train - y_mean) / y_std

x1_test_scaled = (x1_test - x1_mean) / x1_std
x2_test_scaled = (x2_test - x2_mean) / x2_std


# 5. Initialize Parameters
w1 = 0
w2 = 0
b = 0

alpha = 0.0024
m = len(y_train)
repeat = 2001

cost_history = []


# 6. Gradient Descent
for i in range(repeat):

    y_pred = (
        w1 * x1_train_scaled
        + w2 * x2_train_scaled
        + b
    )

    error = y_pred - y_train_scaled

    cost = np.sum(error ** 2) / (2 * m)
    cost_history.append(cost)

    if i % 100 == 0:
        print(f"Iteration: {i}, Cost: {cost}")
        print(f"w1 = {w1}")
        print(f"w2 = {w2}")
        print(f"b = {b}")
        print("-" * 40)

    dw1 = np.mean(error * x1_train_scaled)
    dw2 = np.mean(error * x2_train_scaled)
    db = np.mean(error)

    w1 -= alpha * dw1
    w2 -= alpha * dw2
    b -= alpha * db


# 7. Predict on Test Data
y_pred_test_scaled = (
    w1 * x1_test_scaled
    + w2 * x2_test_scaled
    + b
)

# Convert predictions back to original prices
y_pred_test = y_pred_test_scaled * y_std + y_mean


# 8. Compare Actual and Predicted Prices
results = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": y_pred_test,
    "Error": y_pred_test - y_test
})

print("\nTest Results:")
print(results.round(2).to_string(index=False))


# 9. Evaluate Model
mae = np.mean(np.abs(y_pred_test - y_test))

rmse = np.sqrt(
    np.mean((y_pred_test - y_test) ** 2)
)

# Baseline: predict the training-set average for every house
baseline_pred = np.full(len(y_test), y_mean)

baseline_mae = np.mean(
    np.abs(baseline_pred - y_test)
)

print("\nTest Evaluation:")
print(f"Model MAE: {mae:.2f}")
print(f"Model RMSE: {rmse:.2f}")
print(f"Baseline MAE: {baseline_mae:.2f}")


# 10. Plot Cost History
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Gradient Descent - Cost History")
plt.grid(True)
plt.show()


# 11. Plot Actual vs Predicted Prices
plt.scatter(y_test, y_pred_test)

min_price = min(y_test.min(), y_pred_test.min())
max_price = max(y_test.max(), y_pred_test.max())

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    linestyle="--"
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices (Test Set)")
plt.grid(True)
plt.show()
