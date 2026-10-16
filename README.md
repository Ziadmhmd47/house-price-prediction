# House Price Prediction — Linear Regression

## Overview

This project implements a Multivariable Linear Regression model from scratch using NumPy to predict house prices based on house area and number of rooms.

The model is trained using Gradient Descent to minimize the prediction error. The project demonstrates fundamental Machine Learning concepts through a practical implementation without relying on prebuilt Machine Learning models.

## Dataset

The dataset is stored in `house_price.csv` and loaded using Pandas.

It contains the following columns:

- `area`: House area.
- `rooms`: Number of rooms.
- `price`: House price (target variable).

**Input Features:**
- House area
- Number of rooms

**Target Variable:**
- House price

Make sure `house_price.csv` is located in the same directory as the Python script.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib

## Machine Learning Algorithm

### Multivariable Linear Regression

The model predicts house prices using two input features: area and number of rooms.

The prediction equation is:

\[
\hat{y} = w_1 x_1 + w_2 x_2 + b
\]

Where:
- \(x_1\): Scaled house area.
- \(x_2\): Scaled number of rooms.
- \(w_1, w_2\): Model weights.
- \(b\): Bias.
- \(\hat{y}\): Predicted scaled house price.

### Gradient Descent

Gradient Descent is used to update the model parameters iteratively to reduce the Mean Squared Error-based cost function.

The implementation includes:
- Parameter initialization.
- Cost calculation.
- Gradient calculation for both weights and the bias.
- Parameter updates using a learning rate.

## Project Workflow

1. Load the dataset using Pandas.
2. Check for missing or invalid numerical values.
3. Split the data into training and testing sets (80% training, 20% testing).
4. Apply Feature Scaling using the training-set mean and standard deviation.
5. Initialize the model parameters.
6. Train the model using Gradient Descent.
7. Predict house prices on the test set.
8. Convert predictions back to the original price scale.
9. Evaluate predictions using MAE and RMSE.
10. Compare the model's MAE with a baseline that predicts the training-set average price.
11. Visualize the training cost and actual versus predicted prices.

## Model Evaluation

The model is evaluated on the test set using:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted prices.
- **RMSE (Root Mean Squared Error):** Measures prediction error while penalizing larger errors more heavily.
- **Baseline MAE:** Provides a simple reference by predicting the training-set average price for every test example.

The model's MAE is compared with the baseline MAE to help assess whether the trained model improves on this simple reference.

The numerical results are printed when the script runs.

## Visualizations

The project generates two plots:

1. **Gradient Descent — Cost History:** Shows how the cost changes during training.
2. **Actual vs Predicted Prices:** Compares actual house prices with model predictions on the test set. The dashed diagonal line represents perfect predictions.

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Ziadmhmd47/house-price-prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd house-price-prediction
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Script

If the Python file is named `home.py`, run:

```bash
python home.py
```

Make sure `home.py` and `house_price.csv` are in the same directory.

## Key Concepts Learned

- Supervised Learning
- Multivariable Linear Regression
- Cost Functions
- Gradient Descent
- Feature Scaling
- Training and Testing Data
- Model Evaluation
- MAE and RMSE
- Baseline Comparison
- Data Visualization

## Project Status

Completed learning project.
