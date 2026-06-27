# Polynomial Regression using Scikit-Learn

## Project Overview

This project demonstrates Polynomial Regression using Scikit-Learn's `PolynomialFeatures`, `LinearRegression`, and `Pipeline`.

The notebook explains how polynomial regression models nonlinear relationships by transforming the original input features into higher-degree polynomial features while still using a linear regression algorithm.

---

## Objectives

- Understand Polynomial Regression
- Learn Polynomial Feature Transformation
- Train Linear Regression on transformed features
- Visualize regression curves
- Compare different polynomial degrees
- Understand Underfitting and Overfitting

---

## Technologies Used

- Python
- NumPy
- Matplotlib
- Scikit-Learn
- Jupyter Notebook

---

## Libraries

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
```

---

## Dataset

The notebook uses a synthetic nonlinear dataset generated using NumPy.

Example equation:

```
y = 0.5x² + 1.5x + 2 + noise
```

---

## Project Workflow

1. Generate nonlinear dataset
2. Split dataset into training and testing sets
3. Create Polynomial Features
4. Train Linear Regression
5. Predict values
6. Plot regression curve
7. Compare multiple polynomial degrees

---

## Function

### poly_regression(degree)

This function:

- Creates polynomial features
- Builds a machine learning pipeline
- Trains the model
- Predicts new values
- Displays the regression curve

Parameter

```
degree : int
```

Example

```python
poly_regression(1)
poly_regression(2)
poly_regression(5)
poly_regression(10)
```

---

## Output

The notebook displays:

- Training Data
- Testing Data
- Polynomial Regression Curve

for different polynomial degrees.

---

## Concepts Covered

- Polynomial Regression
- Feature Engineering
- Pipeline
- Linear Regression
- Model Training
- Model Prediction
- Visualization
- Bias-Variance Tradeoff
- Underfitting
- Overfitting

---

## Learning Outcomes

After completing this notebook, you will understand:

- Why Polynomial Regression is needed
- How PolynomialFeatures works
- How Pipeline simplifies preprocessing
- How model complexity changes with polynomial degree
- How to visualize nonlinear regression models

---

## Future Improvements

- Ridge Regression
- Lasso Regression
- Cross Validation
- Hyperparameter Tuning
- Learning Curves
- Model Evaluation Metrics (R², MAE, MSE, RMSE)

---

## Dipesh Bante 
Created as part of a Machine Learning learning project using Python and Scikit-Learn.
