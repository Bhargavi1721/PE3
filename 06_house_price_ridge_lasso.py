"""Compare Ridge and Lasso regression for house prices."""
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


RANDOM_STATE = 41


def create_house_data():
    """Create correlated property features with some irrelevant variables."""
    return make_regression(
        n_samples=700,
        n_features=14,
        n_informative=8,
        noise=25,
        random_state=RANDOM_STATE,
    )


def split_data(features, prices):
    """Reserve test data for the final comparison."""
    return train_test_split(
        features,
        prices,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )


def evaluate_model(name, estimator, X_train, X_test, y_train, y_test):
    """Scale features, fit one regularized model, and report its results."""
    model = make_pipeline(StandardScaler(), estimator)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    coefficients = model[-1].coef_
    nonzero = np.count_nonzero(np.abs(coefficients) > 1e-6)
    print(f"{name} MAE: {mae:.2f}")
    print(f"{name} RMSE: {rmse:.2f}")
    print(f"{name} R2 score: {r2:.3f}")
    print(f"{name} non-zero coefficients: {nonzero}/{len(coefficients)}")
    return model, predictions


def main():
    features, prices = create_house_data()
    X_train, X_test, y_train, y_test = split_data(features, prices)
    print("House Price Prediction with Ridge and Lasso")
    print("=" * 44)
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    ridge, ridge_predictions = evaluate_model(
        "Ridge", Ridge(alpha=10.0), X_train, X_test, y_train, y_test
    )
    lasso, lasso_predictions = evaluate_model(
        "Lasso", Lasso(alpha=2.0, max_iter=10000), X_train, X_test, y_train, y_test
    )
    print(f"Ridge example prediction: {ridge_predictions[0]:.1f}")
    print(f"Lasso example prediction: {lasso_predictions[0]:.1f}")


if __name__ == "__main__":
    main()
