"""Predict car prices using a gradient boosting regression model."""
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np


RANDOM_STATE = 61


def create_car_data():
	"""Create numeric vehicle features such as age, mileage, and engine size."""
	return make_regression(
		n_samples=900,
		n_features=9,
		n_informative=7,
		noise=16,
		random_state=RANDOM_STATE,
	)


def split_data(features, prices):
	"""Keep a final holdout set for an honest price estimate."""
	return train_test_split(
		features,
		prices,
		test_size=0.2,
		random_state=RANDOM_STATE,
	)


def train_model(features, prices):
	"""Fit boosting stages that learn from the previous stage's errors."""
	model = GradientBoostingRegressor(
		n_estimators=180,
		learning_rate=0.06,
		max_depth=3,
		random_state=RANDOM_STATE,
	)
	model.fit(features, prices)
	return model


def evaluate_model(model, features, prices):
	"""Report average, squared, and relative prediction errors."""
	predictions = model.predict(features)
	mae = mean_absolute_error(prices, predictions)
	rmse = np.sqrt(mean_squared_error(prices, predictions))
	print(f"MAE: {mae:.2f} price units")
	print(f"RMSE: {rmse:.2f} price units")
	print(f"R2 score: {r2_score(prices, predictions):.3f}")
	return predictions


def main():
	features, prices = create_car_data()
	X_train, X_test, y_train, y_test = split_data(features, prices)
	model = train_model(X_train, y_train)
	print("Car Price Prediction with Regression")
	print("=" * 38)
	print(f"Training cars: {len(X_train)}")
	print(f"Testing cars: {len(X_test)}")
	predictions = evaluate_model(model, X_test, y_test)
	print(f"Example predicted price index: {predictions[0]:.1f}")
	print(f"Example actual price index: {y_test[0]:.1f}")


if __name__ == "__main__":
	main()
