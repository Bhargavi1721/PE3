"""House price prediction using GridSearchCV."""
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np


RANDOM_STATE = 7


def create_house_data():
	"""Create numeric property features and a continuous price index."""
	return make_regression(
		n_samples=700,
		n_features=10,
		n_informative=8,
		noise=18,
		random_state=RANDOM_STATE,
	)


def prepare_data(features, prices):
	"""Create a holdout set that is never used during tuning."""
	return train_test_split(
		features,
		prices,
		test_size=0.2,
		random_state=RANDOM_STATE,
	)


def build_tuned_model(features, prices):
	"""Use a grid search to choose boosting depth and learning rate."""
	parameters = {
		"n_estimators": [100, 180],
		"learning_rate": [0.03, 0.08],
		"max_depth": [2, 3],
	}
	search = GridSearchCV(
		GradientBoostingRegressor(random_state=RANDOM_STATE),
		parameters,
		cv=4,
		scoring="neg_mean_absolute_error",
		n_jobs=-1,
		return_train_score=True,
	)
	search.fit(features, prices)
	return search


def report_results(model, features, prices):
	"""Report errors in price units and explain the prediction example."""
	predictions = model.predict(features)
	mae = mean_absolute_error(prices, predictions)
	rmse = np.sqrt(mean_squared_error(prices, predictions))
	print(f"MAE: {mae:.2f}")
	print(f"RMSE: {rmse:.2f}")
	print(f"R2 score: {r2_score(prices, predictions):.3f}")
	return predictions


def main():
	features, prices = create_house_data()
	X_train, X_test, y_train, y_test = prepare_data(features, prices)
	search = build_tuned_model(X_train, y_train)
	print("House Price Prediction with Hyperparameter Tuning")
	print("=" * 49)
	print(f"Training rows: {len(X_train)}")
	print(f"Testing rows: {len(X_test)}")
	print(f"Best parameters: {search.best_params_}")
	predictions = report_results(search, X_test, y_test)
	print(f"Example predicted price index: {predictions[0]:.1f}")


if __name__ == "__main__":
	main()
