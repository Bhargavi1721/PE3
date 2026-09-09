"""Student performance prediction using RandomizedSearchCV."""
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import numpy as np


RANDOM_STATE = 42


def create_student_data():
	"""Create features representing study, attendance, and assessment habits."""
	features, target = make_regression(
		n_samples=600,
		n_features=8,
		n_informative=6,
		noise=12,
		random_state=RANDOM_STATE,
	)
	return features, target


def split_data(features, target):
	"""Keep a final test set separate from cross-validation."""
	return train_test_split(
		features,
		target,
		test_size=0.2,
		random_state=RANDOM_STATE,
	)


def tune_model(features, target):
	"""Search several forest settings using four-fold cross-validation."""
	parameter_grid = {
		"n_estimators": [100, 200],
		"max_depth": [None, 8, 16],
		"min_samples_leaf": [1, 2, 4],
	}
	search = RandomizedSearchCV(
		RandomForestRegressor(random_state=RANDOM_STATE),
		parameter_grid,
		n_iter=6,
		cv=4,
		scoring="neg_mean_absolute_error",
		random_state=RANDOM_STATE,
		n_jobs=-1,
	)
	search.fit(features, target)
	return search


def evaluate_model(model, features, target):
	"""Print useful regression metrics on unseen students."""
	predictions = model.predict(features)
	mae = mean_absolute_error(target, predictions)
	rmse = np.sqrt(mean_squared_error(target, predictions))
	r2 = r2_score(target, predictions)
	print(f"MAE: {mae:.2f}")
	print(f"RMSE: {rmse:.2f}")
	print(f"R2 score: {r2:.3f}")
	return predictions


def main():
	features, target = create_student_data()
	X_train, X_test, y_train, y_test = split_data(features, target)
	search = tune_model(X_train, y_train)
	print("Student Performance Prediction")
	print("=" * 32)
	print(f"Training rows: {len(X_train)}")
	print(f"Testing rows: {len(X_test)}")
	print(f"Best parameters: {search.best_params_}")
	evaluate_model(search, X_test, y_test)
	example_score = search.predict(X_test[:1])[0]
	print(f"Example predicted score: {example_score:.1f}")


if __name__ == "__main__":
	main()
