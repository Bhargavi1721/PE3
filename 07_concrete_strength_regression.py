"""Predict concrete compressive strength with a regression model."""
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np


RANDOM_STATE = 51


def create_concrete_data():
	"""Create eight numeric mix-design features and strength in MPa."""
	return make_regression(
		n_samples=1030,
		n_features=8,
		n_informative=7,
		noise=7,
		random_state=RANDOM_STATE,
	)


def split_data(features, strengths):
	"""Separate the data into training and test portions."""
	return train_test_split(
		features,
		strengths,
		test_size=0.2,
		random_state=RANDOM_STATE,
	)


def train_model(features, strengths):
	"""Train a forest that can model non-linear material interactions."""
	model = RandomForestRegressor(
		n_estimators=250,
		min_samples_leaf=2,
		random_state=RANDOM_STATE,
		n_jobs=-1,
	)
	model.fit(features, strengths)
	return model


def evaluate_model(model, features, strengths):
	"""Print error in MPa and the explained variance score."""
	predictions = model.predict(features)
	mae = mean_absolute_error(strengths, predictions)
	rmse = np.sqrt(mean_squared_error(strengths, predictions))
	print(f"MAE: {mae:.2f} MPa")
	print(f"RMSE: {rmse:.2f} MPa")
	print(f"R2 score: {r2_score(strengths, predictions):.3f}")
	return predictions


def main():
	features, strengths = create_concrete_data()
	X_train, X_test, y_train, y_test = split_data(features, strengths)
	model = train_model(X_train, y_train)
	print("Concrete Compressive Strength Prediction")
	print("=" * 42)
	print(f"Training samples: {len(X_train)}")
	print(f"Testing samples: {len(X_test)}")
	predictions = evaluate_model(model, X_test, y_test)
	print(f"Example predicted strength: {predictions[0]:.1f} MPa")
	print(f"Example actual strength: {y_test[0]:.1f} MPa")


if __name__ == "__main__":
	main()
