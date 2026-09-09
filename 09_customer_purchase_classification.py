"""Predict whether a customer will purchase using a tuned classifier."""
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 71


def create_purchase_data():
	"""Create customer browsing, engagement, and purchase features."""
	return make_classification(
		n_samples=1000,
		n_features=8,
		n_informative=5,
		weights=[0.62, 0.38],
		random_state=RANDOM_STATE,
	)


def split_data(features, labels):
	"""Keep the purchase ratio consistent in training and testing."""
	return train_test_split(
		features,
		labels,
		test_size=0.2,
		stratify=labels,
		random_state=RANDOM_STATE,
	)


def tune_classifier(features, labels):
	"""Tune SVM regularization and kernel scale with cross-validation."""
	pipeline = make_pipeline(
		StandardScaler(),
		SVC(probability=True, random_state=RANDOM_STATE),
	)
	parameters = {
		"svc__C": [0.5, 1, 2],
		"svc__gamma": ["scale", 0.05],
	}
	search = GridSearchCV(
		pipeline,
		parameters,
		cv=4,
		scoring="roc_auc",
		n_jobs=-1,
	)
	search.fit(features, labels)
	return search


def evaluate_classifier(model, features, labels):
	"""Print threshold metrics, ranking quality, and errors by class."""
	predictions = model.predict(features)
	probabilities = model.predict_proba(features)[:, 1]
	print(f"Accuracy: {accuracy_score(labels, predictions):.3f}")
	print(f"ROC-AUC: {roc_auc_score(labels, probabilities):.3f}")
	print("Confusion matrix:")
	print(confusion_matrix(labels, predictions))
	print(classification_report(labels, predictions, target_names=["No purchase", "Purchase"]))
	return probabilities


def main():
	features, labels = create_purchase_data()
	X_train, X_test, y_train, y_test = split_data(features, labels)
	search = tune_classifier(X_train, y_train)
	print("Customer Purchase Prediction")
	print("=" * 30)
	print(f"Purchase rate: {labels.mean():.2%}")
	print(f"Training customers: {len(X_train)}")
	print(f"Testing customers: {len(X_test)}")
	print(f"Best parameters: {search.best_params_}")
	probabilities = evaluate_classifier(search, X_test, y_test)
	print(f"Example purchase probability: {probabilities[0]:.3f}")


if __name__ == "__main__":
	main()
