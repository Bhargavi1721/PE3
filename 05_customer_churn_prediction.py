"""Customer churn prediction using a random forest classifier."""
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 31


def create_customer_data():
	"""Create customer behavior features and a churn label."""
	return make_classification(
		n_samples=1200,
		n_features=10,
		n_informative=6,
		weights=[0.72, 0.28],
		random_state=RANDOM_STATE,
	)


def split_data(features, labels):
	"""Use a stratified split to retain the churn proportion."""
	return train_test_split(
		features,
		labels,
		test_size=0.2,
		stratify=labels,
		random_state=RANDOM_STATE,
	)


def train_model(features, labels):
	"""Fit a balanced forest that can capture non-linear churn patterns."""
	model = RandomForestClassifier(
		n_estimators=220,
		class_weight="balanced",
		random_state=RANDOM_STATE,
		n_jobs=-1,
	)
	model.fit(features, labels)
	return model


def evaluate_model(model, features, labels):
	"""Print classification quality and the most useful feature positions."""
	predictions = model.predict(features)
	probabilities = model.predict_proba(features)[:, 1]
	print(f"Accuracy: {accuracy_score(labels, predictions):.3f}")
	print(f"ROC-AUC: {roc_auc_score(labels, probabilities):.3f}")
	print("Confusion matrix:")
	print(confusion_matrix(labels, predictions))
	print(classification_report(labels, predictions, target_names=["Retained", "Churned"]))
	importance = permutation_importance(model, features, labels, n_repeats=3, random_state=RANDOM_STATE)
	ranked = importance.importances_mean.argsort()[::-1][:3]
	print(f"Top feature positions: {ranked.tolist()}")
	return probabilities


def main():
	features, labels = create_customer_data()
	X_train, X_test, y_train, y_test = split_data(features, labels)
	model = train_model(X_train, y_train)
	print("Customer Churn Prediction")
	print("=" * 26)
	print(f"Overall churn rate: {labels.mean():.2%}")
	print(f"Training customers: {len(X_train)}")
	print(f"Testing customers: {len(X_test)}")
	probabilities = evaluate_model(model, X_test, y_test)
	print(f"Example churn probability: {probabilities[0]:.3f}")


if __name__ == "__main__":
	main()
