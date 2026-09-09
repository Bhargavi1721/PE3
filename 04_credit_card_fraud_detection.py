"""Credit-card fraud detection with class weighting and a thresholded score."""
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 21
FRAUD_THRESHOLD = 0.50


def create_transaction_data():
	"""Create an intentionally imbalanced transaction dataset."""
	return make_classification(
		n_samples=2500,
		n_features=12,
		n_informative=8,
		weights=[0.97, 0.03],
		flip_y=0.01,
		random_state=RANDOM_STATE,
	)


def split_data(features, labels):
	"""Preserve the rare fraud class in both train and test data."""
	return train_test_split(
		features,
		labels,
		test_size=0.25,
		stratify=labels,
		random_state=RANDOM_STATE,
	)


def train_detector(features, labels):
	"""Balance the logistic loss so fraud cases are not ignored."""
	detector = make_pipeline(
		StandardScaler(),
		LogisticRegression(
			class_weight="balanced",
			max_iter=1000,
			random_state=RANDOM_STATE,
		),
	)
	detector.fit(features, labels)
	return detector


def evaluate_detector(detector, features, labels):
	"""Use ranking metrics because accuracy is misleading for rare fraud."""
	probabilities = detector.predict_proba(features)[:, 1]
	predictions = (probabilities >= FRAUD_THRESHOLD).astype(int)
	print(f"ROC-AUC: {roc_auc_score(labels, probabilities):.3f}")
	print(f"PR-AUC: {average_precision_score(labels, probabilities):.3f}")
	print("Confusion matrix:")
	print(confusion_matrix(labels, predictions))
	print(classification_report(labels, predictions, target_names=["Legitimate", "Fraud"]))
	return probabilities


def main():
	features, labels = create_transaction_data()
	X_train, X_test, y_train, y_test = split_data(features, labels)
	detector = train_detector(X_train, y_train)
	print("Credit Card Fraud Detection")
	print("=" * 29)
	print(f"Fraud rate in full data: {labels.mean():.2%}")
	print(f"Training transactions: {len(X_train)}")
	print(f"Testing transactions: {len(X_test)}")
	probabilities = evaluate_detector(detector, X_test, y_test)
	print(f"Example fraud probability: {probabilities[0]:.3f}")


if __name__ == "__main__":
	main()
