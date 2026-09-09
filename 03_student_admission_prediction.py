"""Student admission prediction with a scaled logistic classifier."""
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 11


def create_admission_data():
	"""Create features representing academic and application information."""
	return make_classification(
		n_samples=800,
		n_features=7,
		n_informative=5,
		weights=[0.42, 0.58],
		random_state=RANDOM_STATE,
	)


def split_data(features, labels):
	"""Use stratification so both classes remain represented in each split."""
	return train_test_split(
		features,
		labels,
		test_size=0.2,
		stratify=labels,
		random_state=RANDOM_STATE,
	)


def train_model(features, labels):
	"""Scale numeric values before fitting logistic regression."""
	model = make_pipeline(
		StandardScaler(),
		LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
	)
	model.fit(features, labels)
	return model


def evaluate_model(model, features, labels):
	"""Print threshold metrics and probability ranking quality."""
	predictions = model.predict(features)
	probabilities = model.predict_proba(features)[:, 1]
	print(f"Accuracy: {accuracy_score(labels, predictions):.3f}")
	print(f"ROC-AUC: {roc_auc_score(labels, probabilities):.3f}")
	print("Confusion matrix:")
	print(confusion_matrix(labels, predictions))
	print(classification_report(labels, predictions, target_names=["Not admitted", "Admitted"]))
	return predictions, probabilities


def main():
	features, labels = create_admission_data()
	X_train, X_test, y_train, y_test = split_data(features, labels)
	model = train_model(X_train, y_train)
	print("Student Admission Prediction")
	print("=" * 31)
	print(f"Training rows: {len(X_train)}")
	print(f"Testing rows: {len(X_test)}")
	evaluate_model(model, X_test, y_test)
	admission_probability = model.predict_proba(X_test[:1])[0, 1]
	print(f"Example admission probability: {admission_probability:.3f}")


if __name__ == "__main__":
	main()
