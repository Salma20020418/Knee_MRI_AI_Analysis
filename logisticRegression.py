import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Define views and conditions
views = ["axial", "coronal", "sagittal"]
conditions = ["acl", "meniscus"]

# Ensure output directory exists
os.makedirs("weights", exist_ok=True)

# Dictionaries for collecting intercepts and weights
intercepts_dict = {}
weights_dict = {condition: [] for condition in conditions}  # scalar norms
rescaled_intercepts_by_condition = {condition: [] for condition in conditions}

# Training loop
for view in views:
    for condition in conditions:
        print(f"\n--- Training Logistic Regression for {view.upper()} - {condition.upper()} ---")

        try:
            # Load features and labels
            train_features_path = f"features/train_{view}_{condition}_features.npy"
            train_labels_path = f"features/train_{view}_{condition}_labels.npy"
            valid_features_path = f"features/valid_{view}_{condition}_features.npy"
            valid_labels_path = f"features/valid_{view}_{condition}_labels.npy"

            X_train = np.load(train_features_path)
            y_train = np.load(train_labels_path)
            X_valid = np.load(valid_features_path)
            y_valid = np.load(valid_labels_path)

            X_train = X_train.reshape(X_train.shape[0], -1)
            X_valid = X_valid.reshape(X_valid.shape[0], -1)

            if X_train.shape[0] != y_train.shape[0]:
                raise ValueError(f"Mismatch: {X_train.shape[0]} train samples vs {y_train.shape[0]} labels")
            if X_valid.shape[0] != y_valid.shape[0]:
                raise ValueError(f"Mismatch: {X_valid.shape[0]} valid samples vs {y_valid.shape[0]} labels")

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_valid = scaler.transform(X_valid)

            param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
            base_model = LogisticRegression(max_iter=5000, solver='lbfgs', class_weight='balanced', random_state=42)
            grid = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy')
            grid.fit(X_train, y_train)
            clf = grid.best_estimator_

            # Save weights and intercepts
            weights = clf.coef_
            np.save(f"weights/{view}_{condition}_weights.npy", weights)
            intercepts_dict[(view, condition)] = clf.intercept_[0]
            weights_dict[condition].append(np.linalg.norm(weights))

            acc = accuracy_score(y_valid, clf.predict(X_valid))
            print(f"Best C: {clf.C}")
            print(f"Validation Accuracy: {acc * 100:.2f}%")
            print(f"Weights:\n{weights}")
            print(f"Raw Intercept:\n{clf.intercept_}")

        except FileNotFoundError as e:
            print(f"Missing file for {view} - {condition}: {e.filename}")
        except Exception as e:
            print(f"Error during training for {view} - {condition}: {e}")

# Rescale intercepts to (-1, 0)
intercept_values = list(intercepts_dict.values())
min_old = min(intercept_values)
max_old = max(intercept_values)
new_min, new_max = -1, 0

print("\n--- Rescaled Intercepts ---")
for (view, condition), raw_intercept in intercepts_dict.items():
    if max_old != min_old:
        scaled = new_min + ((raw_intercept - min_old) * (new_max - new_min)) / (max_old - min_old)
    else:
        scaled = -0.5
    rescaled_intercepts_by_condition[condition].append(scaled)
    np.save(f"weights/{view}_{condition}_intercept.npy", np.array([scaled]))
    print(f"{view.upper()} - {condition.upper()} intercept rescaled: {scaled:.4f}")

# Normalize weights and calculate single intercept per condition
print("\n--- Final Combined Weights and Intercepts ---")
final_weights = {}
final_intercepts = {}

for condition in conditions:
    norm_weights = np.array(weights_dict[condition])
    total = norm_weights.sum()
    if total == 0:
        normed = np.zeros_like(norm_weights)
    else:
        normed = norm_weights / total
    normed = np.round(normed, 3)
    final_weights[condition] = normed

    intercept_avg = round(np.mean(rescaled_intercepts_by_condition[condition]), 3)
    final_intercepts[condition] = intercept_avg

    print(f"logreg_{condition}_weights = np.array({normed.tolist()})")
    print(f"{condition}_intercept = {intercept_avg}")
