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

# Collect intercepts to rescale later
intercepts_dict = {}

# First pass: train and collect raw intercepts
for view in views:
    for condition in conditions:
        print(f"\n--- Training Logistic Regression for {view.upper()} - {condition.upper()} ---")

        try:
            # File paths
            train_features_path = f"features/train_{view}_{condition}_features.npy"
            train_labels_path = f"features/train_{view}_{condition}_labels.npy"
            valid_features_path = f"features/valid_{view}_{condition}_features.npy"
            valid_labels_path = f"features/valid_{view}_{condition}_labels.npy"

            # Load features and labels
            X_train = np.load(train_features_path)
            y_train = np.load(train_labels_path)
            X_valid = np.load(valid_features_path)
            y_valid = np.load(valid_labels_path)

            # Flatten to 2D (samples x features)
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_valid = X_valid.reshape(X_valid.shape[0], -1)

            # Sanity checks
            if X_train.shape[0] != y_train.shape[0]:
                raise ValueError(f"Training features/labels mismatch: {X_train.shape[0]} vs {y_train.shape[0]}")
            if X_valid.shape[0] != y_valid.shape[0]:
                raise ValueError(f"Validation features/labels mismatch: {X_valid.shape[0]} vs {y_valid.shape[0]}")

            # Standardize features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_valid = scaler.transform(X_valid)

            # Grid search over C
            param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
            base_model = LogisticRegression(
                max_iter=5000,
                solver='lbfgs',
                class_weight='balanced',
                random_state=42
            )
            grid = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy')
            grid.fit(X_train, y_train)
            clf = grid.best_estimator_

            # Save weights
            weights = clf.coef_
            np.save(f"weights/{view}_{condition}_weights.npy", weights)

            # Store intercept for later rescaling
            intercepts_dict[(view, condition)] = clf.intercept_[0]

            # Validate
            y_pred = clf.predict(X_valid)
            acc = accuracy_score(y_valid, y_pred)
            print(f"Best C: {clf.C}")
            print(f"Validation Accuracy: {acc * 100:.2f}%")
            print(f"Weights:\n{weights}")
            print(f"Raw Intercept:\n{clf.intercept_}")

        except FileNotFoundError as e:
            print(f"Missing file for {view} - {condition}: {e.filename}")
        except Exception as e:
            print(f"Error during training for {view} - {condition}: {e}")

# Second pass: Rescale all intercepts to (-1, 0)
intercept_values = list(intercepts_dict.values())
min_old = min(intercept_values)
max_old = max(intercept_values)
new_min = -1
new_max = 0

print("\n--- Rescaled Intercepts ---")
for (view, condition), raw_intercept in intercepts_dict.items():
    # Linear rescaling
    if max_old != min_old:
        scaled_intercept = new_min + ((raw_intercept - min_old) * (new_max - new_min)) / (max_old - min_old)
    else:
        scaled_intercept = -0.5  # Default if all intercepts are the same

    # Save and print
    np.save(f"weights/{view}_{condition}_intercept.npy", np.array([scaled_intercept]))
    print(f"{view.upper()} - {condition.upper()} intercept rescaled: {scaled_intercept:.4f}")
