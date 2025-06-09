import os
from tqdm import tqdm
from sklearn.metrics import multilabel_confusion_matrix, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from enum import Enum
import numpy as np
from results import classify_knee_condition_multiview  # Ensure this is correct
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


# Define class mapping
label_map = {
    "normal": 0,
    "acl": 1,
    "meniscus": 2
}
class_names = ["normal", "acl", "meniscus"]

test_root = "F:\\Graduation_Project\\test"
views = ["axial", "coronal", "sagittal"]

y_true = []
y_pred = []

for class_name in class_names:
    axial_class_dir = os.path.join(test_root, "axial", class_name)
    if not os.path.exists(axial_class_dir):
        continue

    file_list = [f for f in os.listdir(axial_class_dir) if f.lower().endswith(".png")]
    pbar = tqdm(file_list, desc=f"Processing {class_name}")

    for filename in pbar:
        pbar.set_postfix_str(f"Image: {filename}")

        base_id = filename.replace("a", "").replace(".png", "")

        image_path_dict = {}
        missing = False
        for view in views:
            prefix = view[0]
            view_filename = f"{prefix}{base_id}.png"
            view_path = os.path.join(test_root, view, class_name, view_filename)
            if not os.path.exists(view_path):
                missing = True
                break
            image_path_dict[view] = view_path

        if missing:
            continue

        try:
            predicted_status = classify_knee_condition_multiview(image_path_dict)
            predicted_label_str = predicted_status["Diagnosis"].lower()

            true_labels = [label_map[class_name]] if class_name != "normal" else [label_map["normal"]]

            if predicted_label_str == "acl and meniscus":
                pred_labels = [label_map["acl"], label_map["meniscus"]]
            elif predicted_label_str in label_map:
                pred_labels = [label_map[predicted_label_str]]
            else:
                print(f"Skipping ambiguous prediction: {predicted_label_str}")
                continue

            y_true.append(true_labels)
            y_pred.append(pred_labels)

        except Exception as e:
            print(f"Error processing sample {filename}: {e}")


# === Multi-label Binarization ===
mlb = MultiLabelBinarizer(classes=[0, 1, 2])
y_true_bin = mlb.fit_transform(y_true)
y_pred_bin = mlb.transform(y_pred)

# === Binary Evaluation: Normal vs ACL and Normal vs Meniscus ===
def labels_to_binary(y_bin, pos_label_idx):
    """Convert multilabel binarized output to binary labels for one-vs-rest."""
    return [int(row[pos_label_idx]) for row in y_bin]

# === 1. Normal vs ACL ===
print("\n--- Normal vs ACL ---")
normal_vs_acl_idx = [i for i, row in enumerate(y_true) if set(row).issubset({0, 1})]
y_true_acl_bin = [y_true_bin[i] for i in normal_vs_acl_idx]
y_pred_acl_bin = [y_pred_bin[i] for i in normal_vs_acl_idx]

true_acl = labels_to_binary(y_true_acl_bin, pos_label_idx=1)
pred_acl = labels_to_binary(y_pred_acl_bin, pos_label_idx=1)

print("Confusion Matrix (Normal=0, ACL=1):")
print(confusion_matrix(true_acl, pred_acl))
print("\nClassification Report:")
print(classification_report(true_acl, pred_acl, target_names=["normal", "acl"]))

# Confusion Matrix Heatmap: Normal vs ACL
cm_acl = confusion_matrix(true_acl, pred_acl)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_acl, annot=True, fmt="d", cmap="Blues", xticklabels=["normal", "acl"], yticklabels=["normal", "acl"])
plt.title("Confusion Matrix: Normal vs ACL")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# AUC for Normal vs ACL
try:
    auc_acl = roc_auc_score(true_acl, pred_acl)
    print(f"AUC (Normal vs ACL): {auc_acl:.3f}")
except ValueError as e:
    print(f"Cannot compute AUC (Normal vs ACL): {e}")

# === 2. Normal vs Meniscus ===
print("\n--- Normal vs Meniscus ---")
normal_vs_meniscus_idx = [i for i, row in enumerate(y_true) if set(row).issubset({0, 2})]
y_true_meniscus_bin = [y_true_bin[i] for i in normal_vs_meniscus_idx]
y_pred_meniscus_bin = [y_pred_bin[i] for i in normal_vs_meniscus_idx]

true_meniscus = labels_to_binary(y_true_meniscus_bin, pos_label_idx=2)
pred_meniscus = labels_to_binary(y_pred_meniscus_bin, pos_label_idx=2)

print("Confusion Matrix (Normal=0, Meniscus=1):")
print(confusion_matrix(true_meniscus, pred_meniscus))
print("\nClassification Report:")
print(classification_report(true_meniscus, pred_meniscus, target_names=["normal", "meniscus"]))

# Confusion Matrix Heatmap: Normal vs Meniscus
cm_meniscus = confusion_matrix(true_meniscus, pred_meniscus)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_meniscus, annot=True, fmt="d", cmap="Greens", xticklabels=["normal", "meniscus"], yticklabels=["normal", "meniscus"])
plt.title("Confusion Matrix: Normal vs Meniscus")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# AUC for Normal vs Meniscus
try:
    auc_meniscus = roc_auc_score(true_meniscus, pred_meniscus)
    print(f"AUC (Normal vs Meniscus): {auc_meniscus:.3f}")
except ValueError as e:
    print(f"Cannot compute AUC (Normal vs Meniscus): {e}")
