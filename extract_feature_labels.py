from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
from dataloader import train_data, valid_data, device
from modelBackbone import modelBackbone, TransferLearningModel # Ensure this imports your trained model
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def get_loaders(data_root, batch_size=32, img_size=256, shuffle=True):
    """
    Given a data root (e.g. root_folder/axial/acl/train/),
    return a DataLoader for images in {condition}/ and normal/ subfolders.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # Add normalization if needed
    ])

    dataset = datasets.ImageFolder(data_root, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, dataset.classes  # classes like ['acl', 'normal']


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

views = ["axial", "coronal", "sagittal"]
conditions = ["ACL", "meniscus"]

root_folder = "data_set"  # CHANGE to your actual root folder path
batch_size = 32

def extract_features(model, dataloader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader, desc="Extracting features", leave=False):
            imgs = imgs.to(device)
            out = model(imgs)
            features.append(out.cpu().numpy())
            labels.append(lbls.numpy())
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    return features, labels

for view in views:
    for condition in conditions:
        print(f"Processing {view} - {condition}")
        model_path = f"{condition}_{view}.pth"
        train_path = os.path.join(root_folder, view, condition, "train")
        valid_path = os.path.join(root_folder, view, condition, "valid")

        model = TransferLearningModel(model_type="resnet50", output_shape=1, dropout_rate=0).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        train_loader, train_classes = get_loaders(train_path, batch_size=batch_size)
        valid_loader, valid_classes = get_loaders(valid_path, batch_size=batch_size, shuffle=False)

        print("Extracting train features...")
        train_features, train_labels = extract_features(model, train_loader)

        print("Extracting valid features...")
        valid_features, valid_labels = extract_features(model, valid_loader)

        # Before saving files
        os.makedirs("features", exist_ok=True)

        np.save(f"features/train_{view}_{condition}_features.npy", train_features)
        np.save(f"features/train_{view}_{condition}_labels.npy", train_labels)
        np.save(f"features/valid_{view}_{condition}_features.npy", valid_features)
        np.save(f"features/valid_{view}_{condition}_labels.npy", valid_labels)


        print(f"Saved features for {view} - {condition}\n")
