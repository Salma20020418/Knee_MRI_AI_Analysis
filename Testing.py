from pathlib import Path
import torchvision
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from modelBackbone import modelBackbone
from dataloaderPngT import class_names
from typing import List
import numpy as np
import cv2

# Setup custom folder path (include full path to the directory)
image_folder_path = Path("F:\\Graduation_Project\\Sample\\Sara")

# Check if the folder exists
if image_folder_path.is_dir():
    print(f"{image_folder_path} exists, ready for use.")
else:
    print(f"{image_folder_path} does not exist. Please check the folder path.")

# Create transform pipeline to resize image
custom_image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
])


def slice_image_by_edges(image_path, min_slice_dim=80, top_margin=50):
    """
    Slices the input MRI image into a grid and passes each slice individually for predictions.
    """
    # Load the image
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    # Resize the image if necessary (optional step)
    max_width = 1000
    max_height = 1000
    height, width = image.shape[:2]

    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)  # Calculate scaling factor
        new_dim = (int(width * scale), int(height * scale))  # New dimensions after scaling
        image = cv2.resize(image, new_dim)  # Resize the image
    
    print(f"Loaded image: {image_path.name}")

    # Display the image for cropping
    roi = cv2.selectROI("Select Region to Crop", image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    if roi[2] == 0 or roi[3] == 0:
        print("No region selected, proceeding with the full image.")
        roi = (0, 0, image.shape[1], image.shape[0])

    x, y, w, h = roi
    cropped_image = image[y:y+h, x:x+w]

    print(f"Cropped image size: {cropped_image.shape[:2]}")
    rows = int(input(f"Enter the number of rows for the cropped image (min {min_slice_dim}): "))
    cols = int(input(f"Enter the number of columns for the cropped image (min {min_slice_dim}): "))

    height, width, _ = cropped_image.shape
    slice_height = height // rows
    slice_width = width // cols

    # Extract slices and pass them individually for predictions
    for row in range(rows):
        for col in range(cols):
            y_start, y_end = row * slice_height, (row + 1) * slice_height
            x_start, x_end = col * slice_width, (col + 1) * slice_width

            slice_img = cropped_image[y_start:y_end, x_start:x_end]
            slice_img = cv2.resize(slice_img, (256, 256))  # Ensure consistent size
            
            # Save slice temporarily
            temp_slice_path = f"temp_slice_{row}_{col}.jpg"
            cv2.imwrite(temp_slice_path, slice_img)
            
            # Pass the slice for prediction
            pred_and_plot_image(model=modelBackbone,
                                image_path=temp_slice_path,
                                class_names=class_names,
                                transform=custom_image_transform)


def avg_predictions(model: torch.nn.Module, 
                    image_path: str, 
                    class_names: List[str] = None, 
                    transform=None, 
                    num_preds: int = 10):
    """ Averages the predictions of multiple forward passes to get a final result. """
    
    model.eval()
    all_preds = []

    for _ in range(num_preds):
        # 1. Load the image and convert to float32
        target_image = torchvision.io.read_image(str(image_path), mode=torchvision.io.ImageReadMode.GRAY).type(torch.float32)
        
        # 2. Normalize the image pixel values
        target_image = target_image / 255.0
        
        # 3. Apply transformation if provided
        if transform:
            target_image = transform(target_image)

        # 4. Convert grayscale to 3 channels
        if target_image.shape[0] == 1:  # Check if it's a single channel (grayscale)
            target_image = target_image.repeat(3, 1, 1)  # Make it 3 channels (RGB)

        # 5. Add an extra dimension to match the batch size
        target_image = target_image.unsqueeze(dim=0).to(next(model.parameters()).device)

        # 6. Make prediction
        with torch.inference_mode():
            target_image_pred = model(target_image)

        # 7. Convert logits -> prediction probabilities
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

        # Collect all predictions
        all_preds.append(target_image_pred_probs.cpu().detach().numpy())

    # 8. Average the predictions
    avg_preds = np.mean(all_preds, axis=0)
    
    # 9. Threshold the averaged predictions to get final labels
    final_preds = np.argmax(avg_preds, axis=1)

    # 10. Get class names for predicted labels
    predicted_classes = [class_names[final_preds[0]]]

    # Exclude 'normal' if other classes are predicted
    if any(cls in predicted_classes for cls in ['meniscus', 'acl', 'abnormal']):
        predicted_classes = [cls for cls in predicted_classes if cls != 'normal']

    return predicted_classes, avg_preds


def pred_and_plot_image(model: torch.nn.Module, 
                        image_path: str, 
                        class_names: List[str] = None, 
                        transform=None):
    # 1. Get averaged predictions and probabilities
    predicted_classes, avg_preds = avg_predictions(model, 
                                                   image_path, 
                                                   class_names=class_names, 
                                                   transform=transform)
    
    # 2. Load and plot the image
    target_image = torchvision.io.read_image(str(image_path), mode=torchvision.io.ImageReadMode.GRAY).type(torch.float32)
    target_image = target_image / 255.0  # Normalize to [0, 1]
    if target_image.shape[0] == 1:
        target_image = target_image.repeat(3, 1, 1)  # Convert to RGB if grayscale
    
    # 3. Plot the grayscale image with predictions
    plt.imshow(target_image.squeeze().cpu()[0], cmap='gray')  # Show only the first channel for grayscale
    title = f"Pred: {', '.join(predicted_classes) if predicted_classes else 'None'} | Max Prob: {avg_preds.max():.3f}"
    plt.title(title)
    plt.axis(False)
    plt.show()

    # 4. Print the class-wise prediction probabilities
    print(f"Predictions for {image_path}:")
    for i, class_name in enumerate(class_names):
        prob = avg_preds[0][i]  # Probability for each class
        print(f"{class_name}: {prob:.3f}")

# Move the model to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelBackbone.to(device)

# Iterate over all images in the specified folder and make predictions
for image_file in image_folder_path.glob("*.jpg"):  # Change the file extension as needed
    slice_image_by_edges(image_file)
    pred_and_plot_image(model=modelBackbone,
                        image_path=image_file,
                        class_names=class_names,
                        transform=custom_image_transform)


