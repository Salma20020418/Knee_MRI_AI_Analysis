from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pathlib import Path
import torch
from PIL import Image
import io
import numpy as np
import cv2
import torchvision.transforms as transforms
from modelBackbone import TransferLearningModel
import torch.nn.functional as F
from enum import Enum
from io import BytesIO
import os

app = FastAPI()

# Enum for Diagnosis
class Status(Enum):
    ACL = "ACL"
    ACL_MENIISCUS = "ACL and meniscus"
    MENISCUS = "meniscus"
    NORMAL = "normal"

# Device setup (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
acl_models = {
    "sagittal": TransferLearningModel(model_type="resnet50", output_shape=1, dropout_rate=0).to(device),
    "coronal": TransferLearningModel(model_type="resnet50", output_shape=1, dropout_rate=0).to(device),
    "axial": TransferLearningModel(model_type="resnet50", output_shape=1, dropout_rate=0).to(device)
}

meniscus_models = {
    "sagittal": TransferLearningModel(model_type="resnet50", output_shape=1, dropout_rate=0).to(device),
    "coronal": TransferLearningModel(model_type="resnet50", output_shape=1, dropout_rate=0).to(device),
    "axial": TransferLearningModel(model_type="resnet50", output_shape=1, dropout_rate=0).to(device)
}

# Load models' state dicts
for view in ["sagittal", "coronal", "axial"]:
    acl_models[view].load_state_dict(torch.load(f"acl_{view}.pth", map_location=device))
    meniscus_models[view].load_state_dict(torch.load(f"meniscus_{view}.pth", map_location=device))

# Set models to evaluation mode
for model in acl_models.values():
    model.eval()
for model in meniscus_models.values():
    model.eval()

# Transformation for image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def generate_heatmap(image_tensor, model):
    output = model(image_tensor)
    model.zero_grad()
    output[0].backward()

    gradients = model.get_activations_gradient()
    activations = model.get_activations()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()
    return heatmap.cpu().detach().numpy()

def classify_knee_condition(image: Image, view_type: str):
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Get raw model outputs
    acl_output = acl_models[view_type](image_tensor)
    meniscus_output = meniscus_models[view_type](image_tensor)

    # Sigmoid probabilities
    acl_prob = torch.sigmoid(acl_output).item()
    meniscus_prob = torch.sigmoid(meniscus_output).item()

    # Binary classification based on threshold
    acl_result = 0 if acl_prob < 0.3 else 1
    meniscus_result = 0 if meniscus_prob < 0.3 else 1

    # Determine status
    if acl_result == 0 and meniscus_result == 0:
        status = Status.ACL_MENIISCUS
    elif acl_result == 0:
        status = Status.ACL
    elif meniscus_result == 0:
        status = Status.MENISCUS
    else:
        status = Status.NORMAL

    return status.value

@app.post("/predict/")
async def predict(files: List[UploadFile] = File(...), view_type: str = "sagittal"):
    try:
        # Validate view_type input
        if view_type not in acl_models:
            return JSONResponse(status_code=400, content={"error": "Invalid view_type. Choose from 'sagittal', 'coronal', 'axial'."})

        results = []
        
        for file in files:
            # Read image from uploaded file
            image_bytes = await file.read()
            image = Image.open(BytesIO(image_bytes))
            
            # Classify the image
            diagnosis = classify_knee_condition(image, view_type)
            
            results.append({"filename": file.filename, "diagnosis": diagnosis})

        return {"predictions": results}
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
