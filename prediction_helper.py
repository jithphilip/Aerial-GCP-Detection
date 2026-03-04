import os
import glob
import json
import cv2
import torch
import torch.nn as nn
from torchvision import models

# --- Rebuilding the Architecture ---
# PyTorch requires the exact blueprint of the model to load the weights properly
class GCPDetector(nn.Module):
    def __init__(self):
        super(GCPDetector, self).__init__()
        # We don't need to download pretrained weights here, we are about to load our own
        self.backbone = models.resnet18(weights=None) 
        self.backbone.fc = nn.Identity()
        
        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3) 
        )

    def forward(self, x):
        features = self.backbone(x)
        coords = self.regressor(features)
        shape_logits = self.classifier(features)
        return coords, shape_logits

# --- Setup and Loading Weights ---
# Automatically using GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running inference on: {device}")

# Initializing the empty blueprint and fill it with our trained weights
model = GCPDetector().to(device)
weights_path = 'gcp_detector_weights_balanced.pth'

if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Could not find the weights file: {weights_path}")

model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval() # CRITICAL: Turns off dropout layers for deterministic predictions

# --- Prepares the Data Pipeline ---
test_dir = 'test_dataset'
id_to_shape = {0: "Cross", 1: "Square", 2: "L-Shaped"}
predictions = {}

# Recursively finding all images
test_images = []
for ext in ('*.JPG', '*.jpg', '*.jpeg', '*.JPEG'):
    test_images.extend(glob.glob(os.path.join(test_dir, '**', ext), recursive=True))

print(f"Found images. Generating predictions...")

# --- The Inference Loop ---
with torch.no_grad(): # Disables gradient calculation to save memory and speed up
    for img_path in test_images:
        
        # Formating the relative path exactly as the JSON expects (forward slashes)
        rel_path = os.path.relpath(img_path, test_dir).replace('\\', '/')
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}. Skipping.")
            continue
            
        orig_h, orig_w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Standardizing the input size
        img_resized = cv2.resize(img_rgb, (512, 512))
        
        # Converting to PyTorch tensor format: [Batch, Channels, Height, Width]
        img_tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Prediction
        pred_coords, pred_shapes = model(img_tensor)
        
        # Extracting and un-normalize the coordinates back to the original image scale
        x_norm = pred_coords[0, 0].item()
        y_norm = pred_coords[0, 1].item()
        x_final = round(x_norm * orig_w, 1)
        y_final = round(y_norm * orig_h, 1)
        
        # Extracting the shape prediction
        shape_idx = torch.argmax(pred_shapes, dim=1).item()
        shape_final = id_to_shape[shape_idx]
        
        # Stores in dictionary
        predictions[rel_path] = {
            "mark": {"x": x_final, "y": y_final},
            "verified_shape": shape_final
        }

# --- Saving the Output ---
output_file = 'predictions.json'
with open(output_file, 'w') as f:
    json.dump(predictions, f, indent=4)

print(f"Success! predictions saved to {output_file}")