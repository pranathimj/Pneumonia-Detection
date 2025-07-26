import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 17 * 17, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class PneumoniaCNNFeatures(nn.Module):
    def __init__(self, original_model):
        super(PneumoniaCNNFeatures, self).__init__()
        self.feature_extractor = nn.Sequential(*list(original_model.model.children())[:10])
    
    def forward(self, x):
        return self.feature_extractor(x)

class SingleFolderDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        self.image_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) 
                           if f.lower().endswith(self.image_extensions)]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PneumoniaCNN().to(device)

# Load the trained weights
weights_path = os.path.join(os.path.dirname(__file__), 'pneumonia_cnn_weights.pth')
try:
    model.load_state_dict(torch.load(weights_path, map_location=device))
except FileNotFoundError:
    print(f"Error: Weights file '{weights_path}' not found.")
    exit(1)
model.eval()

# Create feature extraction model
feature_model = PneumoniaCNNFeatures(model).to(device)
feature_model.eval()

# Define the transform
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

def compute_reference_features(data_path, feature_model, device, num_samples=100):
    try:
        dataset = SingleFolderDataset(data_path, transform=transform)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        features = []
        feature_model.eval()
        with torch.no_grad():
            for i, (imgs, _) in enumerate(data_loader):
                if i * data_loader.batch_size >= num_samples:
                    break
                imgs = imgs.to(device)
                feats = feature_model(imgs)
                features.extend(feats.cpu().numpy())
        return np.array(features)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

# Compute mean feature vector from dataset
train_data_path = "data"  
reference_features = compute_reference_features(train_data_path, feature_model, device, num_samples=100)
if reference_features is None:
    print("Error: Could not compute reference features. Ensure 'dataset' exists and contains images.")
    exit(1)
mean_features = np.mean(reference_features, axis=0)

# Similarity threshold for OOD detection
similarity_threshold = 0.6

def predict_pneumonia(image_path):
    """
    Predict whether an X-ray image shows pneumonia or not, with OOD detection.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: (diagnosis, confidence, raw_output)
            - diagnosis (str): "Pneumonia", "Normal", or "Unknown"
            - confidence (float): Confidence percentage (0-100) or 0.0 for Unknown/Error
            - raw_output (float): Raw model output (0-1) or 0.0 for Unknown/Error
    """
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Extract features for OOD detection
        with torch.no_grad():
            features = feature_model(image_tensor).cpu().numpy().flatten()
            similarity = cosine_similarity([features], [mean_features])[0][0]
        
        # Debug: Print similarity score
        print(f"Similarity score for {image_path}: {similarity:.4f}")
        
        if similarity < similarity_threshold:
            return "Unknown", 0.0, 0.0
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probability = output.item()
            
        # Convert to diagnosis and confidence
        if probability > 0.5:
            diagnosis = "Pneumonia"
            confidence = probability * 100
            print("Please consult a doctor for further evaluation.")
        else:
            diagnosis = "Normal"
            confidence = (1 - probability) * 100
            
        return diagnosis, confidence, probability
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return "Error", 0.0, 0.0

def get_model_info():
    """
    Get information about the loaded model.
    
    Returns:
        dict: Model information
    """
    return {
        "model_type": "PneumoniaCNN",
        "device": str(device),
        "weights_loaded": os.path.exists(weights_path),
        "input_size": (150, 150),
        "classes": ["Normal", "Pneumonia", "Unknown"],
        "ood_detection": "Feature-based with cosine similarity"
    }