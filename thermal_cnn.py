import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import os
from random import randrange

class ThermalCNN(nn.Module):
    def __init__(self):
        super(ThermalCNN, self).__init__()
        self.conv1 = nn.Conv2d(5, 16, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 8, kernel_size=1, padding=0)

        self.fc1 = nn.Linear(8 * 100 * 100, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.output_layer(x)
        return x

def load_model(model_path='thermal_cnn_model.pth'):
    model = ThermalCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def infer(model, data_dir, series_idx, time_steps=5, grid_size=100):
    series_images = []
    for t in range(time_steps):
        img_path = os.path.join(data_dir, f"series_{series_idx:04d}", f"thermal_data_step_{t:03d}.csv")
        if os.path.exists(img_path):
            df = pd.read_csv(img_path, header=0)
            image = df.to_numpy()
            series_images.append(image)

    if len(series_images) == time_steps:
        series_images = np.stack(series_images, axis=-1)  # Shape (grid_size, grid_size, time_steps)
        image_tensor = torch.tensor(series_images, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            prediction = model(image_tensor)
        return prediction.squeeze().numpy()  # Remove batch dimension
    else:
        raise ValueError(f"Incomplete time series for series index {series_idx}")

def load_actual_coordinates(data_dir, series_idx):
    metadata_path = os.path.join(data_dir, f"series_{series_idx:04d}", "metadata.csv")
    if os.path.exists(metadata_path):
        metadata = pd.read_csv(metadata_path)
        anomaly_point = metadata[['varying_point_x', 'varying_point_y']].values[0]
        return anomaly_point
    else:
        raise ValueError(f"Metadata file not found for series index {series_idx}")

def run_inference(data_dir, model_path='thermal_cnn_model.pth', series_idx=None):
    if series_idx is None:
        series_idx = randrange(93)  # Change this to the index of the series you want to infer

    model = load_model(model_path)
    predicted_coordinates = infer(model, data_dir, series_idx)

    return predicted_coordinates
