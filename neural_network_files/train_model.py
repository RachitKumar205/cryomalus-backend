import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("MPS device not found.")

# Custom Dataset for loading CSV data
class ThermalDataset(Dataset):
    def __init__(self, directory, num_series=10000, time_steps=5, grid_size=100):
        self.images = []
        self.labels = []
        for series_idx in range(num_series):
            series_images = []
            for t in range(time_steps):
                img_path = os.path.join(directory, f"series_{series_idx:04d}", f"thermal_data_step_{t:03d}.csv")
                if os.path.exists(img_path):
                    df = pd.read_csv(img_path, header=0)
                    image = df.to_numpy()
                    series_images.append(image)

            if len(series_images) == time_steps:
                series_images = np.stack(series_images, axis=-1)  # Shape (grid_size, grid_size, time_steps)
                self.images.append(series_images)

                metadata_path = os.path.join(directory, f"series_{series_idx:04d}", "metadata.csv")
                metadata = pd.read_csv(metadata_path)
                anomaly_point = metadata[['varying_point_x', 'varying_point_y']].values[0]
                self.labels.append(anomaly_point)

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        print(self.images.shape)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32).permute(2, 0,
                                                                            1)  # (time_steps, grid_size, grid_size)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

# Define the CNN model in PyTorch
class ThermalCNN(nn.Module):
    def __init__(self):
        super(ThermalCNN, self).__init__()
        self.conv1 = nn.Conv2d(5, 16, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 8, kernel_size=1, padding=0)

        # Gradually reducing dimensions
        self.fc1 = nn.Linear(8 * 100 * 100, 2048)  # First reduce to 1024
        self.fc2 = nn.Linear(2048, 512)            # Then reduce to 512
        self.fc3 = nn.Linear(512, 128)             # Further reduce to 128
        self.fc4 = nn.Linear(128, 64)              # Finally reduce to 64

        self.output_layer = nn.Linear(64, 2)       # Output layer for coordinates (x, y)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        # Flatten the output from the convolutional layers
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flattening to (batch_size, 128 * 10 * 10)

        # Pass through the fully connected layers with ReLU activations
        x = torch.relu(self.fc1(x))  # First dense layer
        x = torch.relu(self.fc2(x))  # Second dense layer
        x = torch.relu(self.fc3(x))  # Third dense layer
        x = torch.relu(self.fc4(x))  # Fourth dense layer

        # Final output layer for predicting (x, y) coordinates
        x = self.output_layer(x)
        return x

# Load dataset
data_dir = "thermal_dataset_output"
dataset = ThermalDataset(data_dir)

# Split data into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader for batching
batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, loss, and optimizer
model = ThermalCNN()

# Load pre-trained model if available
model_path = 'thermal_cnn_model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Loaded pre-trained model.")
else:
    print("No pre-trained model found, starting from scratch.")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Save the model
torch.save(model.state_dict(), 'thermal_cnn_model.pth')

