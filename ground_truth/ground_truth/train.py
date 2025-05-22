import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
from sklearn.model_selection import train_test_split
import os

# 1. Load CSV
def load_data(csv_path):
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # convert strings to floats
            data.append([float(x) for x in row])
    data = np.array(data)
    X = data[:, :-2]  # all but last 2 columns = inputs
    y = data[:, -2:]  # last 2 columns = steering angle & speed
    return X, y

# 2. Dataset class for PyTorch
class ExpertDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 3. Simple model
class SimpleNet(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def train_model(csv_path):
    X, y = load_data(csv_path)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = ExpertDataset(X_train, y_train)
    val_dataset = ExpertDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    model = SimpleNet(input_dim=X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(500):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    models_dir = "/sim_ws/src/ground_truth/models"
    os.makedirs(models_dir, exist_ok=True)

    # Create a unique filename with timestamp
    model_path = os.path.join(models_dir, f"dagger.pth")


    # Save the model
    torch.save(model.state_dict(), model_path)    
    print(f"Model saved to {model_path}")

def main():
    train_model("/sim_ws/src/ground_truth/raw_data/dataset_20250516_173325.csv")

if __name__ == "__main__":
    main()