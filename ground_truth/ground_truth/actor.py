"""
where is the updated ground truth csv
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import csv


class LidarAutoencoder(nn.Module):
    # Dummy definition for completeness
    def __init__(self, input_dim=1080, encoding_dim=16): #input dim wrong
        super().__init__()
        dim1 = max(512, encoding_dim * 8)
        dim2 = max(256, encoding_dim * 4)
        dim3 = max(128, encoding_dim * 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, dim1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim2, dim3),
            nn.ReLU(),
            nn.Linear(dim3, encoding_dim)
        )
    def forward(self, x):
        return self.encoder(x)

class SliceLidar(Dataset):
    def __init__(self, csv_path):
        self.lidar_data = []
        self.actions = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                row = [float(x) for x in row]
                self.lidar_data.append(row[:-2])
                self.actions.append(row[-2:])

    def __len__(self):
        return len(self.lidar_data)

    def __getitem__(self, idx):
        lidar = torch.tensor(self.lidar_data[idx], dtype=torch.float32)
        actions = torch.tensor(self.actions[idx], dtype=torch.float32)
        return lidar, actions

class Actor(nn.Module):
    def __init__(self, input_dim=16, output_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return torch.cat([
            torch.sigmoid(self.net(x)[:, [0]]),  
            torch.tanh(self.net(x)[:, [1]])    
        ], dim=1)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 1e-3  # Learning rate

    autoencoder = LidarAutoencoder(input_dim=1080, encoding_dim=16).to(device) # defining autoencoder as the same net as in autoencoder train
    checkpoint = torch.load('/sim_ws/src/ground_truth/ground_truth/auto_encoder_data/lidar_autoencoder_16D.pth', map_location=device)
    autoencoder.load_state_dict(checkpoint['model_state_dict'],strict=False)

    autoencoder.eval()
    encoder = autoencoder.encoder

    actor = Actor().to(device)
    dataset = SliceLidar("/sim_ws/src/ground_truth/raw_data/dataset_20250605_224934.csv") 
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = optim.Adam(actor.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(100):
        for lidar, actions in loader:
            lidar = lidar.to(device)
            actions = actions.to(device)
            with torch.no_grad():
                compressed = encoder(lidar)
            pred_actions = actor(compressed)
            loss = criterion(pred_actions, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    torch.save(actor.state_dict(), "lidar_policy.pth")

if __name__ == "__main__":
    main()
