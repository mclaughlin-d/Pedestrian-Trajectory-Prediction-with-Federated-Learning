"""pytorchexample: A Flower / PyTorch app."""

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib
matplotlib.use("Agg")  # for headless servers
import matplotlib.pyplot as plt

# Ignore FutureWarnings (swapaxes, Ray GPU warnings) to clean up logs
warnings.simplefilter(action='ignore', category=FutureWarning)

# Dataset files
dataset_files = [
    "datasets/biwi_eth.txt",
    "datasets/biwi_hotel.txt",
    "datasets/crowds_zara01.txt",
    "datasets/crowds_zara02.txt",
    "datasets/crowds_zara03.txt",
    "datasets/students001.txt",
    "datasets/students003.txt",
    "datasets/uni_examples.txt"
]

# Model
class TrajectoryLSTM(nn.Module):

    def __init__(self, input_dim=2, hidden_dim=128, pred=12, output_dim=2, num_layers=2):
        super().__init__()
        self.pred = pred
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    # Predicts trajectory step by step, using previous predicted position as input
    # for next time step (autoregressive)
    def forward(self, x):
        batch_size = x.size(0)
        out1, (h1, c1) = self.lstm(x)
        _, (h2, c2) = self.lstm2(out1)
        last_pos = x[:, -1, :]

        preds = []
        current_pos = last_pos

        for _ in range(self.pred):
            lstm1_out, (h1, c1) = self.lstm(current_pos.unsqueeze(1), (h1, c1))
            out_step, (h2, c2) = self.lstm2(lstm1_out, (h2, c2))
            delta = self.fc2(torch.relu(self.fc1(out_step.squeeze(1))))
            current_pos = current_pos + delta   # update position
            preds.append(current_pos.unsqueeze(1))  # append NEW position

        preds = torch.cat(preds, dim=1)
        return preds

# Dataset
class TrajectoryDataset(Dataset):
    def __init__(self, df, hist=8, pred=12):
        self.sequences = []
        for pid, g in df.groupby("pedestrian_id"):
            coords = g.sort_values("frame")[["x", "y"]].values

            for i in range(len(coords) - hist - pred + 1):
                obs_abs = coords[i:i+hist]
                fut_abs = coords[i+hist:i+hist+pred]

                # compute relative displacements
                obs = np.vstack([np.zeros((1,2)), np.diff(obs_abs, axis=0)])
                fut = np.vstack([np.diff(fut_abs, axis=0)[0:1], np.diff(fut_abs, axis=0)])

                self.sequences.append((obs, fut, obs_abs, fut_abs))

    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        obs, fut, obs_abs, fut_abs = self.sequences[idx]
        return (
            torch.FloatTensor(obs),
            torch.FloatTensor(fut),
            torch.FloatTensor(obs_abs),
            torch.FloatTensor(fut_abs)
        )

# Load single file for client 
def load_data(partition_id: int, num_partitions: int, batch_size: int, hist=8, pred=12):
    # Load a dataset file as training and validation DataLoader.
    file = dataset_files[partition_id]
    df = pd.read_csv(file, sep="\t", header=None)
    df.columns = ["frame", "pedestrian_id", "x", "y"]
    df["pedestrian_id"] = pd.factorize(df["pedestrian_id"])[0] # Make IDs unique per file

    dataset = TrajectoryDataset(df, hist=hist, pred=pred)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size)
    return trainloader, valloader

# For purposes of server evaluation / visualization
def load_full_dataset(batch_size=32, hist=8, pred=12, num_partitions=5):
    datasets = []
    for file in dataset_files:
        df = pd.read_csv(file, sep="\t", header=None)
        df.columns = ["frame", "pedestrian_id", "x", "y"]
        datasets.append(TrajectoryDataset(df, hist=hist, pred=pred))

    concat_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(concat_dataset, batch_size=batch_size)

    return dataloader

# Training
def train(model, trainloader, epochs, lr, device):
    model.to(device)  # move model to GPU if available
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    running_loss = 0.0
    for _ in range(epochs):
        for x, y, _, _ in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip gradients
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_loss = running_loss / (epochs * len(trainloader))
    return avg_loss

def test(model, testloader, device, miss_threshold=0.1):
    model.to(device)
    model.eval()
    
    total_ade = 0
    total_fde = 0
    total_traj = 0
    misses = 0

    with torch.no_grad():
        for x, y, _, _ in testloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            dist = torch.sqrt(((pred - y) ** 2).sum(dim=2))
            ade = dist.mean(dim=1)
            fde = dist[:, -1]

            total_ade += ade.sum().item()
            total_fde += fde.sum().item()
            misses += (fde > miss_threshold).sum().item()
            total_traj += x.size(0)


    ADE = total_ade / total_traj
    FDE = total_fde / total_traj
    miss_rate = misses / total_traj

    return ADE, FDE, miss_rate

# Converts relative displacements back to absolute positions
def reconstruct(start, displacements):
    positions = [start]
    for d in displacements:
        positions.append(positions[-1] + d)
    return np.array(positions)

# Visualization
# Picks one sample_index from dataset
# sample_index is not selecting a pedestrian - it is selecting a subsequence
# self.sequences conatins all subsequences of each pedestrian
# You could pick a "bad" subsequence with a very short trajectory, corner cases with only a few frames, etc
# Just pick a good representative one based on trial and error 
# We could modify later to pick a specific pedestrian that we know have enough frames, but for
# our purposes this is probably good enough 
def visualize_prediction(model, dataloader, device, save_path="trajectory.png", sample_index=0):
    model.to(device) # added to hopefully fix error?
    model.eval()
    full_dataset = dataloader.dataset
    # This part is no necessary if I'm just showing one trajectory as with my current call
    if isinstance(full_dataset, ConcatDataset): # Check if dataset is concatentation of multiple
        # If concatentaion, need to get local index
        ds_cum_lengths = np.cumsum([len(d) for d in full_dataset.datasets])
        for i, l in enumerate(ds_cum_lengths): # loop over dataset boundaries to find which sub-dataset has the sample
            if sample_index < l:
                local_index = sample_index if i==0 else sample_index - ds_cum_lengths[i-1]
                obs, fut, obs_abs, fut_abs = full_dataset.datasets[i][local_index]
                break
    else:
        obs, fut, obs_abs, fut_abs = full_dataset[sample_index]

    obs = obs.to(device)
    fut_abs = fut_abs.cpu().numpy()
    obs_abs = obs_abs.cpu().numpy()

    with torch.no_grad(): # disable gradient tracking for speed
        pred_deltas = model(obs.unsqueeze(0)).squeeze(0).cpu().numpy()

    # Reconstruct absolute trajectory from last observed position
    start = obs_abs[-1]
    pred_positions = reconstruct(start, pred_deltas)

    plt.figure(figsize=(6,6))
    plt.plot(obs_abs[:,0], obs_abs[:,1], 'bo-', label='Observed')
    plt.plot(fut_abs[:,0], fut_abs[:,1], 'go-', label='Ground truth')
    plt.plot(pred_positions[:,0], pred_positions[:,1], 'ro--', label='Predicted')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
