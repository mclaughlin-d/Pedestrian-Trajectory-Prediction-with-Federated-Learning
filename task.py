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
sorted_dataset = "sorted_by_mean_speed.txt"

# Model
class TrajectoryLSTM(nn.Module):

    def __init__(self, input_dim=2, embed_dim=64, hidden_dim=32, latent_dim=32, pred=12, timesteps=12, output_dim=2, num_layers=2):
        super().__init__()
        self.pred = pred
        self.pred_timestep = timesteps

        self.fc1 = nn.Linear(input_dim, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        self.decoder = nn.LSTM(embed_dim + latent_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

        # destination LSTM
        self.dest_lstm_obs = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dest_lstm_pred = nn.LSTM(input_dim, hidden_dim, batch_first=True)
 
        # FC -> mu + log_var for reparameterisation (VAE-style KL, like GTPPO)
        self.dest_fc_obs     = nn.Linear(hidden_dim, latent_dim)
        self.dest_fc_pred    = nn.Linear(hidden_dim, latent_dim)
 
        # Xavier init for all linear layers (paper: "After Xavier initialization")
        # https://www.geeksforgeeks.org/deep-learning/xavier-initialization/
        # basically, initializes weights from some specific distribution, rather than intializing
        # them all to 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode_destination(self, coords, lstm, fc):
        """
        Run absolute position sequence through a destination LSTM, then
        map the final hidden state to a 32-dim destination vector via FC.
 
        Args:
            coords : (B, T, 2) -- absolute positions
            lstm   : destination LSTM module
            fc     : FC layer projecting hidden_dim -> dest_dim
        Returns:
            D : (B, dest_dim)  deterministic destination vector
        """
        _, (h, _) = lstm(coords)
        h = h.squeeze(0)
        return fc(h)

    def forward(self, obs_disp, obs_abs, gt_abs):
        # input displacement mapped to 64-dim motion feature vector
        Z = self.fc1(obs_disp) # input -> embedding, taken from eq 1
        out1, (h1, c1) = self.encoder(Z) # H1, taken from eq 2


        D = self._encode_destination(
            obs_abs, self.dest_lstm_obs, self.dest_fc_obs)
 
        dest_dict = None
        # ground truth absolute destination
        # this is the arrow going from left to right in the diagram
        # so, KL <- D^ <- FC <- Pos~ , basically
        if gt_abs is not None:
            # D_hat: 32-dim vector derived from ground-truth future (training only)
            D_hat = self._encode_destination(
                gt_abs, self.dest_lstm_pred, self.dest_fc_pred)
            dest_dict = {'D': D, 'D_hat': D_hat}
            
        last_z   = Z[:, -1:, :]
        D_expand = D.unsqueeze(1)
        dec_input = torch.cat([last_z, D_expand], dim=-1)
 
        h_dec, c_dec = h1, c1
        last_abs = obs_abs[:, -1, :] # running absolute position
 
        pred_abs_list = []
        for _ in range(self.pred_timestep):
            # Q^{T_obs+1} = F_dec(H^{T_obs}, Z^{T_obs} || D; W_d) - this is equation 5
            out, (h_dec, c_dec) = self.decoder(dec_input, (h_dec, c_dec))
            # delta = delta(Q, W_c) - this is equation 6
            delta = self.fc2(out.squeeze(1))          # (B, 2)
            last_abs = last_abs + delta # inverse of Eqs 3-4
            pred_abs_list.append(last_abs.unsqueeze(1))
 
            # Prepare next decoder input
            z_next = self.fc1(delta.unsqueeze(1))
            dec_input = torch.cat([z_next, D_expand], dim=-1)
 
        pred_abs = torch.cat(pred_abs_list, dim=1)          # (B, pred_len, 2)
        return pred_abs, dest_dict

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
# Split a DataFrame by pedestrian into N clients
def split_df_into_clients(df: pd.DataFrame, num_clients: int):
    """
    Split a DataFrame into `num_clients` by pedestrian ID, sequentially.
    Each pedestrian stays in a single client.
    Returns a list of DataFrames, one per client.
    """
    df = df.copy()
    df["pedestrian_id"] = df["pedestrian_id"].str.strip()
    unique_peds = df["pedestrian_id"].unique()

    # Split pedestrian IDs sequentially
    df["pedestrian_id"] = df["pedestrian_id"].str.strip()
    ped_groups = np.array_split(unique_peds, num_clients)

    client_dfs = []
    for ped_ids in ped_groups:
        client_dfs.append(df[df["pedestrian_id"].isin(ped_ids)].copy())
    return client_dfs

def load_data(partition_id: int, num_partitions: int, batch_size: int, hist=8, pred=12, big_df=None):
    # If we have just one large, sorted file
    if big_df is not None:
        client_dfs = split_df_into_clients(big_df, num_partitions)
        df = client_dfs[partition_id]
    else:
        file = dataset_files[partition_id]
        df = pd.read_csv(file, sep="\t", header=None)
        df.columns = ["frame", "pedestrian_id", "x", "y"]

    dataset = TrajectoryDataset(df, hist=hist, pred=pred)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size)
    return trainloader, valloader

# For purposes of server evaluation / visualization
def load_full_dataset(batch_size=32, hist=8, pred=12, num_partitions=8):
    datasets = []
    for file in dataset_files:
        df = pd.read_csv(file, sep="\t", header=None)
        df.columns = ["frame", "pedestrian_id", "x", "y"]
        datasets.append(TrajectoryDataset(df, hist=hist, pred=pred))

    concat_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(concat_dataset, batch_size=batch_size)

    return dataloader

# Training
def train(model, trainloader, epochs, lr, device, kl_weight = 0.001):
    model.to(device)  # move model to GPU if available
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    running_loss = 0.0
    for _ in range(epochs):
        for obs_disp, y, obs_abs, fut_abs in trainloader:
            obs_disp, y, obs_abs, fut_abs = obs_disp.to(device), y.to(device), obs_abs.to(device), fut_abs.to(device)
            optimizer.zero_grad()
            pred, dest_dict = model(obs_disp, obs_abs, fut_abs)
            loss = criterion(pred, fut_abs)
            #dest_loss = nn.functional.mse_loss(dest_dict["D"], dest_dict["D_hat"])
            #loss = loss + 0.001 * dest_loss
            if dest_dict is not None:
                kl_loss = nn.functional.mse_loss(dest_dict["D"], dest_dict["D_hat"])
                loss += kl_weight * kl_loss
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip gradients
            optimizer.step()
            running_loss += loss.item()
    avg_loss = running_loss / (epochs * len(trainloader))
    return avg_loss

def test(model, testloader, device, miss_threshold=0.5):
    model.to(device)
    model.eval()
    
    total_ade = 0
    total_fde = 0
    total_traj = 0
    misses = 0

    with torch.no_grad():
        for obs_disp, y, obs_abs, fut_abs in testloader:
            obs_disp, y, obs_abs, fut_abs = obs_disp.to(device), y.to(device), obs_abs.to(device), fut_abs.to(device)
            pred, dest_dict = model(obs_disp, obs_abs, fut_abs)
            dist = torch.sqrt(((pred - fut_abs) ** 2).sum(dim=2))
            ade = dist.mean(dim=1)
            fde = dist[:, -1]

            total_ade += ade.sum().item()
            total_fde += fde.sum().item()
            misses += (fde > miss_threshold).sum().item()
            total_traj += obs_disp.size(0)


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
    obs_abs_t = obs_abs.to(device)
    fut_abs_t = fut_abs.to(device)
    fut_abs = fut_abs.cpu().numpy()
    obs_abs = obs_abs.cpu().numpy()

    with torch.no_grad(): # disable gradient tracking for speed
        pred_abs, _ = model(obs.unsqueeze(0), obs_abs_t.unsqueeze(0), fut_abs_t.unsqueeze(0))
        pred_positions = pred_abs.squeeze(0).cpu().numpy()

    plt.figure(figsize=(6,6))
    plt.plot(obs_abs[:,0], obs_abs[:,1], 'bo-', label='Observed')
    plt.plot(fut_abs[:,0], fut_abs[:,1], 'go-', label='Ground truth')
    plt.plot(pred_positions[:,0], pred_positions[:,1], 'ro--', label='Predicted')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
