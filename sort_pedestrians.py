import pandas as pd
import numpy as np
import random

# Dataset files
dataset_files = [
    "datasets/original_datasets/biwi_eth.txt",
    "datasets/original_datasets/biwi_hotel.txt",
    "datasets/original_datasets/crowds_zara01.txt",
    "datasets/original_datasets/crowds_zara02.txt",
    "datasets/original_datasets/crowds_zara03.txt",
    "datasets/original_datasets/students001.txt",
    "datasets/original_datasets/students003.txt",
    "datasets/original_datasets/uni_examples.txt"
]

# Load all 8 txt files into one structure and add a scene/file ID so pedestrians don’t collide.
def load_data(files):
    dfs = []
    for i, f in enumerate(files):
        df = pd.read_csv(f, sep="\t", header=None, names=["frame", "ped_id", "x", "y"])
        df["scene_id"] = i
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def create_global_id(data):
    data = data.copy()
    data["global_ped_id"] = data["scene_id"].astype(str) + "_" + data["ped_id"].astype(str)
    return data


def compute_speed(data):
    """
    Assumes:
    - data is already subsampled (every 10 frames)
    - each row = 0.4 seconds apart in real time
    """

    data = data.sort_values(["global_ped_id", "frame"]).copy()

    dx = data.groupby("global_ped_id")["x"].diff()
    dy = data.groupby("global_ped_id")["y"].diff()

    dist = np.sqrt(dx**2 + dy**2)

    TIME_STEP = 0.4  # ground-truth assumption for ETH/UCY style data

    speed = dist / TIME_STEP

    # only keep valid consecutive points
    dt_frames = data.groupby("global_ped_id")["frame"].diff()
    speed[dt_frames != 10] = np.nan

    data["speed"] = speed
    return data
"""# Compute speed in m/s
def compute_speed(data):
    data = data.sort_values(by=["global_ped_id", "frame"]).reset_index(drop=True)

    dx = data.groupby("global_ped_id")["x"].diff()
    dy = data.groupby("global_ped_id")["y"].diff()
    dt = data.groupby("global_ped_id")["frame"].diff()

    FRAME_TIME = 0.4

    speed = np.sqrt(dx**2 + dy**2) / FRAME_TIME

    # Invalidate bad time steps
    speed[dt != 10] = np.nan

    data["speed"] = speed
    return data"""

# Get mean speed per pedestrian
def compute_mean_speed(data):
    data["mean_speed"] = data.groupby("global_ped_id")["speed"].transform("mean")
    return data

# Sort entire dataset by pedestrian mean speed
def sort_by_feature(data, feature, ascending=False):

    # Get ordering of pedestrians based on feature
    ped_order = (
        data.groupby("global_ped_id")[feature]
        .mean()   # or .first() if already constant
        .sort_values(ascending=ascending)
        .index
    )

    # Apply ordering
    data = data.copy()
    data["global_ped_id"] = pd.Categorical(
        data["global_ped_id"],
        categories=ped_order,
        ordered=True
    )

    # Sort
    data = data.sort_values(by=["global_ped_id", "frame"])

    return data

def compute_acceleration(data):
    data = data.sort_values(by=["global_ped_id", "frame"]).reset_index(drop=True)
    
    dt = data.groupby("global_ped_id")["frame"].diff()
    dv = data.groupby("global_ped_id")["speed"].diff()
    FRAME_TIME = 0.4
    
    # Compute acceleration
    acceleration = dv / (FRAME_TIME)
    
    # Invalidate first row per pedestrian
    first_idx = data.groupby("global_ped_id").head(1).index
    acceleration.loc[first_idx] = np.nan
    
    # Also invalidate rows where dt is NaN (should catch any missing frames)
    acceleration[dt.isna()] = np.nan
    
    data["acceleration"] = acceleration
    return data


def compute_curvature(data):
    data = data.sort_values(by=["global_ped_id", "frame"]).reset_index(drop=True)
    
    dx = data.groupby("global_ped_id")["x"].diff()
    dy = data.groupby("global_ped_id")["y"].diff()
    ddx = dx.groupby(data["global_ped_id"]).diff()
    ddy = dy.groupby(data["global_ped_id"]).diff()
    
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx**2 + dy**2)**1.5
    
    curvature = numerator / denominator
    
    # Invalidate first two rows per pedestrian or any row where diff is NaN
    invalid_idx = dx.isna() | dy.isna() | ddx.isna() | ddy.isna()
    curvature[invalid_idx] = np.nan
    
    # Also invalidate curvature where speed is NaN
    curvature[data["speed"].isna()] = np.nan
    
    data["curvature"] = curvature
    return data

def compute_density(data, radius=2.0):
    densities = {}

    for (scene_id, frame), frame_data in data.groupby(["scene_id", "frame"]):
        coords = frame_data[["x", "y"]].values

        dists = np.sqrt(((coords[:, None, :] - coords[None, :, :])**2).sum(axis=2))
        density = (dists < radius).sum(axis=1) - 1  # exclude self

        for idx, d in zip(frame_data.index, density):
            densities[idx] = d

    data["local_density"] = data.index.map(densities)
    return data

def compute_nearest_neighbor(data):
    nn_dist = {}

    for (scene_id, frame), frame_data in data.groupby(["scene_id", "frame"]):
        coords = frame_data[["x", "y"]].values

        dists = np.sqrt(((coords[:, None, :] - coords[None, :, :])**2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)

        nn = dists.min(axis=1)
        nn[nn == np.inf] = np.nan

        for idx, d in zip(frame_data.index, nn):
            nn_dist[idx] = d

    data["nn_distance"] = data.index.map(nn_dist)
    return data

# Compute velocity per pedestrian
def compute_velocity(data):
    data = data.sort_values(by=["global_ped_id", "frame"]).reset_index(drop=True)
    dx = data.groupby("global_ped_id")["x"].diff()
    dy = data.groupby("global_ped_id")["y"].diff()
    dt = data.groupby("global_ped_id")["frame"].diff()
    FRAME_TIME = 0.4

    vx = dx / (FRAME_TIME)
    vy = dy / (FRAME_TIME)

    # First row per pedestrian is NaN
    first_idx = data.groupby("global_ped_id").head(1).index
    vx.loc[first_idx] = np.nan
    vy.loc[first_idx] = np.nan

    data["vx"] = vx
    data["vy"] = vy
    #data["speed"] = np.sqrt(vx**2 + vy**2)
    return data

# Compute group metrics per pedestrian
def compute_group_metrics(data, radius=2.0, epsilon=0.1):
    in_group = {}
    group_size = {}

    for (scene_id, frame), frame_data in data.groupby(["scene_id", "frame"]):
        coords = frame_data[["x", "y"]].values
        vels = frame_data[["vx", "vy"]].values

        dists = np.sqrt(((coords[:, None, :] - coords[None, :, :])**2).sum(axis=2))
        vel_diffs = np.sqrt(((vels[:, None, :] - vels[None, :, :])**2).sum(axis=2))

        group_mask = (dists < radius) & (vel_diffs < epsilon)
        group_sizes = group_mask.sum(axis=1)

        for idx, gs in zip(frame_data.index, group_sizes):
            in_group[idx] = int(gs > 1)
            group_size[idx] = gs

    data["in_group"] = data.index.map(in_group)
    data["group_size"] = data.index.map(group_size)

    ped_group = data.groupby("global_ped_id").agg(
        fraction_time_in_group=("in_group", "mean"),
        avg_group_size=("group_size", "mean")
    ).reset_index()

    data = data.merge(ped_group, on="global_ped_id", how="left")
    return data

def compute_feature_stats(df, feature, label="GLOBAL"):
    # collapse to one value per pedestrian
    ped_df = df.groupby("global_ped_id", observed=True)[feature].mean()
    num_pedestrians = ped_df.shape[0]
    stats = {
        "label": label,
        "num_pedestrians": num_pedestrians,
        "mean": ped_df.mean(),
        "std": ped_df.std(),
        "min": ped_df.min(),
        "max": ped_df.max()
    }
    return stats

def split_into_partitions(df, num_partitions=5):
    unique_peds = df["global_ped_id"].unique()
    ped_groups = np.array_split(unique_peds, num_partitions)

    partitions = []
    for ped_ids in ped_groups:
        part_df = df[df["global_ped_id"].isin(ped_ids)].copy()
        partitions.append(part_df)

    return partitions

if __name__ == "__main__":

    # Load data
    data = load_data(dataset_files)
    data = create_global_id(data)

    # Compute
    data = compute_speed(data)
    data = compute_acceleration(data)
    data = compute_curvature(data)
    data = compute_density(data)
    data = compute_nearest_neighbor(data)
    data = compute_velocity(data)  # needed for group metrics
    data = compute_group_metrics(data, radius=1.0, epsilon=0.1)  # compute group metrics


    # Assign pedestrian-level constants to all frames
    data["mean_speed"] = data.groupby("global_ped_id")["speed"].transform("mean")
    data["speed_var"] = data.groupby("global_ped_id")["speed"].transform("var")
    data["acc_mean"] = data.groupby("global_ped_id")["acceleration"].transform("mean")
    data["acc_var"] = data.groupby("global_ped_id")["acceleration"].transform("var")
    data["curvature_mean"] = data.groupby("global_ped_id")["curvature"].transform("mean")
    data["local_density_mean"] = data.groupby("global_ped_id")["local_density"].transform("mean")
    data["nn_mean"] = data.groupby("global_ped_id")["nn_distance"].transform("mean")


    # List of features to sort by
    features = {
        "mean_speed": ("mean_speed", False),
        "speed_variance": ("speed_var", False),
        "mean_acceleration": ("acc_mean", False),
        "acc_variance": ("acc_var", False),
        "curvature": ("curvature_mean", False),
        "density": ("local_density_mean", False),
        "nn_mean": ("nn_mean", False),
        "fraction_time_in_group": ("fraction_time_in_group", False),
        "avg_group_size": ("avg_group_size", False)
    }

    for name, (feature, asc) in features.items():
        sorted_data = sort_by_feature(data, feature, ascending=asc)
        filename = f"datasets/sorted_datasets/sorted_by_{name}.txt"
        
        # Keep only relevant columns
        cols_to_keep = ["frame", "global_ped_id", "x", "y", feature]
        sorted_data[cols_to_keep].to_csv(filename, sep="\t", index=False)

    for name, (feature, asc) in features.items():
        print(f"\n===== FEATURE: {name} =====")

        sorted_data = sort_by_feature(data, feature, ascending=asc)

        # ---- GLOBAL STATS ----
        global_stats = compute_feature_stats(sorted_data, feature, label="GLOBAL")
        print(global_stats)

        # ---- SPLIT INTO 5 PARTITIONS ----
        partitions = split_into_partitions(sorted_data, num_partitions=5)

        all_stats = [global_stats]

        for i, part in enumerate(partitions):
            num_unique = part["global_ped_id"].nunique()
            print(f"CLIENT_{i} unique pedestrians: {num_unique}")
            stats = compute_feature_stats(part, feature, label=f"CLIENT_{i}")
            all_stats.append(stats)
            print(stats)

        # ---- SAVE STATS ----
        stats_df = pd.DataFrame(all_stats)
        stats_filename = f"stats_{name}.csv"
        stats_df.to_csv(stats_filename, index=False)

    # also output combined data for easier non-sorted testing
    filename = f"datasets/sorted_datasets/non_sorted.txt"
    cols_to_keep = ["frame", "global_ped_id", "x", "y", feature]
    data[cols_to_keep].to_csv(filename, sep="\t", index=False)
