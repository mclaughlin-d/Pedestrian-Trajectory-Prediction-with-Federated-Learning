import pandas as pd
import numpy as np

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

# Compute speed in m/s
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
    return data

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
    acceleration = dv / (dt * FRAME_TIME)
    
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
    densities = []

    for frame, frame_data in data.groupby("frame"):
        coords = frame_data[["x", "y"]].values
        n = len(coords)

        dists = np.sqrt(((coords[:, None, :] - coords[None, :, :])**2).sum(axis=2))
        density = (dists < radius).sum(axis=1) - 1  # exclude self

        densities.extend(density)

    data["local_density"] = densities
    return data

def compute_nearest_neighbor(data):
    nn_dist = []

    for frame, frame_data in data.groupby("frame"):
        coords = frame_data[["x", "y"]].values

        dists = np.sqrt(((coords[:, None, :] - coords[None, :, :])**2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)

        nn = dists.min(axis=1)
        # Replace inf with NaN
        nn[nn == np.inf] = np.nan
        nn_dist.extend(nn)

    data["nn_distance"] = nn_dist
    return data

# Compute velocity per pedestrian
def compute_velocity(data):
    data = data.sort_values(by=["global_ped_id", "frame"]).reset_index(drop=True)
    dx = data.groupby("global_ped_id")["x"].diff()
    dy = data.groupby("global_ped_id")["y"].diff()
    dt = data.groupby("global_ped_id")["frame"].diff()
    FRAME_TIME = 0.4

    vx = dx / (dt * FRAME_TIME)
    vy = dy / (dt * FRAME_TIME)

    # First row per pedestrian is NaN
    first_idx = data.groupby("global_ped_id").head(1).index
    vx.loc[first_idx] = np.nan
    vy.loc[first_idx] = np.nan

    data["vx"] = vx
    data["vy"] = vy
    data["speed"] = np.sqrt(vx**2 + vy**2)
    return data

# Compute group metrics per pedestrian
def compute_group_metrics(data, radius=1.0, epsilon=0.1):
    """
    radius: max distance to consider two pedestrians in a group
    epsilon: max speed difference for velocity similarity
    """
    # Initialize storage
    data["in_group"] = 0
    data["group_size"] = 1  # include self

    for frame, frame_data in data.groupby("frame"):
        coords = frame_data[["x", "y"]].values
        vels = frame_data[["vx", "vy"]].values
        n = len(frame_data)

        # Compute pairwise distances
        dists = np.sqrt(((coords[:, None, :] - coords[None, :, :])**2).sum(axis=2))
        # Compute pairwise velocity differences
        vel_diffs = np.sqrt(((vels[:, None, :] - vels[None, :, :])**2).sum(axis=2))

        # Group mask: distance < radius and velocity difference < epsilon
        group_mask = (dists < radius) & (vel_diffs < epsilon)

        # Count number of other pedestrians in group for each pedestrian
        group_sizes = group_mask.sum(axis=1)
        data.loc[frame_data.index, "in_group"] = (group_sizes > 1).astype(int)
        data.loc[frame_data.index, "group_size"] = group_sizes

    # Aggregate pedestrian-level metrics
    ped_group = data.groupby("global_ped_id").agg(
        fraction_time_in_group=("in_group", "mean"),
        avg_group_size=("group_size", "mean")
    ).reset_index()

    data = data.merge(ped_group, on="global_ped_id", how="left")
    return data

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
        filename = f"sorted_by_{name}.txt"
        
        # Keep only relevant columns
        cols_to_keep = ["frame", "global_ped_id", "x", "y", feature]
        sorted_data[cols_to_keep].to_csv(filename, sep="\t", index=False)
