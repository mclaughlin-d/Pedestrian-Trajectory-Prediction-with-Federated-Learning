"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
import pandas as pd
import numpy as np

from task import TrajectoryLSTM, load_data, test as test_fn, train as train_fn, dataset_files, sorted_dataset

num_partitions = 5 # 8
# Flower ClientApp
app = ClientApp()

big_df = pd.read_csv(sorted_dataset, sep="\t")
big_df = big_df.iloc[:, :4]  # keep only first 4 columns
big_df.columns = ["frame", "pedestrian_id", "x", "y"]

# Ensure numeric columns are floats/ints
big_df["frame"] = big_df["frame"].astype(int)
big_df["x"] = big_df["x"].astype(float)
big_df["y"] = big_df["y"].astype(float)

@app.train()
# Train the model on local data.
def train(msg: Message, context: Context):
    # Load the model and initialize it with the received weights
    model = TrajectoryLSTM(pred=12)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    
    partition_id = context.node_config["partition-id"] % num_partitions
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size, hist=8, pred=12, big_df=big_df)


    # Train the model
    train_metrics = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device
    )
    """# Print convergence info
    print(f"[Client {partition_id}] Training metrics:")
    print(f"  Avg loss: {train_metrics['avg_loss']:.6f}")
    print(f"  Std loss: {train_metrics['std_loss']:.6f}")
    print(f"  Min loss: {train_metrics['min_loss']:.6f}")
    print(f"  Max loss: {train_metrics['max_loss']:.6f}")"""

    # Construct reply Message with aggregated metrics
    
    metrics = {
        "avg_loss": float(train_metrics["avg_loss"]),
        "std_loss": float(train_metrics["std_loss"]),
        "min_loss": float(train_metrics["min_loss"]),
        "max_loss": float(train_metrics["max_loss"]),
        "num-examples": len(trainloader.dataset)
    }

    
    model_record = ArrayRecord(model.state_dict())
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
# Evaluate the model on local data.
def evaluate(msg: Message, context: Context):
    # Load the model and initialize it with the received weights
    model = TrajectoryLSTM(pred=12)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"] % num_partitions
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size, hist=8, pred=12, big_df=big_df)

    # Call the evaluation function
    ADE, FDE, miss_rate = test_fn(model, valloader, device)

    # Construct and return reply Message
    metrics = {
        "ADE": ADE,
        "FDE": FDE,
        "miss_rate": miss_rate,
        "num-examples": len(valloader.dataset)
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
