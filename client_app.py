"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from task import TrajectoryLSTM, load_data, test as test_fn, train as train_fn, dataset_files

num_partitions = len(dataset_files)
# Flower ClientApp
app = ClientApp()


@app.train()
# Train the model on local data.
def train(msg: Message, context: Context):
    # Load the model and initialize it with the received weights
    model = TrajectoryLSTM(pred=12)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    
    #partition_id = context.node_config["partition-id"]
    partition_id = context.node_config["partition-id"] % num_partitions
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size, hist=8, pred=12)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
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
    #partition_id = context.node_config["partition-id"]
    partition_id = context.node_config["partition-id"] % num_partitions
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size, hist=8, pred=12)

    # Call the evaluation function
    ADE, FDE, miss_rate = test_fn(model, valloader, device)

    # Construct and return reply Message
    metrics = {
        "ADE": ADE,
        "FDE": FDE,
        "miss_rate": miss_rate,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
