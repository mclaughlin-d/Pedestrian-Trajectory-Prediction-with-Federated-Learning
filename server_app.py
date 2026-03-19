"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
from task import TrajectoryLSTM, load_data, test, load_full_dataset, visualize_prediction

# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    # Main entry point for the ServerApp.
    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Load global model
    global_model = TrajectoryLSTM(pred=12)
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )
    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    global_model.load_state_dict(state_dict)
    torch.save(state_dict, "final_model.pt")
 
    global_model.load_state_dict(state_dict)
    torch.save(state_dict, "final_model.pt")
    # Load the test dataset and device for visualization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataloader = load_full_dataset(batch_size=32, hist=8, pred=12)

    # Visualize predictions for the final model
    visualize_prediction(global_model, test_dataloader, device)
    ''' I put the visualization here for simplicity. After training is finshed, the server
    has the final aggregated global model. Convenient here because the server already has the globel model state.
    We could also try to visualizate on clients to see how local models behave.
    If we have more time, then we could write a separate script that loads the saved global model.
    That would allow us to run visualization multiple times without retraining, which would be useful with large datasets.'''



def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    # Evaluate model on central data.

    # Load the model and initialize it with the received weights
    model = TrajectoryLSTM(pred=12)
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_full_dataset(batch_size=32, hist=8, pred=12)

    # Evaluate the global model on the test set
    ADE, FDE, miss_rate = test(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({
        "ADE": ADE,
        "FDE": FDE,
        "miss_rate": miss_rate
    })
