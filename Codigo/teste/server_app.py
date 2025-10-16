"""teste: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters, EvaluateRes, FitRes
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import weighted_loss_avg
from task import Net, get_weights, load_data, test, set_weights
import torch
import logging


def weighted_average(metrics):
    """Custom aggregation function for metrics."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_metrics_aggregation(metrics):
    """Custom aggregation function for fit metrics."""
    train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    train_accuracies = [num_examples * m["train_accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    avg_train_loss = sum(train_losses) / sum(examples)
    avg_train_accuracy = sum(train_accuracies) / sum(examples)
    
    return {"avg_train_loss": avg_train_loss, "avg_train_accuracy": avg_train_accuracy}


def evaluate_global(server_round, parameters, config):
    """Global evaluation function to calculate overall accuracy and loss."""
    model = Net()
    set_weights(model, parameters)
    
    # Load global test dataset (use all data for evaluation)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create a comprehensive global test dataset
    # We'll use a subset of data from each partition for global evaluation
    try:
        # Load data from partition 0 as global test set
        _, global_testloader = load_data(0, 1)  # Use all data as test set
        
        # Evaluate the model
        loss, accuracy = test(model, global_testloader, device)
        
        logging.info(f"=== GLOBAL METRICS - Round {server_round} ===")
        logging.info(f"Global Loss: {loss:.4f}")
        logging.info(f"Global Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logging.info(f"========================================")
        
        return loss, {"global_accuracy": accuracy}
        
    except Exception as e:
        logging.warning(f"Global evaluation failed: {e}")
        return float("inf"), {"global_accuracy": 0.0}


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy with custom aggregation and global evaluation
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
        evaluate_fn=evaluate_global,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
