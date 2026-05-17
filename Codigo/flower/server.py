import sys
from unittest import result
import flwr as fl
import time
import numpy as np
import json
import os
import argparse
import matplotlib
from pathlib import Path

if "--run" in sys.argv:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"

# --- Structure to store global metrics ---
global_metrics = {
    "round_times": [],
    "round_accuracies": [],
    "round_losses": [],
    "client_exec_times": [],
    "client_train_acc": []
}

# --- Custom callback for metrics collection ---
class MetricsFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        round_start = time.time()
        aggregated_params, metrics_agg = super().aggregate_fit(rnd, results, failures)

        # Extract client execution times and accuracies
        exec_times = [fit_res.metrics["exec_time"] for _, fit_res in results if "exec_time" in fit_res.metrics]
        train_accs = [fit_res.metrics["train_acc"] for _, fit_res in results if "train_acc" in fit_res.metrics]

        global_metrics["client_exec_times"].append(np.mean(exec_times))
        global_metrics["client_train_acc"].append(np.mean(train_accs))

        print(f"\n[Round {rnd}] Mean client time: {np.mean(exec_times):.2f}s | Mean local accuracy: {np.mean(train_accs)*100:.2f}%")

        global_metrics["round_times"].append(time.time() - round_start)
        return aggregated_params, metrics_agg

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated_loss, metrics_agg = super().aggregate_evaluate(rnd, results, failures)
        accs = [res.metrics["acc"] for _, res in results if "acc" in res.metrics]
        losses = [res.metrics["loss"] for _, res in results if "loss" in res.metrics]

        global_metrics["round_accuracies"].append(np.mean(accs))
        global_metrics["round_losses"].append(np.mean(losses))

        print(f"[Round {rnd}] Global Accuracy: {np.mean(accs)*100:.2f}% | Global Loss: {np.mean(losses):.4f}")
        return aggregated_loss, metrics_agg


# --- Strategy Configuration ---
strategy = MetricsFedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,         # 3 clients for training
    min_evaluate_clients=3,
    min_available_clients=3,
)

# --- Server Configuration ---
server_config = fl.server.ServerConfig(num_rounds=1)


# --- Server Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower server for Federated Learning")
    parser.add_argument("--run", type=int, default=None,
                        help="Run ID (used to save numbered metrics)")
    args = parser.parse_args()

    print("Starting Flower server with advanced metrics...\n")
    start_time = time.time()

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=server_config,
        strategy=strategy
    )

    total_time = time.time() - start_time

    # --- Create results directory ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Final Distributed Performance Report ---
    print("\n==============================")
    print("DISTRIBUTED COMPUTATION REPORT")
    print("==============================")
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Mean time per round: {np.mean(global_metrics['round_times']):.2f} seconds")
    print(f"Mean Global Accuracy: {np.mean(global_metrics['round_accuracies'])*100:.2f}%")
    print(f"Mean Local Accuracy (clients): {np.mean(global_metrics['client_train_acc'])*100:.2f}%")
    print(f"Mean Global Loss: {np.mean(global_metrics['round_losses']):.4f}")

    # --- Save metrics to JSON ---
    metrics_data = {
        "tempo_total": total_time,
        "tempo_medio_rodada": float(np.mean(global_metrics['round_times'])),
        "acuracia_global_media": float(np.mean(global_metrics['round_accuracies'])),
        "acuracia_local_media": float(np.mean(global_metrics['client_train_acc'])),
        "loss_global_medio": float(np.mean(global_metrics['round_losses'])),
        "detalhes": {
            "tempos_rodadas": global_metrics['round_times'],
            "acuracias_globais": [float(x) for x in global_metrics['round_accuracies']],
            "losses_globais": [float(x) for x in global_metrics['round_losses']],
            "tempos_clientes": global_metrics['client_exec_times'],
            "acuracias_locais": [float(x) for x in global_metrics['client_train_acc']]
        }
    }

    # Metrics filename: numbered by run or default
    if args.run is not None:
        metrics_filename = f"metricas_servidor_run_{args.run}.json"
    else:
        metrics_filename = "metricas_fedlearning.json"

    metrics_path = RESULTS_DIR / metrics_filename
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print(f"\n✓ Metrics saved to: {metrics_path}")

    # --- Generate plots ---
    show_plots = args.run is None
    if len(global_metrics['round_accuracies']) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Global Accuracy per Round
        axes[0, 0].plot(global_metrics['round_accuracies'], marker='o', linestyle='-', color='blue')
        axes[0, 0].set_xlabel("Round")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].set_title("Global Accuracy Per Round")
        axes[0, 0].grid(True)

        # Plot 2: Global Loss per Round
        axes[0, 1].plot(global_metrics['round_losses'], marker='o', linestyle='-', color='red')
        axes[0, 1].set_xlabel("Round")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].set_title("Global Loss Per Round")
        axes[0, 1].grid(True)

        # Plot 3: Execution Time per Round
        axes[1, 0].plot(global_metrics['round_times'], marker='o', linestyle='-', color='green')
        axes[1, 0].set_xlabel("Round")
        axes[1, 0].set_ylabel("Time (seconds)")
        axes[1, 0].set_title("Execution Time Per Round")
        axes[1, 0].grid(True)

        # Plot 4: Local vs Global Accuracies
        if len(global_metrics['client_train_acc']) > 0:
            axes[1, 1].plot(global_metrics['round_accuracies'], marker='o', label='Global', linestyle='-', color='blue')
            axes[1, 1].plot(global_metrics['client_train_acc'], marker='s', label='Local Mean', linestyle='--', color='orange')
            axes[1, 1].set_xlabel("Round")
            axes[1, 1].set_ylabel("Accuracy")
            axes[1, 1].set_title("Comparison: Global vs Local Accuracy")
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()
        if args.run is not None:
            graphs_filename = f"metricas_graficos_run_{args.run}.png"
        else:
            graphs_filename = "metricas_graficos.png"
        graphs_path = RESULTS_DIR / graphs_filename
        plt.savefig(graphs_path, dpi=300, bbox_inches='tight')
        print(f"✓ Graphs saved to: {graphs_path}")
        if show_plots:
            plt.show()
        else:
            plt.close()
