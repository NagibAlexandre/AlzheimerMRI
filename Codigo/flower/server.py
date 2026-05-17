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
RESULTS_DIR = BASE_DIR / "resultados"

# --- Estrutura para salvar métricas globais ---
global_metrics = {
    "round_times": [],
    "round_accuracies": [],
    "round_losses": [],
    "client_exec_times": [],
    "client_train_acc": []
}

# --- Callback personalizado para coleta de métricas ---
class MetricsFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        round_start = time.time()
        aggregated_params, metrics_agg = super().aggregate_fit(rnd, results, failures)

        # Extrair tempos e acurácias dos clientes
        exec_times = [fit_res.metrics["exec_time"] for _, fit_res in results if "exec_time" in fit_res.metrics]
        train_accs = [fit_res.metrics["train_acc"] for _, fit_res in results if "train_acc" in fit_res.metrics]

        global_metrics["client_exec_times"].append(np.mean(exec_times))
        global_metrics["client_train_acc"].append(np.mean(train_accs))

        print(f"\n[Round {rnd}] Tempo médio dos clientes: {np.mean(exec_times):.2f}s | Acurácia média local: {np.mean(train_accs)*100:.2f}%")

        global_metrics["round_times"].append(time.time() - round_start)
        return aggregated_params, metrics_agg

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated_loss, metrics_agg = super().aggregate_evaluate(rnd, results, failures)
        accs = [res.metrics["acc"] for _, res in results if "acc" in res.metrics]
        losses = [res.metrics["loss"] for _, res in results if "loss" in res.metrics]

        global_metrics["round_accuracies"].append(np.mean(accs))
        global_metrics["round_losses"].append(np.mean(losses))

        print(f"[Round {rnd}] Acurácia Global: {np.mean(accs)*100:.2f}% | Loss Global: {np.mean(losses):.4f}")
        return aggregated_loss, metrics_agg


# --- Configuração da Estratégia ---
strategy = MetricsFedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,         # 3 clientes para treinamento
    min_evaluate_clients=3,
    min_available_clients=3,
)

# --- Configuração do Servidor ---
server_config = fl.server.ServerConfig(num_rounds=1)


# --- Execução do Servidor ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Servidor Flower para Federated Learning")
    parser.add_argument("--run", type=int, default=None,
                        help="ID da run (usado para salvar métricas numeradas)")
    args = parser.parse_args()

    print("Iniciando servidor Flower com métricas avançadas...\n")
    start_time = time.time()

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=server_config,
        strategy=strategy
    )

    total_time = time.time() - start_time

    # --- Criar pasta de resultados ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Relatório Final de Desempenho Distribuído ---
    print("\n==============================")
    print("RELATÓRIO DE COMPUTAÇÃO DISTRIBUÍDA")
    print("==============================")
    print(f"Tempo total de treinamento: {total_time:.2f} segundos")
    print(f"Tempo médio por rodada: {np.mean(global_metrics['round_times']):.2f} segundos")
    print(f"Acurácia Global Média: {np.mean(global_metrics['round_accuracies'])*100:.2f}%")
    print(f"Acurácia Local Média (clientes): {np.mean(global_metrics['client_train_acc'])*100:.2f}%")
    print(f"Loss Global Médio: {np.mean(global_metrics['round_losses']):.4f}")

    # --- Salvar métricas em JSON ---
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

    # Nome do arquivo: numerado por run ou padrão
    if args.run is not None:
        metrics_filename = f"metricas_servidor_run_{args.run}.json"
    else:
        metrics_filename = "metricas_fedlearning.json"

    metrics_path = RESULTS_DIR / metrics_filename
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print(f"\n✓ Métricas salvas em: {metrics_path}")

    # --- Gerar gráficos ---
    show_plots = args.run is None
    if len(global_metrics['round_accuracies']) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Gráfico 1: Acurácia Global por Rodada
        axes[0, 0].plot(global_metrics['round_accuracies'], marker='o', linestyle='-', color='blue')
        axes[0, 0].set_xlabel("Rodada")
        axes[0, 0].set_ylabel("Acurácia")
        axes[0, 0].set_title("Acurácia Global Por Rodada")
        axes[0, 0].grid(True)

        # Gráfico 2: Loss Global por Rodada
        axes[0, 1].plot(global_metrics['round_losses'], marker='o', linestyle='-', color='red')
        axes[0, 1].set_xlabel("Rodada")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].set_title("Loss Global Por Rodada")
        axes[0, 1].grid(True)

        # Gráfico 3: Tempo de Execução por Rodada
        axes[1, 0].plot(global_metrics['round_times'], marker='o', linestyle='-', color='green')
        axes[1, 0].set_xlabel("Rodada")
        axes[1, 0].set_ylabel("Tempo (segundos)")
        axes[1, 0].set_title("Tempo de Execução Por Rodada")
        axes[1, 0].grid(True)

        # Gráfico 4: Acurácias Local vs Global
        if len(global_metrics['client_train_acc']) > 0:
            axes[1, 1].plot(global_metrics['round_accuracies'], marker='o', label='Global', linestyle='-', color='blue')
            axes[1, 1].plot(global_metrics['client_train_acc'], marker='s', label='Local Média', linestyle='--', color='orange')
            axes[1, 1].set_xlabel("Rodada")
            axes[1, 1].set_ylabel("Acurácia")
            axes[1, 1].set_title("Comparação: Acurácia Global vs Local")
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()
        if args.run is not None:
            graphs_filename = f"metricas_graficos_run_{args.run}.png"
        else:
            graphs_filename = "metricas_graficos.png"
        graphs_path = RESULTS_DIR / graphs_filename
        plt.savefig(graphs_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráficos salvos em: {graphs_path}")
        if show_plots:
            plt.show()
        else:
            plt.close()
