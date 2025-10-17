from unittest import result
import flwr as fl
import time
import numpy as np

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
    print("Iniciando servidor Flower com métricas avançadas...\n")
    start_time = time.time()

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=server_config,
        strategy=strategy
    )

    total_time = time.time() - start_time

    # --- Relatório Final de Desempenho Distribuído ---
    print("\n==============================")
    print("RELATÓRIO DE COMPUTAÇÃO DISTRIBUÍDA")
    print("==============================")
    print(f"Tempo total de treinamento: {total_time:.2f} segundos")
    print(f"Tempo médio por rodada: {np.mean(global_metrics['round_times']):.2f} segundos")
    print(f"Acurácia Global Média: {np.mean(global_metrics['round_accuracies'])*100:.2f}%")
    print(f"Acurácia Local Média (clientes): {np.mean(global_metrics['client_train_acc'])*100:.2f}%")
    print(f"Loss Global Médio: {np.mean(global_metrics['round_losses']):.4f}")
