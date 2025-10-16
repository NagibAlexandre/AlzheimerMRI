import flwr as fl

# --- Estratégia com FedAvg ---
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=1,         # mínimo 1 cliente
    min_evaluate_clients=1,
    min_available_clients=1,
)

# --- Configuração do servidor ---
server_config = fl.server.ServerConfig(num_rounds=1)

# --- Iniciar servidor ---
if __name__ == "__main__":
    print("Iniciando servidor Flower...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=server_config,
        strategy=strategy
    )
