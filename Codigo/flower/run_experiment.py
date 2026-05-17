import subprocess
import sys
import time
import json
import socket
import argparse
from pathlib import Path

SCRIPT_DIR  = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "resultados"
LOGS_DIR    = RESULTS_DIR / "logs"

N_CLIENTS            = 3
SERVER_READY_TIMEOUT = 120   # segundos aguardando servidor aceitar conexões
RUN_TIMEOUT          = 1200  # 20 min por run (cada run ~10 min)
BETWEEN_RUNS_DELAY   = 20    # segundos entre runs para liberar porta



# Utilitários de rede
def wait_for_server(host="localhost", port=8080, timeout=SERVER_READY_TIMEOUT):
    """Bloqueia até o servidor aceitar TCP na porta, ou até o timeout."""
    print(f"  Aguardando servidor em {host}:{port}", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                print(" OK")
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(2)
            print(".", end="", flush=True)
    print(" TIMEOUT")
    return False


def wait_for_port_free(host="localhost", port=8080, timeout=60):
    """Aguarda até a porta ser liberada (útil entre runs)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                time.sleep(3)
        except (ConnectionRefusedError, OSError):
            return True
    return False


# Execução de uma run individual
def kill_all(procs):
    for proc in procs:
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass


def run_single(run_id, n_clients=N_CLIENTS):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  RUN {run_id}  |  {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    run_start = time.time()

    server_log_path = LOGS_DIR / f"run_{run_id}_server.log"
    server_log = open(server_log_path, "w", encoding="utf-8")

    # --- Iniciar servidor ---
    print(f"  [run {run_id}] Iniciando servidor...")
    server_proc = subprocess.Popen(
        [sys.executable, "server.py", "--run", str(run_id)],
        cwd=SCRIPT_DIR,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,  # Windows
    )

    if not wait_for_server():
        print(f"  [run {run_id}] ERRO: Servidor não iniciou a tempo.")
        server_proc.kill()
        server_log.close()
        return None


    client_procs = []
    client_logs  = []
    print(f"  [run {run_id}] Iniciando {n_clients} clientes...")
    for cid in range(n_clients):
        log_path = LOGS_DIR / f"run_{run_id}_client_{cid}.log"
        log = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "client.py",
             "--run", str(run_id),
             "--client-id", str(cid)],
            cwd=SCRIPT_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        client_procs.append(proc)
        client_logs.append(log)
        time.sleep(0.5)  # evitar condição de corrida na inicialização


    deadline    = time.time() + RUN_TIMEOUT
    timed_out   = False
    for cid, proc in enumerate(client_procs):
        remaining = max(1, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
            rc = proc.returncode
            status = "OK" if rc == 0 else f"código {rc}"
            print(f"  [run {run_id}] Cliente {cid} finalizado ({status})")
        except subprocess.TimeoutExpired:
            print(f"  [run {run_id}] TIMEOUT: Cliente {cid} excedeu {RUN_TIMEOUT}s.")
            timed_out = True
            break

    if timed_out:
        kill_all(client_procs + [server_proc])
        for log in client_logs:
            log.close()
        server_log.close()
        return None


    try:
        server_proc.wait(timeout=60)
        print(f"  [run {run_id}] Servidor finalizado (código: {server_proc.returncode})")
    except subprocess.TimeoutExpired:
        print(f"  [run {run_id}] AVISO: Servidor não encerrou em 60s, forçando.")
        server_proc.kill()

    for log in client_logs:
        log.close()
    server_log.close()

    run_time = time.time() - run_start
    print(f"  [run {run_id}] Tempo total da run: {run_time:.1f}s ({run_time/60:.1f} min)")

    
    metric_keys   = ["accuracy", "precision", "recall", "f1", "specificity", "fpr", "fnr"]
    client_metrics = []
    for cid in range(n_clients):
        mfile = RESULTS_DIR / f"metricas_cliente_{cid}_run_{run_id}.json"
        if mfile.exists():
            with open(mfile) as f:
                client_metrics.append(json.load(f))
        else:
            print(f"  [run {run_id}] AVISO: {mfile.name} não encontrado.")

    if not client_metrics:
        print(f"  [run {run_id}] ERRO: Nenhuma métrica de cliente encontrada.")
        return None

    aggregated = {}
    for key in metric_keys:
        vals = [m[key] for m in client_metrics if key in m]
        aggregated[key] = float(sum(vals) / len(vals)) if vals else None

    server_metrics = {}
    smfile = RESULTS_DIR / f"metricas_servidor_run_{run_id}.json"
    if smfile.exists():
        with open(smfile) as f:
            server_metrics = json.load(f)

    run_result = {
        "run_id":             run_id,
        "run_time_seconds":   run_time,
        "n_clients_reported": len(client_metrics),
        "final_metrics":      aggregated,
        "server_metrics":     server_metrics,
    }

    run_file = RESULTS_DIR / f"metricas_run_{run_id}.json"
    with open(run_file, "w") as f:
        json.dump(run_result, f, indent=4)
    print(f"  [run {run_id}] ✓ Resultado agregado em: {run_file}")

    return run_result



# Orquestrador principal
def print_summary(all_results):
    metric_keys  = ["accuracy", "precision", "recall", "f1", "specificity"]
    metric_labels = {
        "accuracy":    "Accuracy",
        "precision":   "Precision",
        "recall":      "Recall",
        "f1":          "F1-Score",
        "specificity": "Specificity",
    }
    print(f"\n{'='*60}")
    print("  RESUMO FINAL")
    print(f"{'='*60}")
    print(f"  Runs completadas: {len(all_results)}")
    total_time = sum(r["run_time_seconds"] for r in all_results)
    print(f"  Tempo total acumulado: {total_time/60:.1f} min")

    print(f"\n  {'Métrica':<20} {'Média':>9}  (n={len(all_results)} runs)")
    print(f"  {'-'*33}")
    for key in metric_keys:
        vals = [r["final_metrics"][key] for r in all_results
                if r["final_metrics"].get(key) is not None]
        if vals:
            mean = sum(vals) / len(vals)
            label = metric_labels.get(key, key)
            print(f"  {label:<20} {mean*100:>8.2f}%")

    print(f"\n  Execute 'python analyze_results.py' para análise estatística completa.")


def main():
    parser = argparse.ArgumentParser(
        description="Orquestrador de múltiplas runs de Federated Learning"
    )
    parser.add_argument("--runs",       type=int, default=5,
                        help="Número de runs (padrão: 5)")
    parser.add_argument("--clients",    type=int, default=N_CLIENTS,
                        help="Número de clientes por run (padrão: 3)")
    parser.add_argument("--start-from", type=int, default=1,
                        help="ID da primeira run (padrão: 1, útil para retomar)")
    args = parser.parse_args()

    print(f"\nFederated Learning — Experimento Automatizado")
    print(f"  {args.runs} runs × {args.clients} clientes")
    print(f"  Resultados em: {RESULTS_DIR}")
    print(f"  Logs em: {LOGS_DIR}")

    all_results  = []
    failed_runs  = []
    experiment_start = time.time()

    for i in range(args.runs):
        run_id = args.start_from + i

        if i > 0:
            print(f"\nAguardando {BETWEEN_RUNS_DELAY}s + liberação da porta 8080...")
            time.sleep(BETWEEN_RUNS_DELAY)
            if not wait_for_port_free(timeout=60):
                print("AVISO: Porta 8080 pode ainda estar em uso, tentando assim mesmo.")

        result = run_single(run_id, n_clients=args.clients)
        if result is not None:
            all_results.append(result)
        else:
            failed_runs.append(run_id)

    if failed_runs:
        print(f"\nAVISO: Runs com falha: {failed_runs}")

    if all_results:
        print_summary(all_results)

    total_elapsed = time.time() - experiment_start
    print(f"\n  Experimento concluído em {total_elapsed/60:.1f} min total.")


if __name__ == "__main__":
    main()
