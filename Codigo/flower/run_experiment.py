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
SERVER_READY_TIMEOUT = 120   # seconds waiting for server to accept connections
RUN_TIMEOUT          = 7200  # 2 hours per run
BETWEEN_RUNS_DELAY   = 20    # seconds between runs to free the port



# Network utilities
def wait_for_server(host="localhost", port=8080, timeout=SERVER_READY_TIMEOUT):
    """Block until the server accepts TCP on the port, or until timeout."""
    print(f"  Waiting for server at {host}:{port}", end="", flush=True)
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
    """Wait until the port is free (useful between runs)."""
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

    # --- Start server ---
    print(f"  [run {run_id}] Starting server...")
    server_proc = subprocess.Popen(
        [sys.executable, "server.py", "--run", str(run_id)],
        cwd=SCRIPT_DIR,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,  # Windows
    )

    if not wait_for_server():
        print(f"  [run {run_id}] ERROR: Server did not start in time.")
        server_proc.kill()
        server_log.close()
        return None


    client_procs = []
    client_logs  = []
    print(f"  [run {run_id}] Starting {n_clients} clients...")
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
        time.sleep(0.5)  # avoid starting all clients at the exact same moment


    deadline    = time.time() + RUN_TIMEOUT
    timed_out   = False
    for cid, proc in enumerate(client_procs):
        remaining = max(1, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
            rc = proc.returncode
            status = "OK" if rc == 0 else f"code {rc}"
            print(f"  [run {run_id}] Client {cid} finished ({status})")
        except subprocess.TimeoutExpired:
            print(f"  [run {run_id}] TIMEOUT: Client {cid} exceeded {RUN_TIMEOUT}s.")
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
        print(f"  [run {run_id}] Server finished (code: {server_proc.returncode})")
    except subprocess.TimeoutExpired:
        print(f"  [run {run_id}] WARNING: Server did not stop in 60s, forcing.")
        server_proc.kill()

    for log in client_logs:
        log.close()
    server_log.close()

    run_time = time.time() - run_start
    print(f"  [run {run_id}] Total run time: {run_time:.1f}s ({run_time/60:.1f} min)")

    
    metric_keys   = ["accuracy", "precision", "recall", "f1", "specificity", "fpr", "fnr"]
    client_metrics = []
    for cid in range(n_clients):
        mfile = RESULTS_DIR / f"metricas_cliente_{cid}_run_{run_id}.json"
        if mfile.exists():
            with open(mfile) as f:
                client_metrics.append(json.load(f))
        else:
            print(f"  [run {run_id}] WARNING: {mfile.name} not found.")

    if not client_metrics:
        print(f"  [run {run_id}] ERROR: No client metrics found.")
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
    print(f"  [run {run_id}] ✓ Aggregated result saved to: {run_file}")

    return run_result



# main Orchestrator
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
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Runs completed: {len(all_results)}")
    total_time = sum(r["run_time_seconds"] for r in all_results)
    print(f"  Total accumulated time: {total_time/60:.1f} min")

    print(f"\n  {'Metric':<20} {'Mean':>9}  (n={len(all_results)} runs)")
    print(f"  {'-'*33}")
    for key in metric_keys:
        vals = [r["final_metrics"][key] for r in all_results
                if r["final_metrics"].get(key) is not None]
        if vals:
            mean = sum(vals) / len(vals)
            label = metric_labels.get(key, key)
            print(f"  {label:<20} {mean*100:>8.2f}%")

    print(f"\n  Run 'python analyze_results.py' for full statistical analysis.")


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrator for multiple Federated Learning runs"
    )
    parser.add_argument("--runs",       type=int, default=5,
                        help="Number of runs (default: 5)")
    parser.add_argument("--clients",    type=int, default=N_CLIENTS,
                        help="Number of clients per run (default: 3)")
    parser.add_argument("--start-from", type=int, default=1,
                        help="First run ID (default: 1, useful to resume)")
    args = parser.parse_args()

    print(f"\nFederated Learning — Automated Experiment")
    print(f"  {args.runs} runs × {args.clients} clients")
    print(f"  Results in: {RESULTS_DIR}")
    print(f"  Logs in: {LOGS_DIR}")

    all_results  = []
    failed_runs  = []
    experiment_start = time.time()

    for i in range(args.runs):
        run_id = args.start_from + i

        if i > 0:
            print(f"\nWaiting {BETWEEN_RUNS_DELAY}s + port 8080 to be freed...")
            time.sleep(BETWEEN_RUNS_DELAY)
            if not wait_for_port_free(timeout=60):
                print("WARNING: Port 8080 may still be in use, proceeding anyway.")

        result = run_single(run_id, n_clients=args.clients)
        if result is not None:
            all_results.append(result)
        else:
            failed_runs.append(run_id)

    if failed_runs:
        print(f"\nWARNING: Failed runs: {failed_runs}")

    if all_results:
        print_summary(all_results)

    total_elapsed = time.time() - experiment_start
    print(f"\n  Experiment completed in {total_elapsed/60:.1f} min total.")


if __name__ == "__main__":
    main()
