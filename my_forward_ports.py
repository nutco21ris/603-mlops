import subprocess
import threading
import time
import argparse

def run_port_forward(command):
    while True:
        try:
            process = subprocess.Popen(command, shell=True)
            process.wait()
        except:
            pass
        time.sleep(5)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-argo', action='store_true')
    parser.add_argument('--include-airflow', action='store_true')
    args = parser.parse_args()

    commands = [
        "kubectl port-forward deployment/metadata-service 8080:8080",
        "kubectl port-forward deployment/metaflow-ui-backend-service 8083:8083",
        "kubectl port-forward deployment/metaflow-ui-static-service 3000:3000"
    ]

    if args.include_argo:
        commands.extend([
            "kubectl port-forward -n argo deployment/argo-server 2746:2746",
            "kubectl port-forward -n argo service/argo-events-webhook-eventsource-svc 12000:12000"
        ])

    if args.include_airflow:
        commands.extend([
            "kubectl port-forward -n airflow deployment/airflow-webserver 8081:8080"
        ])

    threads = []
    for cmd in commands:
        thread = threading.Thread(target=run_port_forward, args=(cmd,))
        thread.daemon = True
        thread.start()
        threads.append(thread)
        time.sleep(2)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping port forwards...")

if __name__ == '__main__':
    main() 