import datetime
import psutil
import socket
import time

log_file_name = "./port_connections_log.txt"
MONITOR_INTERVAL = 5


with open(log_file_name, "w") as f:
    f.write("-" * 50 + "\n")
    f.write(str(datetime.datetime.now()) + "\n")
    f.write(
        f"Local IP Address: {socket.gethostbyname(socket.gethostname())}"
        + "\n"
    )

prev_connections = set(psutil.net_connections())

while True:
    with open(log_file_name, "a") as f:
        connections = set(psutil.net_connections())
        new_connections = connections - prev_connections
        deprecated_connections = prev_connections - connections
        prev_connections = connections

        f.write(str("-" * 50) + "\n")
        f.write(str(datetime.datetime.now()) + "\n")
        f.write(f"{len(connections)} system-wide connections" + "\n")
        f.write(
            f"{len(new_connections)} new connections: {new_connections}" + "\n"
        )
        f.write(
            f"{len(deprecated_connections)} deprecated connections: {deprecated_connections}"
            + "\n"
        )
    time.sleep(MONITOR_INTERVAL)
