"""Forward 127.0.0.1:<port> -> <host>:<port> so jupyter-server-proxy can serve
the Frisky dashboard.

Why this exists: jupyter-server-proxy's `/proxy/<host>:<port>/` route is gated
by `host_allowlist`, which defaults to localhost only. The Frisky scheduler is
bound to the VM's LAN address (the workers have to reach it), so the bare
`/proxy/<port>/` route -- which always targets 127.0.0.1 -- cannot see it.

A plain TCP relay fixes that without restarting the scheduler, which matters
when a run is in flight. It is raw TCP, so the dashboard's WebSocket upgrade
passes through untouched.

    python dashboard_forward.py --to 192.168.1.129:8791 --listen 8791

Then: https://<hub>/user/<user>/proxy/8791/
"""
from __future__ import annotations

import argparse
import socket
import threading


def pipe(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for s in (src, dst):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            s.close()


def handle(client, target):
    try:
        upstream = socket.create_connection(target, timeout=15)
    except OSError:
        client.close()
        return
    upstream.settimeout(None)
    threading.Thread(target=pipe, args=(client, upstream), daemon=True).start()
    threading.Thread(target=pipe, args=(upstream, client), daemon=True).start()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--to", required=True, help="host:port to forward to")
    p.add_argument("--listen", type=int, required=True, help="local port")
    args = p.parse_args()

    host, _, port = args.to.rpartition(":")
    target = (host, int(port))

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.listen))
    srv.listen(128)
    print(f"127.0.0.1:{args.listen} -> {target[0]}:{target[1]}", flush=True)
    while True:
        client, _ = srv.accept()
        handle(client, target)


if __name__ == "__main__":
    main()
