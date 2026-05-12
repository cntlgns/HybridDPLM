"""
Cleanup local experiment files on all RTX3090 nodes.
Submits a short srun job to each node to remove /data_large/unsynced_store/sihun/diffprotein/dplm/.

Usage:
    python cleanup_local.py              # all idle RTX3090 nodes
    python scripts/cleanup_local.py peach tomato  # specific nodes only
"""
import sys
import subprocess

RTX3090_NODES = [
    "kiwi", "mango", "nutella", "orange", "peach",
    "quiznos", "radish", "tomato", "udon", "watermelon", "xoi",
    "yogurt", "vanilla",
] # "lemon",

LOCAL_BASE = "/data_large/unsynced_store/sihun/diffprotein/dplm"

CLEANUP_CMD = f"""
if [ -d "{LOCAL_BASE}" ]; then
    echo "[$(hostname)] Cleaning $( du -sh {LOCAL_BASE} 2>/dev/null | cut -f1 ) from {LOCAL_BASE}"
    rm -rf {LOCAL_BASE}
    echo "[$(hostname)] Done"
else
    echo "[$(hostname)] Nothing to clean"
fi
"""


def get_my_running_nodes():
    """Get nodes where the current user has running jobs."""
    import os
    username = os.environ["USER"]
    out = subprocess.check_output(
        ["squeue", "-u", username, "-p", "rtx3090", "-h", "-o", "%N"],
    ).decode().strip()
    nodes = set()
    for line in out.splitlines():
        if line.strip():
            nodes.add(line.strip())
    return nodes


def cleanup_nodes(nodes: list[str]):
    busy_nodes = get_my_running_nodes()

    procs = []
    for node in nodes:
        if node in busy_nodes:
            print(f"[skip] {node} — has running jobs, skipping")
            continue

        cmd = (
            f"srun -p ada -w {node} --time=00:01:00 -n1 "
            f"--cpus-per-task=1 --mem=1G --job-name=cleanup "
            f"bash -c '{CLEANUP_CMD}'"
        )
        print(f"[submit] {node}")
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        procs.append((node, p))

    for node, p in procs:
        stdout, stderr = p.communicate()
        out = stdout.decode().strip()
        err = stderr.decode().strip()
        if out:
            print(out)
        if p.returncode != 0:
            print(f"[error] {node}: {err}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        nodes = sys.argv[1:]
    else:
        nodes = RTX3090_NODES

    cleanup_nodes(nodes)
