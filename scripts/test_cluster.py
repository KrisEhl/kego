"""Quick test to verify all Ray cluster nodes and GPUs are available."""

import ray

ray.init()

nodes = ray.nodes()
print(f"Nodes: {len(nodes)}", flush=True)
for node in nodes:
    print(
        f"  {node['NodeManagerAddress']} - GPUs: {node['Resources'].get('GPU', 0)}",
        flush=True,
    )


@ray.remote(num_gpus=1)
def gpu_test():
    import torch

    hostname = ray.util.get_node_ip_address()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    return f"{hostname}: {gpu_name}, CUDA={torch.cuda.is_available()}"


# Launch 2 tasks simultaneously — forces both nodes to be used
print("Launching 2 GPU tasks in parallel...", flush=True)
futures = [gpu_test.remote() for _ in range(2)]
ready, not_ready = ray.wait(futures, num_returns=2, timeout=60)

if len(ready) == 2:
    for r in ray.get(ready):
        print(f"  OK: {r}", flush=True)
else:
    for r in ray.get(ready):
        print(f"  OK: {r}", flush=True)
    for ref in not_ready:
        print(f"  TIMEOUT: task did not complete in 60s", flush=True)
        ray.cancel(ref)

print("Done!", flush=True)
