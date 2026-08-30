# experiment_utils

Instructions and scripts for SOSP'26 artifact reviewers to reproduce the
StreamInfer experiments.

## Experiments

Each experiment's README has the full step-by-step reproduction:

- **Throughput vs. ITL** — [`throughput-itl/README.md`](throughput-itl/README.md).
  The latency/throughput trade-off across request rates, StreamInfer vs. sglang EP.
  Produces `results/comparison.png`.
- **Network-interference tolerance** — [`interference-resist/README.md`](interference-resist/README.md).
  Serving performance at a fixed rate under trace-driven **RDMA** congestion
  (single-link, single-link-2x, all-links, bidir-all-links). Requires **UCX**.
  Produces `results/interference.png`.

## On the provided testbed (SPHERE)

Everything is pre-installed and pre-configured — proceed directly to the experiment
READMEs above, running from the head node (**sgpu6**).

## On any other cluster

1. **Install both systems on every node** — follow the root
   [`README.md`](../README.md)'s Getting started.
2. **Adapt the scripts for your cluster** — the defaults target SPHERE; override the
   cluster variables (hosts, IPs, NIC/RoCE devices) in each experiment's `config.sh`.
   Each README's detailed-configuration section documents every knob.
