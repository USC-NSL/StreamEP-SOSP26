# StreamEP-Artifact

For SOSP'26 artifact review, this repo contains

- StreamInfer: prototype distributed MoE decoding system implementing StreamEP. Currently only work with dummy model weights, yet it performs all model computations and data movements, and can replay authentic expert routing profiled from real model execution of each dataset.
- sglang_dummy_prefill: a fork of sglang with modifications to support decoding from dummy KV-Cache tensors. It can also replay expert routing from profiles. It also supports other minor options (e.g. skipping the 1st dense layer in GLM) for fair comparison with StreamInfer.
- experiment_utils: Instructions and scripts for artifact reviewers.

## Getting started

- Throughput vs. ITL sweep: [`experiment_utils/throughput-itl/`](experiment_utils/throughput-itl/README.md), corresponding to Figure 11 in the paper draft.
- Network-interference tolerance: [`experiment_utils/interference-resist/`](experiment_utils/interference-resist/README.md), corresponding to Figure 13 in the paper draft.

**On the provided cluster**

Each of the above have two scripts, one for StreamInfer and one for baseline, so there are 4 major experiment scripts to run. On sphere, each script is about 30 minutes, so about 2 hours are needed in total. Note that the SPHERE cluster has one node's network that is sometimes flaky (distributed engine boots can hang on it), which is why the run scripts retry failed boots/benchmarks automatically — occasional retry messages in the output are expected.

**On any other cluster**, install both systems on every node first:

- StreamInfer: see [`StreamInfer/readme.md`](StreamInfer/readme.md).
  [`experiment_utils/setup_node.sh`](experiment_utils/setup_node.sh) is the per-node
  script we used to provision SPHERE — it assumes that environment, so elsewhere treat
  it as a **reference** for the steps, not a turnkey installer.
- sglang_dummy_prefill (baseline): see [`sglang_dummy_prefill/readme.md`](sglang_dummy_prefill/readme.md)
  — installs into its **own** torch-2.8 conda env (separate from StreamInfer).