<div align="center">
<h1 style="display: flex; justify-content: center; align-items: center; gap: 10px; margin: 0;">
  Neural QAOA²: Differentiable Joint Graph Partitioning and Parameter Initialization for Quantum Combinatorial Optimization
</h1><p><em><a href="https://github.com/0SliverBullet/">Zubin Zheng</a>*¹, <a href="https://github.com/JiahaoWuGit/">Jiahao Wu</a>*¹, <a href="https://github.com/senshineL/">Shengcai Liu</a>¹📧</em></p>

![main_method](./image/main_method.svg)

<p align="center">
<div align="center">
<img src="./image/main_method.svg" alt="overview" style="width: 90%; height: auto;">
</div>

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)]() 
[![alphaXiv](https://img.shields.io/badge/discussion-A42C25?style=for-the-badge&logo=arxiv&logoColor=white&color=blue
)]()
[![Github](https://img.shields.io/badge/Neural%20QAOA%20Squared-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/0SliverBullet/Neural-QAOA-Squared)



<sup>*</sup>Equal contribution <sup>1</sup>Guangdong Provincial Key Laboratory of Brain-Inspired Intelligent Computation, Department of CSE, SUSTech. Correspondence to: [Shengcai Liu](https://senshinel.github.io/)\<liusc3@sustech.edu.cn\>.

---


This repository provides an implementation of **Neural QAOA<sup>2</sup>**, a differentiable framework for **quantum combinatorial optimization** under qubit constraints.

Neural QAOA<sup>2</sup> targets two recurring bottlenecks in divide-and-conquer QAOA pipelines:

1. **Partitioning misalignment**: heuristic graph partition objectives may not correlate with downstream QAOA performance.
2. **Initialization cold start**: topology-blind parameter initialization can slow QAOA optimization.

The core idea is to **jointly learn** (i) graph partitions and (ii) QAOA parameter initializations via gradient signals from a **differentiable quantum evaluator**, enabling topology-aware, performance-aligned decisions.

Double-blind note: this README avoids any personally identifying information. Please refer to the accompanying anonymous manuscript for full experimental context.

## Environment

The recommended setup uses Conda with the pinned dependencies in `environment.yml`.

### Option A (recommended): Conda from `environment.yml`

```bash
conda env create -f environment.yml
conda activate quantum-env
```

### Option B: Conda (manual) + pip

```bash
conda create -n quantum-env python=3.10 -y
conda activate quantum-env
pip install -r requirements.txt
```

Notes:
- The provided `environment.yml` is configured for a CUDA-enabled PyTorch stack (cu121). If you are on CPU-only hardware, replace the Torch-related pip entries with CPU wheels (or install `torch` from the official CPU index) before proceeding.
- On Windows PowerShell, setting CUDA visibility is done via `$env:CUDA_VISIBLE_DEVICES="0"` (Linux examples often use `CUDA_VISIBLE_DEVICES=0 ...`).


## Quick Start (toy example)

Run QAOA<sup>2</sup> on a single small MaxCut instance (no training required):

```bash
python competitors/QAOA-in-QAOA/QAOA_in_QAOA.py --data_path data/instances/data/test_instances_only/mc/bqp50-1.txt --experiment m --runs 1 --depth 1 --sub_size 10 --policy random --base qaoa
```

This command matches the algorithm entry-point argument parser in `competitors/QAOA-in-QAOA/QAOA_in_QAOA.py`.


## Full Pipeline (Neural QAOA<sup>2</sup>)

Neural QAOA<sup>2</sup> consists of (i) dataset generation, (ii) training a differentiable quantum evaluator (critic), (iii) training the joint generator, and (iv) evaluation.

### 1) Dataset generation

The dataset generator is `src/data.py` and supports two dataset types.

```bash
# Generate critic dataset (graphs + partitions + parameters + performance ratio)
python src/data.py --type critic --model train

# Generate actor/generator dataset (graphs)
python src/data.py --type actor --model train
```

CLI arguments (must match exactly):
- `src/data.py`: `--type {critic,actor}` and `--model {train,test}`

### 2) Train critic (differentiable evaluator)

```bash
python src/train_critic.py --model train
```

CLI arguments:
- `src/train_critic.py`: `--model {train,test}`, optional `--pretrained_id`, and `--id` (used for testing).

### 3) Train joint generator

```bash
python src/train_generator.py --mode train
```

Optional flags:
- `--resume` (resume training)
- `--finetune` (load weights but reset optimizer/epochs)
- `--c <path>` (load critic model)
- `--g <path>` (load generator model)

### 4) Batch evaluation on a dataset folder

Use the parallel runner to evaluate an algorithm over a folder of instances.

```bash
python scripts/batch-process-parallel.py \
  --algorithm competitors/QAOA-in-QAOA/QAOA_in_QAOA.py \
  --dataset_path data/instances/data/test_instances_only/mc \
  --experiment m \
  --policy JointGenerator+Critic \
  --base qaoa \
  --depth 1 \
  --optimal_values_file data/instances/data/osv.json \
  --runs 10 \
  --gpus 0,1,2,3
```

For convenience, `run_policies.sh` runs multiple partition policies sequentially (Linux-oriented, uses `taskset`).

## Repository Structure

```text
Neural-QAOA-Squared/
  environment.yml                 Conda environment (pinned)
  README.md                       This file
  LICENSE                         MIT License (anonymous)
  run_policies.sh                 Batch evaluation across policies (Linux)

  src/
    config.py                     Global hyperparameters, paths, seeds
    data.py                       Dataset generation CLI (critic/actor)
    train_critic.py               Critic training/testing CLI
    train_generator.py            Generator training/testing CLI
    local_search.py               Optional local search refinement utilities
    utils.py                      Shared helpers

    models/
      critic_r.py                 Differentiable quantum evaluator (critic)
      generator.py                Joint generator
      partition_generator.py      Partition generator architecture
      param_generator.py          QAOA parameter generator architecture
      gat_encoder.py              GAT-based encoder
      gcn_encoder.py              GCN-based encoder

  scripts/
    batch-process-parallel.py     Parallel evaluation driver (multi-GPU)
    README.md                     Script-level notes

  competitors/
    QAOA-in-QAOA/
      QAOA_in_QAOA.py             QAOA² baseline + policy hooks (algorithm entrypoint)
      QAOA.py                     QAOA subsolver
      utilities.py                Graph parsing, partitioning, helpers

  data/
    instances/                    Provided benchmark instances + optimal values (OPT)
```

## Pretrained Model Weights

Pretrained weights for the differentiable quantum evaluator and joint generator will be released upon paper acceptance.

## License

This project is released under the MIT License.

```text
MIT License

Copyright (c) 2026 Neural QAOA Squared Authors
```
