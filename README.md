# Neural-QAOA-Squared

The Neural QAOA<sup>2</sup> source code and datasets used in our paper, 'Neural QAOA<sup>2</sup>: Differentiable Joint Graph Partitioning and Parameter
Initialization for Quantum Combinatorial Optimization,' are available here.

```bash
# dataset generation
CUDA_VISIBLE_DEVICES=0  python src/data.py --type critic --model train
CUDA_VISIBLE_DEVICES=0  python src/data.py --type actor --model train
```

```bash
# train quantum evaluator
CUDA_VISIBLE_DEVICES=0  python src/train_critic.py --model train

# train joint generator 
CUDA_VISIBLE_DEVICES=0 python src/train_generator.py --mode train
```


```bash
python scripts/batch-process-parallel.py \
  --algorithm competitors/QAOA-in-QAOA/QAOA_in_QAOA.py \
  --dataset_path data/instances/data/test_instances_only/mc \
  --experiment m \
  --policy JointGenerator+Critic \
  --base qaoa \
  --optimal_values_file data/instances/data/osv.json \
  --runs 10 \
  --gpus 0,1,2,3
```

```bash
. run_policies.sh
```
