# GraphMAE GPU Experiment Summary

Date: 2026-06-19

## Environment

- WSL can see the local GPU: NVIDIA RTX 2000 Ada Generation Laptop GPU.
- Conda environment: `/home/xsj/miniconda3/envs/graphmae`
- PyTorch was switched from CPU build to CUDA build:
  - `pytorch 1.13.1 py3.8_cuda11.7_cudnn8.5.0_0`
  - `pytorch-cuda 11.7`
  - `pytorch-mutex cuda`
- DGL was switched from CPU build to CUDA build:
  - `dgl 1.1.3+cu117`

GPU verification passed under elevated execution:

```text
torch 1.13.1
cuda_available True
cuda_version 11.7
device_count 1
device_name NVIDIA RTX 2000 Ada Generation Laptop GPU
dgl 1.1.3+cu117
```

## Experiments Run

### Graph Classification: MUTAG, GAT/GAT, GPU Smoke Test

Command:

```bash
/usr/bin/time -p /home/xsj/miniconda3/envs/graphmae/bin/python main_graph.py --dataset MUTAG --encoder gat --decoder gat --seeds 0 --device 0 --max_epoch 20 --batch_size 32
```

Result:

```text
#Test_f1: 0.6482+-0.0544
# final_acc: 0.6482+-0.0000
real 11.56
```

Status: completed successfully on GPU.

### Graph Classification: MUTAG, GIN/GIN, Repository Config

Command:

```bash
/usr/bin/time -p /home/xsj/miniconda3/envs/graphmae/bin/python main_graph.py --dataset MUTAG --device 0 --use_cfg --seeds 0
```

Result:

```text
AttributeError: module 'dgl.function' has no attribute 'copy_src'
real 3.57
```

Status: failed due to DGL API compatibility in `graphmae/models/gin.py`. The current DGL 1.1.3 API uses newer message function names, while this repository's GIN implementation calls `fn.copy_src`.

### Graph Classification: MUTAG, GAT/GAT, GPU Full Fallback

Command:

```bash
/usr/bin/time -p /home/xsj/miniconda3/envs/graphmae/bin/python main_graph.py --dataset MUTAG --encoder gat --decoder gat --seeds 0 --device 0 --batch_size 32
```

Result:

```text
#Test_f1: 0.7175+-0.0900
# final_acc: 0.7175+-0.0000
real 45.77
```

Status: completed successfully on GPU.

## Skipped Experiments

### Node Classification

Skipped because the local DGL cache only contains MUTAG:

```text
/home/xsj/.dgl/MUTAG_47395044
/home/xsj/.dgl/MUTAG_47395044/MUTAG
```

No local Cora/Citeseer/PubMed cache was found, and the instruction was to avoid downloading missing datasets for now.

## Notes

- Previous CPU MUTAG GAT/GAT run took about `17.02s` for the comparable 200 epoch setup.
- The GPU full fallback took `45.77s`, which is slower on this tiny dataset. This is expected for very small graph workloads where GPU launch/data-transfer overhead can dominate.
- For more meaningful GPU timing, use larger graph classification datasets or node classification datasets after allowing downloads.

## Recommended Next Steps

- Fix GIN compatibility by replacing the old DGL message API usage such as `fn.copy_src` with the current equivalent, then rerun `MUTAG --use_cfg`.
- Allow downloading small node classification datasets, starting with `cora`, then run `main_transductive.py --dataset cora --device 0 --use_cfg --seeds 0`.
- After Cora works, try Citeseer and PubMed before moving to larger datasets.
