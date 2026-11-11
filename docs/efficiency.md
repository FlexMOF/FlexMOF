# Periodic attention comparison

The comparison uses the fixed, size-stratified test conditions in `reproduce/efficiency/cohort.json`. Each condition is evaluated with the same checkpoint and candidate seeds using source-atom aggregation and explicit periodic-image attention.

Prepare a PBC-on conditional run as described in the main README, then run each selected sample and candidate in a separate process with one visible GPU. For a short installation check:

```bash
CUDA_VISIBLE_DEVICES=0 python reproduce/efficiency/run.py \
  --run-dir "$PWD/runs/on" --output-dir "$PWD/runs/efficiency_smoke" \
  --sample-id 17469 --candidate-id 0 --mode aggregated --smoke
CUDA_VISIBLE_DEVICES=0 python reproduce/efficiency/run.py \
  --run-dir "$PWD/runs/on" --output-dir "$PWD/runs/efficiency_smoke" \
  --sample-id 17469 --candidate-id 0 --mode explicit --smoke
```

For the full comparison, use a fresh output directory, omit `--smoke`, and run both modes for every sample in the cohort and candidate ID `0` through `4`. These trials can be distributed across independent GPUs. Each trial records the match, synchronized sampling time and PyTorch peak allocated memory; timing includes batch preparation and sampling, and excludes model loading and scoring.

```bash
python reproduce/efficiency/aggregate.py --results-dir "$PWD/runs/efficiency"
```

The aggregator checks all paired trial identities, computes per-stratum medians and paired time ratios, and weights match rates by the test-set stratum populations.

## Training memory comparison

The training comparison uses the paired, fixed BW test structures in `reproduce/training_memory/cohort.json`. Each trial runs in a new process with one GPU, FP32, TF32 disabled, and a 90% PyTorch allocation limit. It performs two warmup and three measured updates, including the model's own loss, backward pass, gradient clipping, optimizer, and applicable EMA. FlexMOF and AtomMOF use their respective representations and training objectives. Peak allocated and reserved PyTorch memory and timing are recorded separately; input loading and batch construction are excluded from the measured interval.

First prepare the BW data in the main README and download the `structure` checkpoint. Obtain the AtomMOF source at commit `7589209ff53514a136f20043e77408108d6f8530`, its BW test LMDB and metadata from [AtomMOF-data](https://huggingface.co/datasets/nayoung10/AtomMOF-data/tree/81a85299823a7e75e0ed7fe39640c99297906080/bwdb), and the official BW `large.ckpt` from [AtomMOF-ckpt](https://huggingface.co/nayoung10/AtomMOF-ckpt/tree/main/bwdb). Set up AtomMOF in a separate environment following its repository instructions; the AtomMOF source and FlexMOF source both use a `src` package, so keep their Python environments separate. The trial checks the checkpoint and each paired structure's content hashes.

```bash
export FLEXMOF_TRAIN_MEMORY_RUN="$PWD/runs/training_memory"
export FLEXMOF_STRUCTURE_CHECKPOINT="$PWD/checkpoints/structure_ddpm_e370_inference.pt"
export ATOMMOF_ROOT=/absolute/path/to/AtomMOF
export ATOMMOF_BW_TEST=/absolute/path/to/bwdb/bwdb_test.lmdb
export ATOMMOF_CHECKPOINT=/absolute/path/to/bwdb/large.ckpt
mkdir -p "$FLEXMOF_TRAIN_MEMORY_RUN"
git clone https://github.com/nayoung10/AtomMOF.git "$ATOMMOF_ROOT"
git -C "$ATOMMOF_ROOT" checkout --detach 7589209ff53514a136f20043e77408108d6f8530
python -c "import json; print(*[r['data_key'] for r in json.load(open('reproduce/training_memory/cohort.json'))['samples']], sep='\\n')" > "$FLEXMOF_TRAIN_MEMORY_RUN/keys.txt"
```

With the FlexMOF environment active, run the following for batch size 4 and again for batch size 1:

```bash
export PYTHONPATH="$PWD/training_snapshot/src:$PWD/src:$PWD"
while read -r key; do
  CUDA_VISIBLE_DEVICES=0 python reproduce/training_memory/bench_flex.py --key "$key" --batch-size 4
done < "$FLEXMOF_TRAIN_MEMORY_RUN/keys.txt"
```

Activate the separate AtomMOF environment and run the corresponding trials, again with batch sizes 4 and 1:

```bash
unset PYTHONPATH
while read -r key; do
  CUDA_VISIBLE_DEVICES=0 python reproduce/training_memory/bench_atommof.py --key "$key" --batch-size 4
done < "$FLEXMOF_TRAIN_MEMORY_RUN/keys.txt"
python reproduce/training_memory/aggregate.py --run-dir "$FLEXMOF_TRAIN_MEMORY_RUN" > "$FLEXMOF_TRAIN_MEMORY_RUN/summary.json"
```

The summary reports successful, failed, and missing trials by structure-size group. Use the same GPU type and software environment as the paper when comparing measured resource usage; other hardware and software may produce different values.
