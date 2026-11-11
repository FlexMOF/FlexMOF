# Adsorption experiments

These scripts train and evaluate the joint periodic model on ODAC25 host–guest structures. Sampling uses the released adsorption checkpoint, a 5,000-step coordinate DDPM and ten independently seeded candidates per test record. Coverage is evaluated across the reference structures of each system.

## Data and checkpoints

Download the `odac25` directory from the [AtomMOF dataset](https://huggingface.co/datasets/nayoung10/AtomMOF-data/tree/fed7bccddeaf5ec6f20e746beba537e8aa947c8b/odac25) into `data/odac25`. The test workflow uses `odac25_test.lmdb`, `odac25_test_metadata.pkl`, and `test_system_indices.pkl`; training also uses the train and validation files. The repository includes the data-type definitions needed to load those files.

```bash
python download_assets.py structure adsorption
```

## Training

`configs/train_adsorption.json` contains the fine-tuning settings and paths. From the repository root, launch one process per GPU. For the reference four-node, eight-GPU-per-node configuration, set `FLEXMOF_NODE_RANK` to the node's rank and `FLEXMOF_MASTER_ADDR` to the first node's reachable hostname on each node:

```bash
export PYTHONPATH="$PWD/training_snapshot/src:$PWD/src:$PWD"
torchrun --nnodes=4 --nproc_per_node=8 \
  --node_rank="$FLEXMOF_NODE_RANK" --master_addr="$FLEXMOF_MASTER_ADDR" \
  --master_port=29500 reproduce/adsorption/train.py --spec configs/train_adsorption.json
```

The training process saves optimizer and random states for resumption. To resume the same run with the same configuration and world size, add `--resume`.

## Sampling and coverage

```bash
export PYTHONPATH="$PWD/src:$PWD"
python reproduce/adsorption/prepare_run.py \
  --run-dir "$PWD/runs/adsorption" --data-dir "$PWD/data/odac25" \
  --checkpoint "$PWD/checkpoints/adsorption_inference.pt"

CUDA_VISIBLE_DEVICES=0 python reproduce/adsorption/worker.py \
  --run-dir "$PWD/runs/adsorption" --rank 0 --world 1 --smoke --full-pilot --limit 1
```

For the complete test set, omit `--smoke`, `--full-pilot`, and `--limit`. Launch ranks `0` through `world - 1`, with one visible GPU per process, using the same run directory. Once every worker finishes:

```bash
python reproduce/adsorption/finalize_generation.py --run-dir "$PWD/runs/adsorption"
python reproduce/adsorption/score_coverage.py --run-dir "$PWD/runs/adsorption" --workers 16
```

Failed candidate slots remain in the evaluation denominator. Coverage is reported at one and ten candidates, both per system and over all references.

## Adsorption energies

Energy evaluation uses eSEN-ODAC25-Full for fixed-cell ASE FIRE relaxation and UMA-S-1.2 for energy evaluation. Install `fairchem-core==2.22.0` in a compatible separate environment and obtain the checkpoints from [ODAC25](https://huggingface.co/facebook/ODAC25) and [UMA](https://huggingface.co/facebook/UMA); access is subject to their terms. The required checkpoint SHA256 values are recorded in `reproduce/adsorption/energy_eval.py`.

```bash
export FLEXMOF_ADSORPTION_RUN="$PWD/runs/adsorption"
export FLEXMOF_ESEN_CHECKPOINT=/absolute/path/to/esen_sm_odac25_full.pt
mkdir -p "$FLEXMOF_ADSORPTION_RUN/weights"
ln -s /absolute/path/to/uma-s-1p2.pt "$FLEXMOF_ADSORPTION_RUN/weights/uma-s-1p2.pt"

python reproduce/adsorption/prepare_energy_manifest.py --root "$FLEXMOF_ADSORPTION_RUN" --data "$PWD/data/odac25"
python reproduce/adsorption/prepare_guest_bonds.py --root "$FLEXMOF_ADSORPTION_RUN" --data "$PWD/data/odac25"
export ENERGY_ATTEMPT=review
export ENERGY_INPUTS_MANIFEST_SHA256="$(sha256sum "$FLEXMOF_ADSORPTION_RUN/energy/inputs/manifest.json" | cut -d' ' -f1)"
export ENERGY_GUEST_BONDS_SHA256="$(sha256sum "$FLEXMOF_ADSORPTION_RUN/energy/inputs/guest_bonds.json" | cut -d' ' -f1)"
torchrun --standalone --nproc_per_node=8 reproduce/adsorption/energy_eval.py --root "$FLEXMOF_ADSORPTION_RUN"
python reproduce/adsorption/score_energy.py --root "$FLEXMOF_ADSORPTION_RUN"
```

Use a new `ENERGY_ATTEMPT` for a new launch. The scorer records failed calculations and reports the reconstructed bare-host reference used when a system contains multiple host identities.
