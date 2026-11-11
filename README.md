# FlexMOF

FlexMOF generates metal–organic framework structures with a joint multigraph model and periodic boundary conditions. It trains with native PyTorch DistributedDataParallel, generates periodic structures, writes CIF files, and evaluates the resulting structures.

The repository provides training, sampling, and evaluation code together with the model checkpoints and configurations used for the experiments.

## Installation

The reference environment uses Linux, Python 3.11, PyTorch 2.10.0, a CUDA-capable GPU, and a CUDA toolkit compatible with the installed PyTorch build.

```bash
git clone https://github.com/FlexMOF/FlexMOF.git
cd FlexMOF
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python setup_cuda.py build_ext --inplace
export PYTHONPATH="$PWD/src:$PWD"
chmod +x bin/network
```

Download and verify the assets needed for an experiment:

```bash
python download_assets.py structure generation
```

The optional `ablation` and `adsorption` assets provide the PBC-off and adsorption checkpoints. `python download_assets.py all` downloads every asset. Checkpoint hashes are also listed in `checkpoints/checksums.json`.

## Data

Download `bw_db.tar.gz` from [MOFDiff's Zenodo record](https://zenodo.org/records/10806179) and extract it. The archive has MD5 `eea0af77d3282ca44fba242411a50a99`. Point `BW_LMDB` to its extracted `MetalOxo.lmdb`, then prepare an isolated data directory with the supplied split and ordering files:

```bash
export BW_LMDB=/absolute/path/to/MetalOxo.lmdb
mkdir -p "$PWD/dataset/bw_db"
ln -s "$BW_LMDB" "$PWD/dataset/bw_db/MetalOxo.lmdb"
python data/prepare_bw_index.py --bw-db "$PWD/dataset/bw_db"
export FLEXMOF_DATA="$PWD/dataset"
```

## YAML configuration

Training settings are provided in `configs/train_joint.yaml` and `configs/train_composition.yaml`. Command-line arguments take precedence over values in the configuration file.

## Conditional structure generation

Prepare a run and execute one full 5,000-step pilot:

```bash
python reproduce/conditional/prepare_run.py \
  --run-dir "$PWD/runs/on" --mode on --attempt-id review \
  --checkpoint "$PWD/checkpoints/structure_ddpm_e370_inference.pt" \
  --data-root "$FLEXMOF_DATA"

CUDA_VISIBLE_DEVICES=0 python runs/on/worker.py \
  --run-dir "$PWD/runs/on" --attempt-id review --rank 0 --local-gpu 0 --pilot
```

For independent five-candidate sampling, launch ranks `0` through `15` with the same run directory and attempt ID, assigning one GPU to each process. Omit `--pilot`. After all ranks finish, score the saved CIFs:

```bash
python runs/on/score.py --run-dir "$PWD/runs/on" --attempt-id review --processes 16
```

For grouped multi-candidate sampling, use a fresh run prepared as above. You can first check the grouped sampler with:

```bash
CUDA_VISIBLE_DEVICES=0 python reproduce/grouped/worker.py \
  --run-dir "$PWD/runs/on" --attempt-id review --rank 0 --local-gpu 0 --smoke
```

Then launch the conditional worker for ranks `0` through `15` with `--candidate-count 1` and the same attempt ID. After every rank completes, validate these first candidates:

```bash
python reproduce/grouped/prepare_resume.py --run-dir "$PWD/runs/on" \
  --first-attempt review --grouped-attempt grouped
```

Then launch `reproduce/grouped/worker.py` for ranks `0` through `31` with `--run-dir "$PWD/runs/on" --attempt-id grouped`, and run `reproduce/grouped/score.py` with those same arguments. Grouped sampling reuses sequence-encoder outputs across candidates of the same structure; this is an approximation and is not bitwise identical to independent sampling.

For the PBC-off ablation, prepare a fresh directory named `off` with `--mode off` and `checkpoints/structure_ddpm_pbc_off_inference.pt`.

## Unconditional generation

```bash
python reproduce/generation/prepare_run.py \
  --run-dir "$PWD/runs/generation" \
  --checkpoint "$PWD/checkpoints/joint_generation_e226_inference.pt" \
  --data-root "$FLEXMOF_DATA"

CUDA_VISIBLE_DEVICES=0 python reproduce/generation/worker.py \
  --root "$PWD/runs/generation/model" --rank 0 --world 1 --limit 2 --smoke
```

The smoke run uses fewer coordinate steps. For the full experiment, omit `--smoke` and `--limit`, and launch ranks `0` through `15` with `--world 16`, each with one visible GPU.

The validity, novelty, and uniqueness scorer requires MOFChecker 0.9.6, Java 11, and a specific version of [MOFid](https://github.com/snurr-group/mofid). Build MOFid from its pinned source with CMake and a C++ compiler:

```bash
python -m pip install 'mofchecker==0.9.6'
export FLEXMOF_MOFID_ROOT="$HOME/mofid"
git clone https://github.com/snurr-group/mofid.git "$FLEXMOF_MOFID_ROOT"
git -C "$FLEXMOF_MOFID_ROOT" checkout --detach 36873683083c7bc62c1da2062df2833adc35b48a
cmake -S "$FLEXMOF_MOFID_ROOT/openbabel" -B "$FLEXMOF_MOFID_ROOT/openbabel/build" \
  -DCMAKE_INSTALL_PREFIX="$FLEXMOF_MOFID_ROOT/openbabel/installed" \
  -DENABLE_TESTS=OFF -DBUILD_GUI=OFF -DEIGEN3_INCLUDE_DIR="$FLEXMOF_MOFID_ROOT/eigen"
cmake --build "$FLEXMOF_MOFID_ROOT/openbabel/build" --parallel 4
cmake --install "$FLEXMOF_MOFID_ROOT/openbabel/build"
cmake -S "$FLEXMOF_MOFID_ROOT/src" -B "$FLEXMOF_MOFID_ROOT/bin" \
  -DOpenBabel3_DIR="$FLEXMOF_MOFID_ROOT/openbabel/build" \
  -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$FLEXMOF_MOFID_ROOT/bin" --parallel 4
```

Put `bin/network` on `PATH`, set the Java 11 binary directory and dataset paths, and run the scorer after sampling:

```bash
export PATH="$PWD/bin:$PATH"
export FLEXMOF_JAVA_BIN=/absolute/path/to/java/bin
export FLEXMOF_BW_DB="$FLEXMOF_DATA/bw_db"
python reproduce/generation/score.py --root "$PWD/runs/generation/model" --workers 16
```

Add `--smoke` to the scoring command to evaluate the two smoke structures. Individual structures may fail a geometry check; these failures are recorded as invalid. The scorer builds a resumable MOFid reference from the public training split. It verifies the MOFid source commit and records the locally built `bin/sbu` SHA256 so resumed scoring uses the same build. For the 15-rule validity score, activate a separate Python environment and install the pinned [MOFChecker 2.0](https://github.com/Au-4/mofchecker_2.0) source:

```bash
python -m pip install 'pymatgen==2026.5.4' 'ase==3.25.0' \
  'mofchecker @ git+https://github.com/Au-4/mofchecker_2.0.git@a8076cd1dc854960eabe6b56fe08942c6f0d312c'
export FLEXMOF_EVAL_ROOT="$PWD/runs/generation/model"
python reproduce/generation/checker2.py
```

## Pore properties

```bash
python reproduce/properties/properties_method.py init-config \
  --engine-root "$PWD/reproduce/properties/engine" \
  --binary "$PWD/bin/network" --output "$PWD/runs/property_config.json"
python reproduce/properties/properties_method.py preflight --config "$PWD/runs/property_config.json"
python reproduce/properties/properties_method.py calculate \
  --config "$PWD/runs/property_config.json" \
  --original-cif /absolute/path/to/original.cif \
  --sampled-cif /absolute/path/to/sampled.cif \
  --output-dir "$PWD/runs/properties"
```

## Adsorption and periodic attention

The [adsorption experiment guide](docs/adsorption.md) covers ODAC25 fine-tuning, sampling, coverage and energy evaluation. The [periodic attention guide](docs/efficiency.md) describes the paired quality and computational-cost comparison.

## Training

The joint training and composition fine-tuning configurations are in `configs/train_joint.yaml` and `configs/train_composition.yaml`. The training source is in `training_snapshot/src`. The reference configuration uses four nodes with eight GPUs each, a per-GPU batch size of four, and no gradient accumulation. On each node set the node rank to `0` through `3` and the master address to the first node's reachable hostname:

```bash
export FLEXMOF_NNODES=4
export FLEXMOF_NODE_RANK=0
export FLEXMOF_MASTER_ADDR=worker-0
export FLEXMOF_MASTER_PORT=29500
export PYTHONPATH="$PWD/training_snapshot/src:$PWD/src:$PWD"

python -m flexmof.cli train \
  --config configs/train_joint.yaml --data-root "$FLEXMOF_DATA" \
  --output "$PWD/runs/training" --devices 8

python -m flexmof.cli train \
  --config configs/train_composition.yaml --data-root "$FLEXMOF_DATA" \
  --init-checkpoint "$PWD/checkpoints/structure_ddpm_e370_inference.pt" \
  --output "$PWD/runs/composition_training" --devices 8
```

Released checkpoints contain model parameters and configuration for inference and initialization; they do not contain optimizer state. Third-party license notices are in `third_party/`.
