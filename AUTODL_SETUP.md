# AutoDL Setup

This note captures the minimal setup used to run Wall-X inference on AutoDL with a single `RTX 4090 24GB` instance.

## Layout

- Code: `/root/wall-x`
- Python environment: `/root/wall-x/.venv`
- Downloaded model: `/root/autodl-tmp/models/wall-oss-flow`
- Recommended place for large models and datasets: `/root/autodl-tmp`

Do not store large checkpoints on `/` unless necessary. `/root/autodl-tmp` has more free space and survives normal shutdowns.

## One-Time Setup

The repo now includes two helper scripts:

- `scripts/autodl_env.sh`: activates the local virtualenv and exports CUDA/model env vars
- `scripts/autodl_run.sh`: one-command entrypoint for smoke tests and VQA
- `scripts/run_fake_inference.sh`: dedicated one-command smoke test
- `scripts/run_vqa_example.sh`: dedicated one-command VQA example
- `scripts/run_vqa_repl.sh`: persistent VQA REPL that loads the model once

Default environment values:

```bash
CUDA_HOME=/usr/local/cuda
WALL_X_MODEL_PATH=/root/autodl-tmp/models/wall-oss-flow
HF_ENDPOINT=https://hf-mirror.com
```

## Usage

Activate the environment manually:

```bash
cd /root/wall-x
source scripts/autodl_env.sh
```

Run a smoke test:

```bash
cd /root/wall-x
bash scripts/autodl_run.sh
```

Run fake inference explicitly:

```bash
cd /root/wall-x
bash scripts/autodl_run.sh fake --seq-length 32
```

Run VQA on the bundled example image:

```bash
cd /root/wall-x
bash scripts/autodl_run.sh vqa
```

Run VQA with a custom question:

```bash
cd /root/wall-x
bash scripts/autodl_run.sh vqa --question "What should you do next?"
```

Run a persistent VQA REPL so the model stays loaded:

```bash
cd /root/wall-x
bash scripts/run_vqa_repl.sh
```

Inside the REPL:

- `:help` shows commands
- `:image /path/to/file.png` switches the image
- `:tokens 128` changes generation length
- `:quit` exits

Drop into a shell with the environment loaded:

```bash
cd /root/wall-x
bash scripts/autodl_run.sh shell
```

## Notes

- The downloaded `wall-oss-flow` snapshot does not contain `config.yml`. The inference scripts were adjusted to fall back to the model directory itself when that file is absent.
- `flash-attn` is not required for the default AutoDL inference path used here. The code now tolerates its absence as long as the model runs with `sdpa`.
- A small compatibility fallback was added for `transformers==4.49.0` because `AttentionInterface` is not exposed there.
- The local CUDA extension `wallx_csrc` is still built and required.

## Dependency Split

### Already enough for inference

The current AutoDL environment is sufficient for:

- model download
- fake inference
- VQA example inference
- CUDA extension build

This path intentionally does **not** require `flash-attn`.

### Install later for training

Before switching to a larger training GPU such as `A800 80G`, install the training extras:

```bash
cd /root/wall-x
bash scripts/install_training_extras.sh
```

That script currently installs:

- `flash-attn==2.7.4.post1`

Training also requires `lerobot` at the commit documented in the main README:

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout c66cd401767e60baece16e1cf68da2824227e076
pip install -e .
```

## A800 Training Layout

The current recommended split for the AutoDL `A800 80G` training test is:

- model: `/root/autodl-fs/models/wall-oss-flow`
- dataset: `/root/autodl-tmp/datasets/lerobot/aloha_mobile_cabinet`
- outputs: `/root/autodl-tmp/outputs/wall-x-train`
- norm stats: `/root/autodl-tmp/norm_stats/aloha_mobile_cabinet_stats.json`

Prepared files:

- `workspace/lerobot_example/config_qact_a800.yml`
- `workspace/lerobot_example/run_a800.sh`
- `workspace/lerobot_example/config_qact_a800_stage1.yml`
- `workspace/lerobot_example/run_a800_stage1.sh`

Notes:

- `batch_size_per_gpu` is set to `1` for the initial validation run.
- `FSDP2` is disabled for the single-GPU A800 setup.
- The original full-run config is long. A more practical stage-1 run is provided with:
  - `num_training_steps: 20000`
  - `num_epoch: 3`
  - `epoch_save_interval: 1`

## File Storage

If AutoDL file storage is mounted later, prefer moving large assets there and then overriding:

```bash
export WALL_X_MODEL_PATH=/root/autodl-fs/models/wall-oss-flow
```

or update `scripts/autodl_env.sh`.
