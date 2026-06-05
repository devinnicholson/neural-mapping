# Modal Workflow

This is the portable replacement for the SLURM smoke workflow. The cluster established the stack; Modal should now provide reproducible GPU jobs and persistent artifacts.

## Setup

Install and authenticate Modal locally:

```bash
python -m pip install ".[modal]"
modal setup
```

The app uses three Modal Volumes:

- `u3dgs-data` mounted at `/workspace/neural-mapping/data`
- `u3dgs-outputs` mounted at `/workspace/neural-mapping/outputs`
- `u3dgs-hf-cache` mounted at `/root/.cache/huggingface`

They are created automatically by `modal.Volume.from_name(..., create_if_missing=True)`.

The image is based on `nvidia/cuda:12.8.1-devel-ubuntu22.04` because `gsplat` needs the CUDA toolkit and `nvcc`, not only PyTorch CUDA wheels.

## First Smoke

Run an environment check:

```bash
modal run modal_app.py --action env
```

This checks `nvidia-smi`, `nvcc`, PyTorch CUDA availability, and the `gsplat` CUDA extension before launching training.

Prepare the Nerfstudio `poster` sample:

```bash
modal run modal_app.py --action prepare
```

Run the full smoke path:

```bash
modal run modal_app.py --action smoke --budget 25 --iterations 3000 --scene-name poster_modal_smoke
```

This executes:

```text
env check -> Hugging Face download -> frame filtering -> split generation -> split materialization -> Splatfacto train -> ns-eval
```

## Separate Train/Eval Commands

Train a prepared split:

```bash
modal run modal_app.py --action train --budget 50 --iterations 10000 --scene-name poster_modal_50_10k
```

Evaluate a run:

```bash
modal run modal_app.py --action eval --budget 50 --scene-name poster_modal_50_10k
```

Print compact metrics:

```bash
modal run modal_app.py --action metrics
```

## GPU Choice

The default GPU is `L4`, matching the cluster smoke environment. Override it at image/function definition time by setting:

```bash
MODAL_GPU=L40S modal run modal_app.py --action smoke --budget 25 --iterations 3000
```

Modal currently supports GPU requests through `@app.function(gpu=...)`; common choices include `L4`, `A10G`, `L40S`, `A100`, and `H100`.

## Notes

- The image pins PyTorch to `2.11.0+cu128`, matching the CUDA 12.8 stack that worked on the cluster.
- Evaluation sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for trusted locally generated Nerfstudio checkpoints.
- Keep checkpoints and rendered images in Modal Volumes, not GitHub.
- Use the `poster` sample only as a smoke test. The real project should move to Replica/ScanNet-style datasets after Modal smoke passes.

## References

- Modal Images: https://modal.com/docs/guide/images
- Modal Volumes: https://modal.com/docs/guide/volumes
- Modal GPU acceleration: https://modal.com/docs/guide/gpu
