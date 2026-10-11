# Example workflows

Run from the repository root in Bash (Git Bash, WSL, Linux, or macOS).

```bash
./run_examples/training.sh split
./run_examples/training.sh physics
./run_examples/training.sh spatial
```

`physics` trains a delta-calibrated SqrtE DPM example. `spatial` trains all
three spatial ML backends with a 20% measurement budget. Override the Python
command or settings when needed:

```bash
PYTHON="conda run -n beol_tddb python" N_TRIALS=5 ./run_examples/training.sh physics
```

After training, score held-out test lots:

```bash
./run_examples/inference.sh physics
./run_examples/inference.sh gp_boost
./run_examples/inference.sh graphsage
./run_examples/inference.sh spatial_transformer
```
