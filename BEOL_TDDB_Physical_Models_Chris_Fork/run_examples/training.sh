# Example commands for training the parameters of physical models.
# Recommended: use --manifest split_manifest.json (built by create_split.py). It carries
# the train / val / test split, prevents lot-level leakage, and grades on a held-out test set.
# (Legacy alternative: pass --training-path and --validation-path to two folders -- no held-out test.)

# --- Baseline fits (no deltas) ---
python train.py --pipeline-type GPR                      --manifest split_manifest.json --batch-size 1 --num-workers 20 --n-trials 20 --save-path configs/GPR/
python train.py --pipeline-type DPM --model-type PowerLaw --manifest split_manifest.json --batch-size 1 --num-workers 20 --n-trials 20 --save-path configs/DPM_PowerLaw/
python train.py --pipeline-type DPM --model-type SqrtE    --manifest split_manifest.json --batch-size 1 --num-workers 20 --n-trials 20 --save-path configs/DPM_SqrtE/
python train.py --pipeline-type DPM --model-type InverseE --manifest split_manifest.json --batch-size 1 --num-workers 20 --n-trials 20 --save-path configs/DPM_InverseE/

# --- Delta-trained fits (fit the empirical-fit corrections; --delta-l2 shrinks toward the physics baseline) ---
python train.py --pipeline-type DPM --model-type PowerLaw --train-deltas --delta-l2 0.01 --manifest split_manifest.json --n-trials 40 --num-workers 20 --save-path configs/DPM_PowerLaw_deltas/
python train.py --pipeline-type DPM --model-type SqrtE    --train-deltas --delta-l2 0.01 --manifest split_manifest.json --n-trials 40 --num-workers 20 --save-path configs/DPM_SqrtE_deltas/
python train.py --pipeline-type DPM --model-type InverseE --train-deltas --delta-l2 0.01 --manifest split_manifest.json --n-trials 40 --num-workers 20 --save-path configs/DPM_InverseE_deltas/
