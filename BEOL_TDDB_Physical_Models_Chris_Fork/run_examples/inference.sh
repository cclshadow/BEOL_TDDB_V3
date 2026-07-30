# Example commands for running trained models on new wafers.
# Each run writes ttf.csv, binary_label.csv and wafer_maps.png per wafer under --save-path.
# Scaled and delta-trained models: ALWAYS use --config-path -- the scaling factor k
# and the deltas live inside config.json, so pointing at that file picks them up automatically.
#
# Two ways to choose which wafers to run:
#   --manifest split_manifest.json --split test   -> RECOMMENDED. Honest held-out
#         evaluation: runs ONLY the test-split wafers, excluding everything the model
#         was trained/tuned on. (--split may be train | val | test; default test.)
#   --test-path <dir>   -> scans <dir> for lot_*/csv/wafer_* and runs ALL of them.
#         NOTE: pointing this at ./data/ runs the whole dataset (train+val+test mixed),
#         which is NOT a held-out grade -- use it only for scoring brand-new wafers.

# --- Held-out test-split evaluation (recommended) ---
python inference.py --config-path ./configs/GPR/config.json                --manifest split_manifest.json --split test --save-path ./output/GPR/ --num-workers 20
python inference.py --config-path ./configs/DPM_PowerLaw_deltas/config.json --manifest split_manifest.json --split test --save-path ./output/DPM_PowerLaw_deltas/ --num-workers 20
python inference.py --config-path ./configs/DPM_SqrtE_deltas/config.json    --manifest split_manifest.json --split test --save-path ./output/DPM_SqrtE_deltas/ --num-workers 20
python inference.py --config-path ./configs/DPM_InverseE_deltas/config.json --manifest split_manifest.json --split test --save-path ./output/DPM_InverseE_deltas/ --num-workers 20

# --- Directory scan (score arbitrary new wafers laid out as lot_*/csv/wafer_*) ---
# python inference.py --config-path ./configs/DPM_PowerLaw_deltas/config.json --test-path ./data/ --save-path ./output/DPM_PowerLaw_deltas_all/ --num-workers 20
