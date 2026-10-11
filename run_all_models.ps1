# Trains the current model set in one go on the shared split manifest.
# Each run saves its config + metadata + convergence curve to ./configs/<name>/.

$MANIFEST = "split_manifest.json"; $BETA = 2.0; $BATCH_SIZE = 1; $NUM_WORKERS = 8

# name; pipeline; model-type; extra-args; n-trials  (deltas add search dims -> use 40+)
$models = @(
    @{ name="Linear";              pipe="Linear"; model=$null;      extra="";                              trials=20 },
    @{ name="GPR";                 pipe="GPR";    model=$null;      extra="";                              trials=20 },
    @{ name="DPM_PowerLaw";        pipe="DPM";    model="PowerLaw"; extra="";                              trials=20 },
    @{ name="DPM_SqrtE";           pipe="DPM";    model="SqrtE";    extra="";                              trials=20 },
    @{ name="DPM_InverseE";        pipe="DPM";    model="InverseE"; extra="";                              trials=20 },
    @{ name="DPM_PowerLaw_deltas"; pipe="DPM";    model="PowerLaw"; extra="--train-deltas --delta-l2 0.01"; trials=40 },
    @{ name="DPM_SqrtE_deltas";    pipe="DPM";    model="SqrtE";    extra="--train-deltas --delta-l2 0.01"; trials=40 },
    @{ name="DPM_InverseE_deltas"; pipe="DPM";    model="InverseE"; extra="--train-deltas --delta-l2 0.01"; trials=40 }
)

$i = 0
foreach ($m in $models) {
    $i++
    $cmd = "py train.py --pipeline-type $($m.pipe) --manifest $MANIFEST --fscore-beta $BETA " +
           "--batch-size $BATCH_SIZE --num-workers $NUM_WORKERS --n-trials $($m.trials) " +
           "--save-path configs/$($m.name)/ $($m.extra)"
    if ($m.model) { $cmd += " --model-type $($m.model)" }

    Write-Host "`n[$i/$($models.Count)] $($m.name)`n  $cmd`n"
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "[FAILED] $($m.name) exited with code $LASTEXITCODE" -ForegroundColor Red }
    else { Write-Host "[DONE] $($m.name)" -ForegroundColor Green }
}

Write-Host "`nAll $($models.Count) runs complete. Results saved under ./configs/" -ForegroundColor Cyan
