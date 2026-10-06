# Risk-Budgeted Inspection under Delayed Feedback

Reproduce risk-ledger inspection experiments with delayed, potentially outcome-dependent feedback and finite inspection capacity. The project includes a discrete-event simulator, 11 comparison policies, fixed scenarios, manufacturing score streams, cached image features, reference measurements, statistical summaries, and invariant tests.

## Reproduce all operating comparisons

Use Python 3.12. The recorded numerical environment uses NumPy 1.26.3, SciPy 1.11.4, pandas 2.2.1, and Numba 0.60.0. Eight CPU workers run independent replications; each numerical solver uses one thread. The complete workflow was executed within 18 GB of shared system memory. Initial Numba compilation adds startup time.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python run_experiments.py --config configs/main.json --workers 8
python run_replays.py --workers 8
python summarize.py
python verify_results.py
```

On Windows, activate the environment with `.venv\Scripts\activate`. Adjust `--workers` to the available CPU capacity. The supplied scored streams make the complete operating comparisons reproducible directly after installation.

`verify_results.py` compares every numerical metric and trace with `reference_results/`, using absolute tolerance 1e-10 and relative tolerance 1e-9. Counts and configuration fields are also checked. The reference outputs were generated with the pinned dependencies on a 64-bit ARM system. Different optimization-library builds can choose different solutions when the allocation linear program has tied optima; retain the pinned environment when checking the reference trajectories.

## Experimental design

`configs/main.json` specifies 850 synthetic scenario–seed combinations, each with 40,000 arriving parts and 11 policies (9,350 policy runs). It includes the main configuration, capacity/delay combinations, calibration-size and risk-budget sensitivity, two outcome-dependent delay settings, and abrupt drift. Every policy in a paired replication receives the same latent parts and potential reporting delays. Seeds are explicit in the configuration. Reused seeds across sensitivity cases create paired comparisons; pooled cases are not treated as independent binomial trials.

`run_replays.py` evaluates 10 policies on SECOM and KolektorSDD2 at two capacities and two delays, with 100 routing seeds per case (8,000 policy runs). SECOM uses 392 chronologically held-out test specimens and KolektorSDD2 uses its 1,004 official test images. Routing seeds repeat decisions on these fixed specimens. Ordinary delays are 2 or 20 slots; defective specimens take five times as long. Each slot introduces four specimens and permits one or two inspection starts. End-of-horizon pending work receives no release credit.

The controller's exposure budget is 0.02 in the main experiment. The confidence allocation is 0.025 for stratum estimation and 0.025 for the realized-defect envelope. The simultaneous exponential boundary uses lambda values 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, and 1.6. Theoretical guarantees require the conditional-stability and accurate-inspection assumptions; finite manufacturing replays are empirical comparisons, and the drift scenario deliberately violates stability.

## Rebuild manufacturing scores

Install the additional pinned dependencies:

```sh
python -m pip install -r requirements-features.txt
python download_data.py secom
python prepare_data.py secom
python prepare_data.py ksdd2
```

The SECOM pipeline fits a 400-tree random forest after training-only median imputation. The chronological split is 783 fitting, 392 calibration, and 392 test specimens. The image pipeline fits standardized logistic regression to the supplied 512-dimensional features, with five-fold selection of regularization on the fitting split only. It uses 1,398 fitting, 933 calibration, and 1,004 test images. Training-score quantiles at 0.60 and 0.85 define the three deployment risk strata. Random seed 20261006 fixes fitting and partitioning.

To recompute image features from the original images:

```sh
python download_data.py ksdd2
python prepare_data.py ksdd2 --refresh-features
```

The downloader verifies SHA-256 against `data/downloads.json` before extraction. The image archive is about 814 MiB. Feature extraction uses a pretrained ResNet-18 in evaluation mode, bilinear resize to 256 × 640, ImageNet normalization, average pooling, and batches of 16. It uses an available Metal GPU or falls back to CPU, with six CPU threads. The recorded feature extraction used the GPU; backend-specific floating-point differences can affect regenerated scores. Cached features preserve the exact inputs to the classifier experiment. Rerun the replay, summary, and verification commands after rebuilding scores.

## Files and results

- `risk_inspection/core.py`: confidence-sequence bounds, allocation linear program, delayed-event simulation, release ledger, and per-run measurements.
- `configs/main.json`: all synthetic scenario parameters, methods, and replication seeds.
- `run_experiments.py`: parallel synthetic experiment driver.
- `run_replays.py`: parallel manufacturing replay driver.
- `download_data.py`: source-data acquisition, checksum validation, and extraction.
- `prepare_data.py`: sensor scoring and frozen-image-feature classifier pipelines.
- `data/scored/`: fixed specimen identifiers, binary labels, scores, strata, partition assignments, and dataset summaries.
- `data/ksdd2_features.npz`: frozen feature matrix in the row order of `data/scored/ksdd2.csv`.
- `results/`: per-run CSV measurements, selected trace arrays, and numerical summaries.
- `reference_results/`: immutable comparison outputs used by verification.
- `summarize.py`: Student-t intervals, paired yield differences, per-scenario one-sided binomial bounds, and descriptive pooled failure counts.
- `verify_results.py`: complete metric and trace comparison against the reference outputs.
- `tests/test_core.py`: numerical confidence-bound, delay-order, credit, inventory, and allocation checks.
- `MANIFEST.sha256`: integrity checksums for the packaged experiment files.
- `LICENSE`, `DATA_LICENSES.md`, `CITATION.cff`: software terms, dataset attribution and terms, and citation metadata.

Key CSV quantities are `good_yield` (good released parts / arrivals), `defect_rate` (escaped defects / releases), `true_risk_rate` (latent predictable defect exposure / releases), `initiation_rate` (inspection starts / arrivals), and `unfinished` (admitted inspections without a result at the horizon). The evaluation-only latent risks never enter the proposed policy's estimator. `potential_violation`, `envelope_violation`, and `cs_violation` indicate any exposure, realized-envelope, or confidence-bound crossing over the run. Diagnostics inspect every stratum, including one absent from the current arrival batch.

Trace arrays have columns: slot, releases, escaped defects, charged risk, latent exposure, waiting queue, pending inspections, ledger balance, followed by one upper bound per stratum. Approximately 50-slot snapshots include the horizon; crossing flags in the CSV are monitored at every decision. `main.csv` has 9,350 rows and `replays.csv` has 8,000 rows.

The recorded main-case good yields are 0.788378 for the risk ledger and 0.312529 for frozen bounds (rounding to six decimals); the paired gain is 0.4758485. The 750 stationary scenario–seed combinations have zero exposure and realized-envelope crossings for the risk ledger, and four stratum confidence-bound crossings. These counts describe the executed experiments and do not replace the probabilistic guarantee.

## Citation and contact

Software authors: Cemil Emre Yavas and Lei Chen, School of Computing, Georgia Southern University. Citation metadata is provided in `CITATION.cff`. Correspondence: Cemil Emre Yavas, cemil@fairesresearch.org.

Original code is MIT-licensed. SECOM-derived data retain CC BY 4.0 and KolektorSDD2-derived data retain CC BY-NC-SA 4.0; attribution and transformation details are in `DATA_LICENSES.md`.
