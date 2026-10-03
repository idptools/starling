# Weighted reconstruction calibration

## Purpose

- Good-faith empirical calibration of distance-map reconstruction weights.
- Issue identification: Davide Mercadante (thanks!).
- Script: calibrate_weighted_stress.py.
- Scope: paired MD/VAE errors and coordinate reconstruction; no modification of production coefficients.

## Production coefficients

- Posterior: seeded sample.
- Calibration: 192 sequences; 16 paired frames per sequence.
- Independent reconstruction validation: 48 additional sequences.
- Separations 1–6, RMS error in Å: 0.1780475402817346, 0.6201276379083143, 0.8913920138032031, 1.0280094843206873, 1.0810112638333218, 1.1067225738830264.
- Long-range RMS error in Å: 0.7130944426723491 + 0.002520151630153439 × sequence length.
- Fit record: weighted_stress_expanded_study.json.
- Frozen panel: weighted_stress_calibration_panel.csv.

## Inputs

- Sequence FASTA and reference-Hamiltonian CLEAN simulation mapping.
- PBC-corrected XTC trajectories and matching CA topologies.
- VAE checkpoint.
- Previously inspected sequence panels.
- Frozen panel for exact reproduction; automatic selection otherwise.
- Selection seed, frame count, reconstruction iterations, and output directory.

## Sequence selection

- Length bands: [30,80), [80,120), [120,180), [180,240), [240,300), [300,380) residues.
- NCPR: (K + R − D − E) count divided by sequence length.
- Acidic: NCPR ≤ −0.15.
- Basic: NCPR ≥ 0.15.
- Hydrophobic: remaining sequences with AILMFWVY fraction ≥ 0.45.
- Balanced: remaining sequences.
- Default quota: eight calibration and two validation sequences per length/chemistry cell.
- Default totals: 192 calibration and 48 validation sequences.
- Family label: parsed protein accession where available; otherwise sequence-name prefix.
- Maximum: eight sequences per family across both sets.
- Randomized candidate order with a recorded seed.
- Select validation sequences first.
- Exclude previously inspected exact sequences from both sets.
- Exclude previously inspected families from validation.
- Exclude validation families from calibration.
- Exclude exact duplicate sequences.
- Require trajectory and topology files.
- Require every quota to be filled.
- Freeze the panel before decoding.
- Independence control: exact-sequence and family labels only; no homology clustering or guarantee of independence from VAE training data.

## Paired maps

- Discard the first 20% of each trajectory.
- Select 32 distinct, evenly spaced frames by default.
- Divide selected frames into eight consecutive blocks, numbered 0–7.
- Calibration frames: even-numbered blocks of calibration sequences.
- Validation frames: odd-numbered blocks of validation sequences.
- Convert CA coordinates from nm to Å.
- Calculate Euclidean CA distance maps.
- Zero-pad maps to the VAE input dimension.
- Encode once; decode posterior mode and a seeded posterior sample separately.
- Crop to sequence length.
- Reflect the decoded upper triangle to obtain symmetric maps.
- Pair each decoded map with its own MD map.

## Error fitting

- Residual: decoded distance minus paired MD distance.
- For each sequence and residue separation, calculate RMS residual across calibration frames and residue pairs.
- Separations 1–6: square root of the equal-sequence mean squared RMS error.
- Separations ≥7: calculate each sequence's RMS over its separation-specific RMS errors.
- Fit the long-range RMS error against sequence length by ordinary least squares with equal sequence weight.
- Long-range model: plateau = intercept + slope × sequence length.
- Short-range prediction: fitted separation error capped at the plateau.
- Require finite, positive predicted errors before reconstruction.
- Off-diagonal stress weights: inverse squared predicted errors.
- Diagonal stress weights: zero.

## Validation

- Calibration cross-validation: exclude an entire family, refit, and predict each excluded sequence's error profile.
- Average profile RMSE equally across sequences.
- Fit pooled coefficients using calibration sequences only.
- Save pooled fits before validation reconstruction.
- Reconstruct validation maps with existing production weights and fitted weights separately.
- Normalize each weight matrix to unit mean over upper-triangle pairs.
- Initialization: classical MDS.
- Refinement: weighted Torch SMACOF.
- Default stopping parameters: 300 iterations; tolerance 10⁻⁶.
- Metrics: pair-distance, long-range pair-distance, and bond-length RMSE; global/local Rg, end-to-end, and bond-angle MAE.
- Local Rg: sliding ten-residue windows.
- Additional metrics: bond-standard-deviation absolute error and contact-probability MAE.
- Contacts: distance <8 Å; residue separation ≥3.
- Tail metric: 95th percentile of frame-level pair-distance RMSE.
- Error-profile metric: RMSE between predicted errors and RMS decoder residuals on validation frames.
- Paired comparison: fitted-weight error minus production-weight error.
- Aggregation: equal-sequence mean, minimum, maximum, and count of improved sequences.
- Uncertainty: 2,000 family-bootstrap replicates; resample families with replacement and retain their sequences.
- Interval: bootstrap 2.5th and 97.5th percentiles; exploratory, not an equivalence or release test.

## Outputs

- Frozen sequence panel.
- Paired MD/decoded-map caches, temporal block labels, and provenance.
- Pooled coefficients and family-held-out predictions.
- Per-sequence validation metrics and paired summaries.
- Trajectory, topology, checkpoint, cache, panel, and source hashes.
- Frame indices, sequences, family labels, seeds, software versions, device, and timings.
- Source archive.
- New output directory required; existing results are not overwritten.
