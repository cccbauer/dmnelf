# EFP vs multivariate band-power — head-to-head

EFP = nested-CV v3, single electrode (RidgeCV). Band-power = multivariate 155-feature (31ch × 5 bands), nested (OOF r + inner-CV α); both ridge and elasticnet shown within-subject.
Cross-cohort uses band-power **elasticnet** (its stronger transfer model — ridge replicates only 1/5). `*` = sign-flip p < 0.05. **Replicates** = significant in BOTH rtBPD sessions.

## Within-subject (both nested/de-biased — matched estimator)

Band-power ridge is the L2-matched comparison to EFP's RidgeCV; elasticnet (sparse) shown too.

| Target | EFP (1 electrode) | Band-power ridge | Band-power enet |
|---|---|---|---|
| DMN | 0.142 | 0.147 | 0.215 |
| PDA | 0.169 | 0.129 | 0.144 |
| GSR_CEN | 0.171 | 0.168 | 0.114 |
| GSR_DMN | 0.117 | 0.137 | 0.078 |
| GSR_PDA | 0.141 | 0.132 | 0.152 |

## Cross-cohort double replication (train DMNELF → predict rtBPD)

| Target | EFP nf1 | EFP nf2 | Band-power nf1 | Band-power nf2 | Replicates (both sess.) |
|---|---|---|---|---|---|
| DMN | +0.097* | +0.087* | +0.118* | +0.225* | EFP ✓ / BP ✓ |
| PDA | +0.067* | +0.153* | +0.051* | +0.013  | EFP ✓ |
| GSR_CEN | +0.098* | +0.117* | -0.006  | +0.015  | EFP ✓ |
| GSR_DMN | +0.045* | +0.028  | +0.053* | +0.069* | BP ✓ |
| GSR_PDA | +0.064* | +0.136* | +0.047* | +0.035  | EFP ✓ |

**Double-replication scorecard (of 5 targets): EFP 4/5, band-power 2/5.**

Band-power transfers the RAW/arousal-loaded networks well but collapses on the GSR'd (arousal-removed) and differential (PDA) targets across cohorts; the single-electrode EFP fingerprint holds. Consistent with eeg_bold_coupling's own finding that its cross-cohort signal is largely global arousal.