# efp_epoc

EFP (electrical fingerprint) calibrated for a consumer 14-channel EEG montage
(Emotiv EPOC-style electrode subset), tested against the full 31-channel
DMNELF research cap. Moved here from ad-hoc scratch work in
`/home/cccbauer/` on the cluster (`efp_epoc_model.npz`, `efp_epoc_out/`,
`cenrel_out/`, `export_model.py`, `efp17_subs.txt`) — none of it was
previously tracked in git.

## Contents

- `efp_epoc_model.npz` — group-level calibrated model for the `epoc12` montage,
  built by `export_model.py --montage epoc12`. **Current version: n=19**
  (rebuilt 2026-08-30 on the full recovered cohort — see `efp19_subs.txt`).
  The prior n=17 version is kept as `efp_epoc_model_n17_backup.npz`.
- `export_model.py` — builds a calibrated model file for a given electrode
  montage (`--montage cap31|epoc12`) from the efp_meirhasson features cache
  (`--cache`) and CEN-mean cache (`--cenmean`), for a given subject list
  (`--subs`).
- `efp19_subs.txt` — the current 19-subject DMNELF cohort used to build the
  deployed model (adds dmnelf002/003, recovered via R128 marker
  reconstruction, to the prior `efp17_subs.txt` list, which is kept for
  reference).
- `cen_mean_cache/` — per-subject CEN-mean timeseries/relations
  (`cenmean_*.npz`, `cenrel_*.csv`), covering both DMNELF and rtBPD subjects
  (cross-cohort input for `export_model.py --cenmean`). dmnelf002/003 were
  added via `cen_ceiling_extract.py --cohort dmnelf --subject <sub> --out
  cen_mean_cache` (the actual generator for this cache — lives in
  `/home/cccbauer/cen_ceiling_extract.py` on the cluster, not yet moved
  into a tracked project).
- `results/` — per-subject electrode-montage comparison
  (`efp_cen_clean_{subject}.csv`, via efp_meirhasson's `efp_cen_clean.py`),
  reporting decoding r for CEN/DMN/PDA under `best`/`frontal`/`all`/`epoc`/
  `epoc_afproxy` electrode subsets, each in `orig` and `clean` variants.
  Now complete for all 19 subjects (dmnelf002/003 added 2026-08-30 via
  efp_cen_clean.py).

## Regenerating / extending

To rebuild the model (e.g. after adding more subjects):
```
python export_model.py --montage epoc12 \
  --cache /projects/swglab/data/DMNELF/analysis/fingerprint/19_fingerprint/efp_meirhasson/results/features_cache \
  --cenmean cen_mean_cache --subs efp19_subs.txt \
  --out efp_epoc_model.npz
```
Note `--cache` now points at the `19_fingerprint` features cache (has all 19
subjects); the original `efp_meirhasson/results/features_cache` only has 17.
If adding a subject not yet in `cen_mean_cache/`, run `cen_ceiling_extract.py`
for them first (needs their fMRIPrep BOLD + personal DMN/CEN masks).
