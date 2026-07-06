# Frozen v1 results (pre-bulletproofing)

Snapshot of the EFP pipeline's headline results **before** the nested-CV bulletproofing
re-run. Preserved so the v1 numbers can be compared side-by-side with (and reverted to
from) the future de-biased re-run.

- **Git tag:** `efp-preprint-v1`  ·  **branch:** `efp-results-v1`  ·  **commit:** `178bcbe`
- **Date frozen:** 2026-07-06

## Known caveat these numbers carry
The within-subject per-subject/group results use **best-electrode selection scored on the
same CV folds** (see `../../VALIDATION.md`, §4 CRITICAL), so the within-subject
`efp_group_summary.csv` / `efp_persubject_all.csv` r's are optimistically biased
(~+0.05–0.15). The **LOSO** (`efp_group_loso.csv`) and **cross-cohort**
(`cross_cohort_efp_summary_tr{,_nf2}.csv`) transfer results are largely immune (fixed
electrode + held-out/independent data) and are the trustworthy headline.

## Files
- `efp_group_summary.csv`, `efp_persubject_all.csv` — within-subject (biased; see above)
- `efp_group_loso.csv` — leave-one-subject-out transfer (clean)
- `cross_cohort_efp_summary_tr.csv`, `_nf2.csv` (+ per-subject) — DMNELF→rtBPD external
  validation, nf1 & nf2 (clean)
- `MANUSCRIPT_v1.md` — the manuscript draft as of v1

To restore the full v1 tree: `git checkout efp-preprint-v1`.
