# Supplementary Material

*Companion to [MANUSCRIPT.md](MANUSCRIPT.md). Parameters transcribed from the Siemens MAGNETOM
Prisma protocol printouts (DMNELF: `Bauer_DMNELF`; rtBPD localizer: `REMIND/LOC3`; rtBPD feedback:
`REMIND/RT15`).*

## Supplementary Table S1 — MRI acquisition parameters

Scanner: Siemens MAGNETOM Prisma 3T (¹H 123.26 MHz), 64-channel head/neck coil. All functional runs
share one simultaneous-multislice (SMS) gradient-echo EPI sequence; the cohorts differ only in slice
coverage, phase-encode direction, and run counts.

### (a) Functional EPI (feedback, transfer, rest) — shared sequence

| Parameter | DMNELF | rtBPD |
|---|---|---|
| Sequence | SMS GE-EPI (`epfid`/`epboM1`) | SMS GE-EPI (`epfid`) |
| TR / TE | 1200 ms / 30 ms | 1200 ms / 30 ms |
| Voxel size | 2.0 mm isotropic | 2.0 mm isotropic |
| **Slices** | **68** | **72** |
| **Phase-encode dir.** | **A ≫ P** | **P ≫ A** |
| Flip angle | 61° | 61° |
| FoV read / base res. | 256 mm / 128 | 256 mm / 128 |
| Multiband (slice accel.) | 4 | 4 |
| In-plane accel. (GRAPPA) | 2 | 2 |
| Bandwidth | 2170 Hz/px | 2170 Hz/px |
| Echo spacing | 0.57 ms | 0.57 ms |
| Fat suppression | Fat sat. | Fat sat. |
| **Feedback run length** | **125 vol × 4 runs** | **150 vol × 5 runs** |
| Rest-baseline (within feedback) | 25 vol (30 s) | 25 vol (30 s) |
| Resting-state runs | 26 vol (in-session, ×2) + 326/376 vol pre/post | 250 vol (localizer, ×2) + 250 vol pre/post |
| Transfer runs | — | 150 vol (pre + post) |

### (b) Anatomical (T1-weighted MPRAGE)

| Parameter | DMNELF (4-echo vNav) | rtBPD (single-echo) |
|---|---|---|
| Voxel size / slices | 1.0 mm iso / 176 sag | 1.0 mm iso / 176 sag |
| TR / TI | 2530 ms / 1400 ms | 2530 ms / 1400 ms |
| TE | 1.69 / 3.55 / 5.41 / 7.27 ms | 1.92 ms |
| Flip angle | 7° | 7° |
| Accel. (GRAPPA) | 3 | 3 |
| Motion correction | vNav prospective (ABCD) | — |
| Fat suppression | Water excitation | — |

### (c) Field maps (susceptibility-distortion correction)

| Parameter | DMNELF | rtBPD |
|---|---|---|
| Sequence | Spin-echo EPI (`epse`), reversed PE (PA/AP) | Spin-echo EPI (`epse`), reversed PE (PA/AP) |
| TR / TE | 6000 ms / 41 ms | 6000 ms / 43 ms |
| Voxel size / slices | 2.0 mm iso / 68 | 2.0 mm iso / 72 |
| Measurements | 3 | 3 |

**Notes.** (1) rtBPD acquired the personalized DMN/CEN network masks in a **separate localizer
session** (long 250-volume resting-state runs), whereas DMNELF derived masks from **short in-session
resting runs** (§2.3). (2) The two cohorts' feedback paradigm was otherwise identical (30 s rest
baseline → continuous PDA feedback). (3) DMNELF used a 4-echo vNav MPRAGE with prospective motion
correction; rtBPD used a standard MPRAGE in the localizer session.
