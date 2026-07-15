# Deploying the EFP PDA decoder on an Emotiv EPOC X

*Feasibility-grounded deployment summary. All performance numbers are measured on our data
(DMNELF n=17), not assumed. Companion to [INTERIM_FINDINGS.md](INTERIM_FINDINGS.md).*

## 1. Goal & premise
Deploy the **PDA = CEN − DMN** neurofeedback marker (Positive Diametric Activity, Bauer 2019) on a
**portable, EEG-only** headset, using the **frozen fMRI-derived EFP model** (Meir-Hasson single-/multi-
electrode Stockwell 10-band sliding-delay ridge). No scanner in the field: the fMRI-validated decoder
is trained once, then applied to scalp EEG. Honest baseline (our cap, 31-ch, LORO, feedback block,
motion-cleaned target): **personalized CEN r ≈ 0.10–0.11**, replicated across DMNELF + rtBPD nf1/nf2.
Calibration-free cross-subject transfer is weak (~0.07); **one calibration run lifts it to ~0.08–0.12.**

## 2. Expected performance on EPOC X — MEASURED (the go/no-go)
We restricted our EFP to exactly the electrodes EPOC X carries ("virtual-EPOC" ablation,
`efp_cen_clean.py` mode `epoc`) and re-ran the honest LORO. Group feedback-block r (n=17):

| Target | best (1 ch) | all (31 ch) | **epoc (12 ch)** | epoc + AF proxy |
|---|---|---|---|---|
| **CEN / clean** (honest) | 0.110 | 0.103 | **0.095** \* | 0.098 \* |
| CEN / orig (motion-retained) | 0.114 | 0.231 | 0.191 \* | 0.210 \* |
| DMN / orig | 0.097 | 0.214 | 0.217 \* | 0.211 \* |
| PDA / orig | 0.197 | 0.208 | 0.196 \* | 0.202 \* |

\* p < 0.05 (sign-flip). **Verdict: GO.** On the honest motion-cleaned CEN target the EPOC-12 montage
retains **0.095 vs 0.103 (~92%)** of the full 31-channel decoder and stays significant. This is the
key, somewhat surprising result: **even though EPOC X has none of the centro-parietal midline
electrodes where the CEN field peaks (Pz/POz/Cz/P3/P4/CP1/CP2), volume conduction to the posterior
P7/P8/O1/O2 ring recovers most of the signal.** Adding Fp1/Fp2 as AF3/AF4 proxies nudges CEN to 0.098.
Caveat: only CEN has a confound-cleaned target here; DMN/PDA rows use the motion-retained `orig`
targets (inflated ~2×), but show the same ~90–100% epoc/all retention, so the montage is not the
bottleneck for any target.

## 3. Hardware & montage
**EPOC X:** 14 saline felt channels `AF3 F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 AF4`; CMS/DRL reference
near P3/P4; 14-bit; hardware band-pass 0.16–43 Hz + 50/60 Hz notch; internal 128 or 256 SPS;
Bluetooth 5.0 (or USB dongle). Saline (not gel) → faster setup, higher/less-stable impedance.

**Channel map to our research cap:** 12 of 14 EPOC channels are present in our cap and used directly;
**AF3/AF4 are absent** → proxied by the nearest neighbours **Fp1→AF3, Fp2→AF4** (the `epoc_afproxy`
variant, marginally better). **Primary montage risk (quantified in §2):** EPOC X has **no
centro-parietal midline** — the peak of the CEN field. Measured cost: only ~8% relative on clean CEN.

## 4. Raw-EEG access & acquisition (hardware in hand)
Raw EEG on Emotiv is **license-gated** — the consumer app exposes only derived metrics. To stream
volts you need an **EmotivPRO** subscription (or the raw-EEG data add-on). Two supported paths:

- **Cortex API** (JSON over WebSocket, `wss://localhost:6868`): `requestAccess` → `authorize`
  (client id/secret) → `createSession` → `subscribe` stream `"eeg"`. Emits 14-ch samples at
  128/256 SPS with timestamps. Best for a closed-loop client.
- **EmotivPRO → LSL**: EmotivPRO can publish an **LSL outlet**; consume in Python via `pylsl`. Simplest
  for buffered/offline and for reusing our existing Python EFP code.

Setup: wet all 14 felts + CMS/DRL, verify contact quality / impedance in EmotivPRO before every session.

## 5. Real-time pipeline
Mirror the offline EFP exactly so the frozen weights apply:

1. **Acquire** 14-ch @ 128/256 SPS (Cortex or LSL).
2. **Band-pass 1–40 Hz** — or **< 20 Hz neural-only** (our EMG finding: muscle lives > 20 Hz; the
   fingerprint is neural and < 20 Hz loses almost nothing while killing field EMG). Recommended default
   for a moving subject: **< 20 Hz**.
3. **Re-reference to common average** over the 14 channels (EPOC's CMS/DRL ≠ our cap reference;
   common-average + calibration absorbs the montage difference).
4. **Stockwell 10-band** power per electrode (same bands as training).
5. **Sliding-delay design** per electrode (reuse `efp_features.make_delay_design`, `n_delays` from
   `delay_window_s / TR`).
6. **Frozen multivariate ridge** over the EPOC-12 (or +AF-proxy) columns → CEN and DMN estimates →
   **PDA = CEN − DMN**, one value per ~1.2 s.
7. **Feedback** (bar / thermometer) driven by the PDA estimate.

## 6. Calibration protocol
0-shot transfer is weak; **calibrate per subject per session:**
- One **calibration block** (rest + task, ~1 run) at session start.
- Fit the personalized ridge on it (or refit gain/offset of the frozen model).
- **Per-run / per-window z-score** the EEG features → removes headset/electrode gain differences and
  makes the model montage- and session-invariant.
- Apply to the rest of the session. Expected operating point: **CEN ≈ 0.08–0.12** (matches our 1-run
  calibration results; the EPOC montage does not degrade this materially per §2).

## 7. Risks & mitigations
| Risk | Mitigation |
|---|---|
| **Centro-parietal coverage gap** (primary) | Quantified: ~8% relative cost on clean CEN. Posterior ring + AF proxy + calibration. |
| Saline impedance drift over a session | Re-wet felts periodically; impedance check each block. |
| Motion / EMG in the field | **< 20 Hz** band restriction (muscle > 20 Hz); common-average reref. |
| Wireless packet loss / dropouts | Ring buffer, flag/interpolate short gaps, drop windows with missing samples. |
| Reference mismatch (CMS/DRL vs cap) | Common-average reref; calibration absorbs the residual. |
| **No fMRI ground truth in the field** | All validation must be **pre-deployment** (see §8). |

## 8. Validation roadmap
1. **Virtual-EPOC ablation** — done (§2): montage retains ~92% of clean-CEN decoding.
2. **Bench replication** — record the *same subject* on EPOC X vs the research cap (rest + task);
   correlate the two EFP output time courses. Confirms the real headset reproduces the ablation.
3. **Gold standard** — a few subjects with **simultaneous EPOC-X + fMRI**; validate the frozen model's
   PDA estimate against the fMRI CEN/DMN mask-means directly.
4. **Calibrated field pilot** — deploy with the §6 calibration; log EEG + feedback for offline audit.

## Reproduce §2
```
# cluster (features_cache holds the 31-ch Stockwell designs for all 17)
sbatch --array=0-16 efp_cen_clean_slurm.sh efp17_subs.txt      # now emits epoc / epoc_afproxy modes
python efp_cen_group.py                                         # group table incl. epoc columns
```
Files: [scripts/efp_cen_clean.py](scripts/efp_cen_clean.py) (`EPOC12`, `EPOC_AFPROXY`, modes `epoc` /
`epoc_afproxy`), [scripts/efp_cen_group.py](scripts/efp_cen_group.py), results in
[results/cen_clean/](results/cen_clean/).
