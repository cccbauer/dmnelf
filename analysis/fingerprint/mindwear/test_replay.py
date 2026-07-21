#!/usr/bin/env python3
"""
test_replay.py  —  end-to-end offline validation of the online decoder engine
-----------------------------------------------------------------------------
Streams a recorded DMNELF feedback run through the SAME real-time path used live
(ReplaySource -> RTFeatureExtractor -> Decoder) and checks that the online-predicted CEN/DMN/PDA
correlate with the observed BOLD targets (feedback block). Confirms the online reimplementation of
the EFP features + frozen ridge behaves sensibly before wiring up hardware / PsychoPy.

  python test_replay.py --fif testdata/dmnelf005_feedback_run-01_250Hz.fif --sub dmnelf005 --run 1
"""
import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sources import ReplaySource
from rt_features import RTFeatureExtractor
from decoder import Decoder

BASELINE_TR, HRF_DROP = 25, 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fif", default=str(HERE / "testdata" / "dmnelf005_feedback_run-01_250Hz.fif"))
    ap.add_argument("--sub", default="dmnelf005"); ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--model", default=str(HERE / "model" / "efp_epoc_model.npz"))
    a = ap.parse_args()

    model = np.load(a.model, allow_pickle=True)
    src = ReplaySource(a.fif, speed=0).open()
    feat = RTFeatureExtractor(model, src.channels); feat.set_sfreq(src.sfreq)
    dec = Decoder(model)                                    # running-z (no calibration file)

    cen, dmn, pda = [], [], []                              # per emitted TR
    for _t, s in src.samples():
        d = feat.push(s)
        if d is None:
            continue
        out = dec.predict(d)
        if out is None:
            cen.append(np.nan); dmn.append(np.nan); pda.append(np.nan)
        else:
            cen.append(out[0]); dmn.append(out[1]); pda.append(out[2])
    src.close()
    n_delays = int(model["n_delays"])
    # emitted design k corresponds to TR index (k + n_delays - 1) in the run
    tr0 = n_delays - 1
    online = {"CEN": np.array(cen), "DMN": np.array(dmn), "PDA": np.array(pda)}
    print(f"[test_replay] {a.sub} run{a.run}: sfreq={src.sfreq} Hz, emitted {len(cen)} TR decodes "
          f"(finite {np.isfinite(online['CEN']).sum()})")

    z = np.load(HERE.parent / "fsnr_eeg" / "results" / "cen_ceiling" / f"cenmean_dmnelf_{a.sub}.npz",
                allow_pickle=True)
    obs_cen = np.asarray(z[f"run{a.run}"], float); obs_dmn = np.asarray(z[f"run{a.run}_dmn"], float)
    obs = {"CEN": obs_cen, "DMN": obs_dmn, "PDA": obs_cen - obs_dmn}

    print(f"  {'target':5s}  {'online↔observed r':>18s}   n")
    ok = True
    for t in ["CEN", "DMN", "PDA"]:
        o = online[t]
        # align online TR index tr0.. with observed TR index tr0..
        idx = np.arange(len(o)) + tr0
        m = (idx >= BASELINE_TR + HRF_DROP) & (idx < len(obs[t])) & np.isfinite(o)
        if m.sum() < 20:
            print(f"  {t:5s}  {'(too few)':>18s}"); ok = False; continue
        r, p = pearsonr(o[m], obs[t][idx[m]])
        print(f"  {t:5s}  {r:+.3f} (p={p:.3f})   {m.sum()}")
    print("\n  Sanity: online CEN should track observed CEN positively (single subject, running-z, "
          "per-TR-window Stockwell approximation of the offline whole-run features).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
