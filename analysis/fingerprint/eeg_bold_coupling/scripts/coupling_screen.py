"""
coupling_screen.py  (STEP A)
----------------------------
Within-subject univariate coupling screen of EEG band-limited power (HRF-convolved)
to the fMRI targets, with a POSITIVE CONTROL, using honest max-statistic
circular-shift nulls.

Two questions:
  (1) POSITIVE CONTROL — is the pipeline sound? Occipital alpha power should
      anti-correlate with visual/posterior BOLD (one of the most replicated
      EEG-fMRI findings). We correlate occipital-alpha BLP with each of the 64
      DiFuMo parcels and test the strongest against a max-stat null over parcels.
      If absent -> the problem is upstream (alignment/artifact/timing), not the model.
  (2) TARGET COUPLING — does ANY channel x band couple to DMN, CEN, or PDA beyond a
      max-stat circular-shift null over the 31 ch x n_band grid?

Per-run z-scoring removes between-run offsets; circular shift is within run
(preserves autocorrelation). Reports per subject + cohort; saves CSV and, for
--detail subjects, a band x channel coupling figure + control bar.

Usage: python scripts/coupling_screen.py --config config.yaml [--detail dmnelf008 dmnelf007]
"""
import argparse, sys, warnings, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bandpower import load_config, canonical_hrf, gather_subject, zscore  # noqa: E402
import mne  # noqa: E402
warnings.simplefilter("ignore"); mne.set_log_level("ERROR")


def zc(x):  # global z-score (after per-run zscore) so corr = mean(product)
    return (x - x.mean(0)) / (x.std(0) + 1e-12)


def maxstat(F, y, run_lens, nperm, rng):
    """F:[T,J] (per-run z), y:[T] (per-run z). Returns r[J], best, p_maxstat."""
    Fz = zc(F); yz = zc(y)
    r = (Fz * yz[:, None]).mean(0)
    obs = np.abs(r); bidx = int(obs.argmax()); obs_max = obs[bidx]
    ge = 0
    for _ in range(nperm):
        sh, off = [], 0
        for L in run_lens:
            seg = y[off:off + L]; k = int(rng.integers(5, max(6, L - 5)))
            sh.append(np.roll(seg, k)); off += L
        ys = zc(np.concatenate(sh))
        if np.abs((Fz * ys[:, None]).mean(0)).max() >= obs_max:
            ge += 1
    return r, bidx, (ge + 1) / (nperm + 1)


def difumo_labels(n=64):
    try:
        from nilearn.datasets import fetch_atlas_difumo
        a = fetch_atlas_difumo(dimension=n, resolution_mm=2)
        lab = a.labels
        if hasattr(lab, "columns"):              # DataFrame
            col = "difumo_names" if "difumo_names" in lab.columns else lab.columns[1]
            return list(lab[col].astype(str))
        return [str(x[1]) for x in lab]          # recarray / list of tuples
    except Exception as e:
        print(f"  (DiFuMo labels unavailable: {e})")
        return None


def build(runs, bands_order):
    """Stack per-run, per-run z-scored: feature F[T,J], labels, targets, occ-alpha, parcels."""
    chs = runs[0]["chs"]; J_labels = [(c, b) for b in bands_order for c in chs]
    Fs, run_lens = [], []
    tgt = {k: [] for k in ("DMN", "CEN", "PDA")}
    parc, occa = [], []
    for r in runs:
        cols = [zscore(r["bp"][b]) for b in bands_order]          # each (n_tr, n_ch)
        Fs.append(np.hstack(cols)); run_lens.append(r["n_tr"])
        for k in tgt:
            tgt[k].append(zscore(r["targets"][k]))
        parc.append(zscore(r["parcels"]))
        occa.append(zscore(r["occ_alpha"]))
    return (np.vstack(Fs), J_labels, {k: np.concatenate(v) for k, v in tgt.items()},
            np.concatenate(occa), np.vstack(parc), run_lens, chs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--detail", nargs="+", default=["dmnelf008", "dmnelf007"])
    a = ap.parse_args(); cfg = load_config(a.config)
    bands_order = list(cfg["bands"]); occ = cfg["data"]["eeg"]["occipital"]
    hcfg = cfg["hrf"]; hrf = canonical_hrf(cfg["data"]["fmri"]["tr"],
                                           hcfg["length_s"], hcfg["delay"], hcfg["undershoot"])
    nperm = cfg["stats"]["nperm"]
    subs = a.subjects or [s for s in cfg["data"]["subjects"]["all"]
                          if s not in set(cfg["data"]["subjects"].get("exclude", []))]
    labels = difumo_labels(cfg["data"]["fmri"]["n_difumo"])
    rng = np.random.default_rng(0)

    print(f"STEP A coupling screen  (band power + HRF, max-stat circ-shift null n={nperm})")
    print(f"{'subject':11s} | {'CONTROL: occ-alpha->bestparcel':34s} | DMN best | CEN best | PDA best")
    print("-" * 110)
    rows = []
    for s in subs:
        runs = gather_subject(cfg, s, hrf)
        if len(runs) < 2:
            print(f"{s:11s} | <2 runs"); continue
        # occipital alpha = mean over occipital channels of alpha-band BLP
        for r in runs:
            ch_idx = [r["chs"].index(c) for c in occ if c in r["chs"]]
            r["occ_alpha"] = r["bp"]["alpha"][:, ch_idx].mean(1)
        F, Jlab, tgt, occa, parc, rl, chs = build(runs, bands_order)

        # (1) positive control: occ-alpha vs 64 parcels
        rc, cbest, cp = maxstat(parc, occa, rl, nperm, rng)
        clab = labels[cbest] if labels else f"parcel{cbest}"
        # (2) targets: occ-alpha grid is F (31*nband)
        tg = {}
        for k in ("DMN", "CEN", "PDA"):
            r_, b_, p_ = maxstat(F, tgt[k], rl, nperm, rng)
            tg[k] = (r_, b_, p_)

        def fmt(k):
            r_, b_, p_ = tg[k]; ch, bd = Jlab[b_]
            return f"{ch}/{bd} r={r_[b_]:+.2f} p={p_:.3f}"
        print(f"{s:11s} | {clab[:20]:20s} r={rc[cbest]:+.2f} p={cp:.3f} | "
              f"{fmt('DMN'):s} | {fmt('CEN'):s} | {fmt('PDA'):s}")

        row = dict(subject=s, ctrl_parcel=cbest, ctrl_label=clab,
                   ctrl_r=float(rc[cbest]), ctrl_p=cp,
                   ctrl_neg_frac=float((rc < 0).mean()))
        for k in ("DMN", "CEN", "PDA"):
            r_, b_, p_ = tg[k]; ch, bd = Jlab[b_]
            row.update({f"{k}_ch": ch, f"{k}_band": bd, f"{k}_r": float(r_[b_]), f"{k}_p": p_})
        rows.append(row)

        if s in a.detail:
            nb, nc = len(bands_order), len(chs)
            fig, ax = plt.subplots(1, 4, figsize=(22, 4.6))
            order = np.argsort(rc)
            ax[0].bar(range(len(rc)), rc[order],
                      color=["C0" if v < 0 else "C3" for v in rc[order]])
            ax[0].axhline(0, color="k", lw=.6)
            ax[0].set_title(f"CONTROL occ-alpha->DiFuMo r\nbest={clab[:24]} r={rc[cbest]:+.2f} p={cp:.3f}",
                            fontsize=9)
            ax[0].set_xlabel("parcel (sorted)"); ax[0].set_ylabel("r")
            for i, k in enumerate(("DMN", "CEN", "PDA")):
                r_, b_, p_ = tg[k]; M = r_.reshape(nb, nc)
                im = ax[i + 1].imshow(M, aspect="auto", cmap="RdBu_r", vmin=-.3, vmax=.3)
                ax[i + 1].set_yticks(range(nb)); ax[i + 1].set_yticklabels(bands_order, fontsize=8)
                ax[i + 1].set_xticks(range(0, nc, 3))
                ax[i + 1].set_xticklabels([chs[j] for j in range(0, nc, 3)], rotation=90, fontsize=5)
                ch, bd = Jlab[b_]
                ax[i + 1].set_title(f"{k}: best {ch}/{bd} r={r_[b_]:+.2f} p={p_:.3f}", fontsize=9)
                fig.colorbar(im, ax=ax[i + 1], fraction=.046, pad=.02)
            fig.suptitle(f"{s}: EEG band-power → fMRI coupling screen (HRF, max-stat null)", fontsize=12)
            fig.tight_layout()
            outd = Path(cfg["project"]["base_dir"]) / "results" / "figures"
            outd.mkdir(parents=True, exist_ok=True)
            fig.savefig(outd / f"coupling_{s}.png", dpi=130); plt.close(fig)
            print(f"            wrote figures/coupling_{s}.png")

    if rows:
        print("-" * 110)
        for k in ("ctrl", "DMN", "CEN", "PDA"):
            rk = "ctrl_r" if k == "ctrl" else f"{k}_r"; pk = "ctrl_p" if k == "ctrl" else f"{k}_p"
            rr = np.array([x[rk] for x in rows]); nsig = sum(1 for x in rows if x[pk] < 0.05)
            print(f"  {k:5s}: mean |r|={np.mean(np.abs(rr)):.3f}  sig(p<.05) {nsig}/{len(rows)}")
        outp = Path(cfg["project"]["base_dir"]) / "results" / "coupling_screen.csv"
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
        print(f"saved: {outp}")


if __name__ == "__main__":
    main()
