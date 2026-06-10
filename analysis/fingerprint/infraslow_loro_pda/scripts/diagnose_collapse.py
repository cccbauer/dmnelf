"""
diagnose_collapse.py
--------------------
Diagnose WHY the within-feedback LORO ElasticNet collapses to a constant for a
subject (e.g. dmnelf007) while it works for another (e.g. dmnelf008).

Separates three hypotheses:
  (1) ElasticNet over-regularizes  -> compare ElasticNetCV vs RidgeCV vs OLS (LORO).
  (2) Non-stationarity             -> run x run transfer matrix: train Ridge on run i,
                                       predict run j; off-diagonal = cross-run transfer.
  (3) No signal at all             -> per-run univariate channel->PDA r heatmap +
                                       cross-run consistency of those r-vectors; and
                                       within-run 5-fold CV r (refit ceiling).

Also plots the ElasticNetCV regularization path for one LORO fold with the
"predict-the-mean" MSE baseline, to show whether CV genuinely prefers the null.

Usage: python scripts/diagnose_collapse.py --config config.yaml --subjects dmnelf007 dmnelf008
Output: results/figures/<subj>/6_diagnose_collapse.png  (+ printed table)
"""
import argparse, sys, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV, RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_loro import load_config, gather  # noqa: E402
import mne  # noqa: E402
warnings.simplefilter("ignore"); mne.set_log_level("ERROR")

RIDGE_ALPHAS = np.logspace(-1, 4, 30)


def fit_predict(model, Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    model.fit(sc.transform(Xtr), ytr)
    return model.predict(sc.transform(Xte))


def loro_r(runs, make_model):
    """Leave-one-run-out pooled Pearson r for a fresh model() per fold."""
    preds, trues = [], []
    for i in range(len(runs)):
        tr = [runs[j] for j in range(len(runs)) if j != i]
        Xtr = np.vstack([X for _, X in tr]); ytr = np.concatenate([y for y, _ in tr])
        yte, Xte = runs[i]
        preds.append(fit_predict(make_model(), Xtr, ytr, Xte)); trues.append(yte)
    pred = np.concatenate(preds); true = np.concatenate(trues)
    pstd = np.concatenate([p.std() * np.ones_like(p) for p in preds])  # per-fold std
    collapsed = int(np.sum([p.std() < 1e-9 for p in preds]))
    r = pearsonr(pred, true)[0] if pred.std() > 1e-12 else np.nan
    return r, collapsed


def transfer_matrix(runs):
    """T[i,j] = Pearson r predicting run j from a Ridge trained on run i (i!=j).
    Diagonal = within-run 5-fold CV r (refit ceiling)."""
    n = len(runs); T = np.full((n, n), np.nan)
    for i in range(n):
        yi, Xi = runs[i]
        for j in range(n):
            yj, Xj = runs[j]
            if i == j:
                # contiguous folds (no shuffle) so temporal autocorrelation does
                # not leak train->test and inflate the within-run estimate
                kf = KFold(5, shuffle=False); pr = np.zeros_like(yi)
                for tr_idx, te_idx in kf.split(Xi):
                    pr[te_idx] = fit_predict(RidgeCV(alphas=RIDGE_ALPHAS),
                                             Xi[tr_idx], yi[tr_idx], Xi[te_idx])
                T[i, j] = pearsonr(pr, yi)[0] if pr.std() > 1e-12 else np.nan
            else:
                pj = fit_predict(RidgeCV(alphas=RIDGE_ALPHAS), Xi, yi, Xj)
                T[i, j] = pearsonr(pj, yj)[0] if pj.std() > 1e-12 else np.nan
    return T


def per_run_chcorr(runs):
    """R[run, ch] = corr(channel feature, PDA) within that run."""
    return np.array([[pearsonr(X[:, c], y)[0] for c in range(X.shape[1])]
                     for y, X in runs])


def enet_path_fold(runs, mcfg):
    """Refit ElasticNetCV on LORO fold 0 (hold run 0); return (alphas, mean_mse,
    chosen_alpha, var_y) for the best l1_ratio."""
    tr = runs[1:]
    Xtr = np.vstack([X for _, X in tr]); ytr = np.concatenate([y for y, _ in tr])
    sc = StandardScaler().fit(Xtr)
    m = ElasticNetCV(l1_ratio=mcfg["l1_ratios"], n_alphas=mcfg["n_alphas"],
                     cv=mcfg["cv_inner"], max_iter=mcfg["max_iter"])
    m.fit(sc.transform(Xtr), ytr)
    # mse_path_: (n_l1, n_alphas, n_folds); alphas_: (n_l1, n_alphas)
    li = list(mcfg["l1_ratios"]).index(m.l1_ratio_)
    return m.alphas_[li], m.mse_path_[li].mean(1), m.alpha_, ytr.var(), m.l1_ratio_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--subjects", nargs="+", default=["dmnelf007", "dmnelf008"])
    a = ap.parse_args(); cfg = load_config(a.config)
    task = cfg["data"]["task"]; mcfg = cfg["model"]
    di = cfg["data"]["eeg"]["desc_infraslow"]

    models = {
        "ElasticNetCV": lambda: ElasticNetCV(l1_ratio=mcfg["l1_ratios"],
                                             n_alphas=mcfg["n_alphas"],
                                             cv=mcfg["cv_inner"], max_iter=mcfg["max_iter"]),
        "RidgeCV": lambda: RidgeCV(alphas=RIDGE_ALPHAS),
        "OLS": lambda: LinearRegression(),
    }

    for s in a.subjects:
        runs = gather(cfg, s, task, di)
        n = len(runs)
        print(f"\n=== {s}: {n} runs, lengths={[len(y) for y,_ in runs]} ===")
        loro = {name: loro_r(runs, mk) for name, mk in models.items()}
        for name, (r, coll) in loro.items():
            print(f"  LORO {name:13s} r={r:+.3f}   (folds collapsed-to-constant: {coll}/{n})")
        T = transfer_matrix(runs)
        Rc = per_run_chcorr(runs)
        # cross-run consistency = mean pairwise corr of per-run channel-r vectors
        pair = [pearsonr(Rc[i], Rc[j])[0] for i in range(n) for j in range(i+1, n)]
        consistency = float(np.mean(pair))
        offdiag = T[~np.eye(n, dtype=bool)]
        print(f"  Ridge transfer  off-diagonal mean r={np.nanmean(offdiag):+.3f} "
              f"(min={np.nanmin(offdiag):+.3f}, max={np.nanmax(offdiag):+.3f})")
        print(f"  within-run CV r (diagonal) = {np.round(np.diag(T),3)}")
        print(f"  cross-run channel-coupling consistency (mean pairwise r) = {consistency:+.3f}")
        alphas, mse, chosen, vary, l1 = enet_path_fold(runs, mcfg)

        # ---------- figure ----------
        fig, ax = plt.subplots(2, 2, figsize=(13, 10))

        # (A) model comparison
        names = list(loro); rs = [loro[k][0] for k in names]
        bars = ax[0, 0].bar(names, rs, color=["C0", "C1", "C2"])
        for b, k in zip(bars, names):
            ax[0, 0].text(b.get_x()+b.get_width()/2, b.get_height(),
                          f"{loro[k][1]}/{n} collapsed", ha="center",
                          va="bottom" if b.get_height() >= 0 else "top", fontsize=8)
        ax[0, 0].axhline(0, color="k", lw=0.7); ax[0, 0].set_ylabel("LORO pooled r")
        ax[0, 0].set_title("(A) model comparison — is collapse just L1?")

        # (B) run x run transfer matrix
        im = ax[0, 1].imshow(T, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        for i in range(n):
            for j in range(n):
                if not np.isnan(T[i, j]):
                    ax[0, 1].text(j, i, f"{T[i,j]:+.2f}", ha="center", va="center", fontsize=8)
        ax[0, 1].set_xticks(range(n)); ax[0, 1].set_yticks(range(n))
        ax[0, 1].set_xticklabels([f"test r{j+1}" for j in range(n)], fontsize=8)
        ax[0, 1].set_yticklabels([f"train r{i+1}" for i in range(n)], fontsize=8)
        ax[0, 1].set_title(f"(B) Ridge run×run transfer (diag=within-run CV)\n"
                           f"off-diag mean={np.nanmean(offdiag):+.3f}")
        fig.colorbar(im, ax=ax[0, 1], fraction=0.046, pad=0.02)

        # (C) per-run channel->PDA correlation consistency
        im2 = ax[1, 0].imshow(Rc, aspect="auto", cmap="RdBu_r", vmin=-0.4, vmax=0.4)
        ax[1, 0].set_yticks(range(n)); ax[1, 0].set_yticklabels([f"run {i+1}" for i in range(n)])
        ax[1, 0].set_xlabel("EEG channel"); ax[1, 0].set_ylabel("feedback run")
        ax[1, 0].set_title(f"(C) per-run channel→PDA r  (cross-run consistency={consistency:+.2f})")
        fig.colorbar(im2, ax=ax[1, 0], fraction=0.046, pad=0.02)

        # (D) ElasticNet regularization path
        ax[1, 1].plot(alphas, mse, "o-", ms=3, color="C0", label="inner-CV MSE")
        ax[1, 1].axhline(vary, color="C3", ls="--", label=f"predict-mean MSE (var y={vary:.3g})")
        ax[1, 1].axvline(chosen, color="k", ls=":", label=f"chosen α={chosen:.3g}")
        ax[1, 1].set_xscale("log"); ax[1, 1].invert_xaxis()
        ax[1, 1].set_xlabel("α (regularization, log)"); ax[1, 1].set_ylabel("MSE")
        ax[1, 1].legend(fontsize=8)
        ax[1, 1].set_title(f"(D) ElasticNetCV path, fold 1 (l1={l1})\n"
                           f"CV min near var(y) ⇒ null model preferred")

        fig.suptitle(f"{s}: why does LORO ElasticNet collapse?  "
                     f"(ENet LORO r={loro['ElasticNetCV'][0]:+.3f})", fontsize=13)
        fig.tight_layout()
        outdir = Path(cfg["project"]["base_dir"]) / "results" / "figures" / s
        outdir.mkdir(parents=True, exist_ok=True)
        f = outdir / "6_diagnose_collapse.png"
        fig.savefig(f, dpi=130); plt.close(fig)
        print(f"  wrote {f}")


if __name__ == "__main__":
    main()
