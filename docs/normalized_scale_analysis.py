"""Recompute normalized label geometry; evaluate ideal grids without reading IQ.

Use --mode stats for numerical outputs and --mode plot for the figure.
No training, source-label editing, or STFT execution is performed.
"""

from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "analysis" / "outputs"
EPS = 1e-9
SCALE_CSV = ROOT / "归一化尺度统计.csv"
GRID_CSV = ROOT / "STFT格点分配评估.csv"
SUMMARY_JSON = ROOT / "归一化尺度核查.json"
FIGURE = ROOT / "归一化尺度分布"
SPLITS = ("train", "test")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ge(a, b):
    return np.asarray(a) >= np.asarray(b) * (1 - EPS)


def lt(a, b):
    return np.asarray(a) < np.asarray(b) * (1 - EPS)


def load_data():
    paths = [SOURCE / "samples.csv", SOURCE / "instances.csv"]
    hashes = {p.name: digest(p) for p in paths}
    s = pd.read_csv(paths[0])
    i = pd.read_csv(paths[1])
    assert s.sample_id.is_unique
    assert not i.duplicated(["sample_id", "signal_id"]).any()
    counts = i.groupby("sample_id").size().reindex(s.sample_id)
    assert np.array_equal(counts.to_numpy(), s.n_signals.to_numpy())
    assert set(i.sample_id) == set(s.sample_id)
    s["T_ms"] = s.duration_ms.round(3)
    d = i.merge(
        s[["sample_id", "split", "bandwidth_mhz", "duration_ms", "T_ms", "n_complex_samples"]],
        on=["sample_id", "split"], validate="many_to_one",
    )
    assert len(d) == len(i)
    assert np.allclose(d.w, d.bw_mhz / d.bandwidth_mhz, rtol=1e-8, atol=1e-12)
    assert np.allclose(d.h, d.dur_ms / d.duration_ms, rtol=1e-8, atol=1e-12)
    assert np.isfinite(d[["w", "h"]]).all().all()
    assert ((d.w > 0) & (d.h > 0) & (d.w <= 1 + EPS) & (d.h <= 1 + EPS)).all()
    d["elongation"] = d.h / d.w
    for p in paths:
        assert digest(p) == hashes[p.name], "Source changed during reading"
    return s, d, hashes


def geometry(g):
    w, h, r = (g[c].to_numpy() for c in ("w", "h", "elongation"))
    row = {"n_samples": g.sample_id.nunique(), "n_instances": len(g)}
    for name, a in (("freq_fraction", w), ("time_fraction", h), ("time_over_freq", r)):
        for q in (.05, .25, .5, .75, .95):
            row[f"{name}_p{round(q * 100):02d}"] = float(np.quantile(a, q))
    longer = h > w * (1 + EPS)
    row["pct_time_longer_than_frequency"] = 100 * longer.mean()
    for k in (4, 10, 100):
        row[f"pct_time_at_least_{k}x_frequency"] = 100 * ge(r, k).mean()
    per_sample = pd.DataFrame({"id": g.sample_id.to_numpy(), "longer": longer, "elongated10": ge(r, 10)})
    row["sample_equal_pct_time_longer"] = 100 * per_sample.groupby("id").longer.mean().mean()
    row["sample_equal_pct_time_at_least_10x"] = 100 * per_sample.groupby("id").elongated10.mean().mean()
    for axis, a in (("freq", w), ("time", h)):
        for k in (1, 2, 4):
            row[f"pct_{axis}_lt_{k}_cells_at_640"] = 100 * lt(a * 640, k).mean()
    return {k: float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v for k, v in row.items()}


def grouping(d, include_joint_class=True):
    specs = [("overall", []), ("B", ["bandwidth_mhz"]), ("T", ["T_ms"]),
             ("BT", ["bandwidth_mhz", "T_ms"]), ("class", ["class"])]
    if include_joint_class:
        specs.append(("BT_class", ["bandwidth_mhz", "T_ms", "class"]))
    for split in SPLITS:
        ds = d[d.split == split]
        for level, cols in specs:
            groups = ds.groupby(cols[0] if len(cols) == 1 else cols, sort=True) if cols else [((), ds)]
            for key, g in groups:
                key = key if isinstance(key, tuple) else (key,)
                meta = dict(zip(cols, key))
                yield {"level": level, "split": split,
                       "B_MHz": meta.get("bandwidth_mhz", np.nan),
                       "T_ms": meta.get("T_ms", np.nan), "class_id": meta.get("class", np.nan)}, g


def grid_metrics(pf, pt):
    row = {"freq_cells_p05": np.quantile(pf, .05), "freq_cells_p50": np.median(pf),
           "time_cells_p05": np.quantile(pt, .05), "time_cells_p50": np.median(pt)}
    for k in (1, 2, 4, 8):
        row[f"pct_freq_lt_{k}"] = 100 * lt(pf, k).mean()
        row[f"pct_time_lt_{k}"] = 100 * lt(pt, k).mean()
        row[f"pct_both_ge_{k}"] = 100 * (ge(pf, k) & ge(pt, k)).mean()
    return row


def compute():
    s, d, hashes = load_data()
    train = d[d.split == "train"]
    rho_global = float(train.elongation.median())
    rho_bt = train.groupby(["bandwidth_mhz", "T_ms"]).elongation.median().rename("rho_bt")
    d = d.merge(rho_bt.reset_index(), on=["bandwidth_mhz", "T_ms"], validate="many_to_one")
    assert d.rho_bt.notna().all()
    scale = pd.DataFrame([{**meta, **geometry(g)} for meta, g in grouping(d)])
    scale.to_csv(SCALE_CSV, index=False, encoding="utf-8-sig")

    allocations = []
    for meta, g in grouping(d, include_joint_class=False):
        n = g.n_complex_samples.to_numpy(dtype=float)
        w, h = g.w.to_numpy(), g.h.to_numpy()
        plans = [("critical_ideal", "square_sqrtN", np.sqrt(n), np.sqrt(n)),
                 ("critical_ideal", "train_global_shape", np.sqrt(n * rho_global), np.sqrt(n / rho_global)),
                 ("critical_ideal", "train_BT_shape", np.sqrt(n * g.rho_bt), np.sqrt(n / g.rho_bt))]
        for length in (2048, 4096, 8192, 16384, 32768):
            plans.append(("critical_ideal", f"fixed_L_{length}", np.full(len(g), length), n / length))
        for freq, time in ((640, 640), (1280, 320), (2560, 160), (5120, 80)):
            plans.append(("fixed_409600_cell_budget", f"F{freq}_K{time}", np.full(len(g), freq), np.full(len(g), time)))
        for family, name, fgrid, tgrid in plans:
            fgrid, tgrid = np.asarray(fgrid), np.asarray(tgrid)
            assert ((fgrid >= 1) & (tgrid >= 1)).all()
            if family == "critical_ideal":
                assert np.allclose(fgrid * tgrid, n)
            else:
                assert np.all(fgrid * tgrid == 409600)
            row = {**meta, "family": family, "strategy": name,
                   "n_samples": g.sample_id.nunique(), "n_instances": len(g),
                   "frequency_grid_p50": np.median(fgrid), "time_grid_p50": np.median(tgrid),
                   "grid_ratio_p50": np.median(fgrid / tgrid),
                   "pct_duration_shorter_than_window": 100 * lt(h * tgrid, 1).mean(),
                   **grid_metrics(w * fgrid, h * tgrid)}
            if family == "critical_ideal":
                row["ideal_window_ms_p50"] = np.median(fgrid / (g.bandwidth_mhz.to_numpy() * 1000))
                row["ideal_grid_spacing_kHz_p50"] = np.median(g.bandwidth_mhz.to_numpy() * 1000 / fgrid)
            allocations.append(row)
    grids = pd.DataFrame(allocations)
    grids.to_csv(GRID_CSV, index=False, encoding="utf-8-sig")

    summary = {
        "date": "2026-11-17", "source_sha256": hashes,
        "n_samples": len(s), "n_instances": len(d),
        "grouping_duration_decimals": 3, "threshold_relative_tolerance": EPS,
        "normalization": "w=bw/B; h=dur/T, retaining original precise T",
        "weighting": "Instance equal by default; sample-equal proportions are separately reported",
        "counts": {sp: geometry(d[d.split == sp]) for sp in SPLITS},
        "train_global_median_h_over_w": rho_global,
        "train_BT_shape_range": [float(rho_bt.min()), float(rho_bt.max())],
        "train_BT_shape_priors": [{"B_MHz": float(b), "T_ms": float(t), "rho": float(r)} for (b, t), r in rho_bt.items()],
        "oracle_geometric_coverage": {sp: {f"pct_possible_both_ge_{k}": float(100 * ge(g.w * g.h * g.n_complex_samples, k * k).mean()) for k in (2, 4, 8)} for sp in SPLITS for g in [d[d.split == sp]]},
        "assumptions": ["No IQ read, STFT execution, or training", "Fs=B and 4 bytes per complex sample remain server-analysis assumptions",
                        "critical_ideal: nonoverlap, no zero padding, continuous ideal F=L and K=N/L; no FFT rounding or tail/border effects",
                        "Cell occupancy is label geometry, not measured spectral visibility, resolving power, or detector performance",
                        "Native grid size is N and varies by sample; fixed_409600_cell_budget is a separate representation comparison",
                        "All shape priors are estimated only on train; test is descriptive evaluation of these fixed priors",
                        "Same number of output cells does not imply identical FFT cost or model resource use"],
        "scale_rows": len(scale), "grid_rows": len(grids),
    }
    for p in (SOURCE / "samples.csv", SOURCE / "instances.csv"):
        assert digest(p) == hashes[p.name]
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"sources": hashes, "scale_rows": len(scale), "grid_rows": len(grids),
                      "overall": scale[scale.level == "overall"].to_dict("records"),
                      "allocation_overall": grids[(grids.level == "overall") & grids.strategy.isin(["square_sqrtN", "train_BT_shape", "F640_K640", "F2560_K160"])][["split", "family", "strategy", "pct_both_ge_4", "pct_time_lt_4", "pct_freq_lt_4"]].to_dict("records")}, ensure_ascii=False, indent=2))


def tradeoff():
    """Physical-window design, checked against all labels without reading IQ."""
    s, d, hashes = load_data()
    existing = pd.read_csv(GRID_CSV)
    existing = existing[existing.family != "physical_window"]
    train = d[d.split == "train"]
    train_b_hz = train.bw_mhz.to_numpy() * 1e6
    tiny = np.isclose(train_b_hz, 6000, rtol=1e-8)
    assert tiny.any()
    assert train_b_hz[~tiny].min() >= 40000 * (1 - EPS)
    plans = [(q, name, taus, 1.0) for q, long_ms in ((3, .5), (4, .7), (5, .9))
             for name, taus in (("short", [.125]), ("long", [long_ms]), ("dual", [.125, long_ms]))]
    plans.append((4, "dual_ENBW_proxy", [.15, 1.0], 1.5))
    new_rows = []
    results = []
    q4_pass = np.zeros(len(d), dtype=bool)

    for q, name, taus, alpha in plans:
        fs = d.bandwidth_mhz.to_numpy() * 1e6
        b = d.bw_mhz.to_numpy() * 1e6
        dur = d.dur_ms.to_numpy() / 1000
        valid_views = []
        display_halfhop_views = []
        for tau_ms in taus:
            w = 2 * np.ceil(fs * tau_ms / 1000 / 2 - 1e-9).astype(np.int64)
            tau = w / fs
            valid_views.append(ge(b * tau / alpha, q) & ge(dur / tau, q))
            display_halfhop_views.append(ge(b * tau / alpha, q) & ge(dur / ((w // 2) / fs), q))
        covered = np.logical_or.reduce(valid_views)
        if name.startswith("dual"):
            # A mathematical upper bound within this stated geometric criterion.
            oracle = ge(b * dur, alpha * q * q)
            assert np.array_equal(covered, oracle)
        tag = f"q{q}_{name}"
        if tag == "q4_dual":
            q4_pass = covered.copy()
        dd = d.assign(passed=covered)
        totals = {}
        for meta, g in grouping(dd, include_joint_class=False):
            row = {**meta, "family": "physical_window", "strategy": tag,
                   "required_q": q, "frequency_proxy_alpha": alpha,
                   "nominal_windows_ms": "|".join(str(t) for t in taus),
                   "n_samples": g.sample_id.nunique(), "n_instances": len(g),
                   "n_covered": int(g.passed.sum()), "n_failed": int((~g.passed).sum()),
                   "pct_both_ge_q": 100 * g.passed.mean(),
                   "coverage_macro_pct": 100 * g.groupby("class").passed.mean().mean()}
            new_rows.append(row)
            if meta["level"] == "overall":
                totals[meta["split"]] = {k: row[k] for k in ["n_instances", "n_covered", "n_failed", "pct_both_ge_q", "coverage_macro_pct"]}
        display_covered = np.logical_or.reduce(display_halfhop_views)
        display_totals = {
            sp: {"n_covered": int(display_covered[d.split.to_numpy() == sp].sum()),
                 "pct": float(100 * display_covered[d.split.to_numpy() == sp].mean())}
            for sp in SPLITS}
        results.append({"q": q, "name": name, "nominal_windows_ms": taus, "frequency_proxy_alpha": alpha,
                        "totals": totals, "halfhop_display_grid_totals": display_totals})

    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
    combined.to_csv(GRID_CSV, index=False, encoding="utf-8-sig")
    dd = d.assign(passed=q4_pass)
    config = []
    for (b_mhz, t_ms), g in dd.groupby(["bandwidth_mhz", "T_ms"]):
        fs = b_mhz * 1e6
        row = {"B_MHz": b_mhz, "T_ms": t_ms, "N_design": int(round(b_mhz * t_ms * 1000))}
        for label, tau_ms in [("short", .125), ("long", .7)]:
            w = int(2 * np.ceil(fs * tau_ms / 1000 / 2 - 1e-9))
            tau_actual_ms = 1000 * w / fs
            row.update({f"{label}_W": w, f"{label}_NFFT": w, f"{label}_hop_half": w // 2,
                        f"{label}_tau_ms": tau_actual_ms,
                        f"{label}_df_kHz": fs / w / 1000,
                        f"{label}_K_no_overlap_ideal": t_ms / tau_actual_ms,
                        f"{label}_K_half_hop_ideal": 2 * t_ms / tau_actual_ms})
        for sp in SPLITS:
            gg = g[g.split == sp]
            row.update({f"{sp}_instances": len(gg), f"{sp}_samples": gg.sample_id.nunique(),
                        f"{sp}_coverage_q4_pct": 100 * gg.passed.mean() if len(gg) else np.nan})
        config.append(row)
    assert len(config) == 31
    pd.DataFrame(config).to_csv(ROOT / "STFT物理窗长配置.csv", index=False, encoding="utf-8-sig")

    failed = dd[~dd.passed]
    assert (failed["class"] == 12).all()
    assert np.allclose(failed.bw_mhz, .006)
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    summary["sampling_rate_status"] = "Fs=B, adopted from the user's confirmation of official dataset documentation on 2026-11-17"
    summary["assumptions"] = [a.replace("Fs=B and 4 bytes per complex sample remain server-analysis assumptions",
                                        "Fs=B is user-confirmed official dataset information; N is taken from the exported sample manifest") for a in summary["assumptions"]]
    summary["physical_window_tradeoff"] = {
        "criterion": "b_hz*(W/Fs)/alpha>=q AND d_s/(W/Fs)>=q in at least one complete view",
        "primary_alpha": 1, "rounding": "W=2*ceil(Fs*tau/2); NFFT=W; hop=W or W/2",
        "main_windows_ms": [.125, .7], "minimum_non6k_train_bandwidth_Hz": float(train_b_hz[~tiny].min()),
        "plans": results,
        "q4_class_coverage": [
            {"split": sp, "class": int(cl), "n": len(g), "covered": int(g.passed.sum()), "pct": float(100 * g.passed.mean())}
            for (sp, cl), g in dd.groupby(["split", "class"])],
        "q4_failed_instances": failed[["sample_id", "signal_id", "class", "bandwidth_mhz", "T_ms", "bw_mhz", "dur_ms"]].to_dict("records"),
        "cost": "Two full views: about 2N coefficients without overlap, about 4N coefficients at half-window hop; excludes borders",
        "limits": "Label geometry only; DFT grid occupancy and window-length time scales are not independent resolving cells. ENBW alpha=1.5 is a separate diagnostic proxy."
    }
    assert summary["source_sha256"] == hashes
    for name, sha in hashes.items():
        assert digest(SOURCE / name) == sha
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"physical_plans": results, "config_rows": len(config),
                      "q4_failures": len(failed), "grid_rows_now": len(combined)}, ensure_ascii=False, indent=2))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.cm import ScalarMappable

    s, d, hashes = load_data()
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    assert summary["source_sha256"] == hashes
    stats = pd.read_csv(SCALE_CSV)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), layout="constrained")
    hist_norm = LogNorm(.01, 5)
    heat_norm = Normalize(0, 100)
    for col, split in enumerate(SPLITS):
        g = d[d.split == split]
        ax = axes[0, col]
        hist, xe, ye = np.histogram2d(np.clip(g.w, 1e-8, 1), np.clip(g.h, 1e-8, 1),
                                     bins=(np.geomspace(5e-5, 1, 81), np.geomspace(.005, 1, 61)))
        assert int(hist.sum()) == len(g)
        hist = np.ma.masked_equal(100 * hist / len(g), 0)
        ax.pcolormesh(xe, ye, hist.T, norm=hist_norm, cmap="Blues", rasterized=True)
        ax.plot([.005, 1], [.005, 1], color="#3E4349", lw=1, ls="--", label="h = w")
        ax.plot([.0005, .1], [.005, 1], color="#AF772B", lw=1, ls=":", label="h = 10w")
        ax.set(xscale="log", yscale="log", xlim=(5e-5, 1), ylim=(.005, 1.08),
               xlabel="Frequency fraction  w = bandwidth / B", ylabel="Time fraction  h = duration / T",
               title=f"{split}  |  {len(g):,} signal instances")
        z = summary["counts"][split]
        ax.text(.03, .97, f"h > w: {z['pct_time_longer_than_frequency']:.2f}%\nh / w >= 10: {z['pct_time_at_least_10x_frequency']:.2f}%",
                transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": .94, "edgecolor": "none", "pad": 3})
        ax.legend(loc="lower right", fontsize=8, frameon=True, facecolor="white", edgecolor="none")
        ax.text(-.12, 1.05, chr(ord('a') + col), transform=ax.transAxes, weight="bold", fontsize=12)

        joint = stats[(stats.level == "BT") & (stats.split == split)].pivot(index="B_MHz", columns="T_ms", values="pct_time_at_least_10x_frequency")
        joint = joint.reindex(index=[5, 20, 30, 40, 50, 80], columns=[20, 40, 60, 80, 100, 150])
        ax = axes[1, col]
        ax.imshow(np.ma.masked_invalid(joint.to_numpy()), cmap="Blues", norm=heat_norm, aspect="auto")
        ax.set(xticks=range(6), xticklabels=[20, 40, 60, 80, 100, 150], yticks=range(6), yticklabels=[5, 20, 30, 40, 50, 80],
               xlabel="Observation duration T (ms)", ylabel="Observation bandwidth B (MHz)",
               title="Instances with time / frequency fraction >= 10")
        for row in range(6):
            for cc in range(6):
                v = joint.iloc[row, cc]
                ax.text(cc, row, f"{v:.1f}" if pd.notna(v) else "--", ha="center", va="center", fontsize=9,
                        color="white" if pd.notna(v) and v > 58 else "#20252B")
        ax.text(-.12, 1.05, chr(ord('c') + col), transform=ax.transAxes, weight="bold", fontsize=12)
    fig.colorbar(ScalarMappable(norm=hist_norm, cmap="Blues"), ax=axes[0].tolist(), label="Instances per log bin (%)", shrink=.88, pad=.025)
    fig.colorbar(ScalarMappable(norm=heat_norm, cmap="Blues"), ax=axes[1].tolist(), label="Elongated instances (%)", shrink=.88, pad=.025)
    fig.get_layout_engine().set(h_pad=.10, w_pad=.10, hspace=.08, wspace=.02)
    fig.savefig(FIGURE.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(FIGURE.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    print("Saved normalized geometry PNG and SVG; all input instances included.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("stats", "plot", "tradeoff"), default="stats")
    args = parser.parse_args()
    if args.mode == "stats":
        compute()
        tradeoff()
    elif args.mode == "tradeoff":
        tradeoff()
    else:
        plot()
