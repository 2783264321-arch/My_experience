"""Label-geometry and slice-workload estimates, without model inference or IQ reads."""
from pathlib import Path
from functools import lru_cache
import argparse
import hashlib
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "analysis" / "outputs"
OUT = ROOT / "SAHI尺寸与成本评估.csv"
EPS = 1e-9
SCHEMES = {
    "short125": [.125], "single250": [.25], "single500": [.5],
    "single667": [2 / 3], "single700": [.7], "single1000": [1.0],
    "dual125_700": [.125, .7],
}
STYLES = [(640, 640), (1024, 640), (1024, 1024)]


def load():
    hashes = {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
              for name in ["samples.csv", "instances.csv"]}
    s = pd.read_csv(SOURCE / "samples.csv")
    i = pd.read_csv(SOURCE / "instances.csv")
    assert s.sample_id.is_unique and not i.duplicated(["sample_id", "signal_id"]).any()
    s["T_ms"] = s.duration_ms.round(3)
    s["sample_index"] = np.arange(len(s))
    d = i.merge(s[["sample_id", "split", "bandwidth_mhz", "duration_ms", "T_ms",
                   "n_complex_samples", "sample_index"]], on=["sample_id", "split"], validate="many_to_one")
    assert len(d) == len(i)
    return s, d, hashes


def ge(x, q=4):
    return np.asarray(x) >= q * (1 - EPS)


@lru_cache(None)
def axis_starts(length, size):
    step = size - int(.2 * size)
    last = max(length - size, 0)
    starts = tuple(range(0, last, step)) + (last,)
    assert starts[0] == 0 and starts[-1] + min(length, size) == length
    assert len(starts) == max(1, 1 + math.ceil((length - size) / step))
    return starts


def containment(a, b, length, size):
    starts = np.asarray(axis_starts(int(length), size))
    idx = np.searchsorted(starts, a + EPS, side="right") - 1
    idx = np.clip(idx, 0, len(starts) - 1)
    return (a >= starts[idx] - EPS) & (b <= starts[idx] + min(int(length), size) + EPS)


def shape(fs_mhz, n, tau_ms):
    window = int(2 * math.ceil(fs_mhz * tau_ms * 1000 / 2 - EPS))
    hop = window // 2
    # Retain frame centers in [0,T); actual STFT boundary policy must match.
    frames = int(math.ceil(n / hop))
    return window, hop, frames


def analyze():
    s, d, hashes = load()
    rows = []
    for name, taus in SCHEMES.items():
        conservative = np.zeros(len(d), bool)
        for tau_ms in taus:
            fs = d.bandwidth_mhz.to_numpy() * 1e6
            w = 2 * np.ceil(fs * tau_ms / 1000 / 2 - EPS)
            tau = w / fs
            conservative |= ge(d.bw_mhz.to_numpy() * 1e6 * tau) & ge(d.dur_ms.to_numpy() / 1000 / tau)
        for sp in ["train", "test"]:
            g = d[d.split == sp]
            flags = conservative[g.index]
            cl = pd.DataFrame({"class": g["class"].to_numpy(), "ok": flags}).groupby("class").ok.mean()
            for cid, val in cl.items():
                rows.append({"record": "class_coverage", "scheme": name, "split": sp, "class_id": cid,
                             "instances": int((g["class"] == cid).sum()), "native_q4_pct": 100 * val})
        for tile, model in STYLES:
            calls = np.zeros(len(s))
            square_pixels = np.zeros(len(s))
            rect_pixels = np.zeros(len(s))
            native_cells = np.zeros(len(s))
            fft_proxy = np.zeros(len(s))
            display_q4 = np.zeros(len(d), bool)
            intact = np.zeros(len(d), bool)
            intact_display = np.zeros(len(d), bool)
            for tau_ms in taus:
                for (b_mhz, n), sg in s.groupby(["bandwidth_mhz", "n_complex_samples"]):
                    f, hop, k = shape(b_mhz, int(n), tau_ms)
                    ids = sg.sample_index.to_numpy()
                    nf, nt = len(axis_starts(f, tile)), len(axis_starts(k, tile))
                    num = nf * nt
                    cf, ct = min(f, tile), min(k, tile)
                    r = min(model / cf, model / ct)
                    rf = int(math.ceil(round(cf * r) / 32) * 32)
                    rt = int(math.ceil(round(ct * r) / 32) * 32)
                    calls[ids] += num
                    square_pixels[ids] += num * model * model
                    rect_pixels[ids] += num * rf * rt
                    native_cells[ids] += f * k
                    fft_proxy[ids] += f * k * math.log2(f)
                    gg = d[(d.bandwidth_mhz == b_mhz) & (d.n_complex_samples == n)]
                    gi = gg.index.to_numpy()
                    p_f = gg.bw_mhz.to_numpy() / b_mhz * f
                    p_t = gg.dur_ms.to_numpy() * b_mhz * 1000 / hop
                    good_display = ge(r * p_f) & ge(r * p_t)
                    full_f = containment(gg.u1.to_numpy() * f, gg.u2.to_numpy() * f, f, tile)
                    full_t = containment(gg.t1_ms.to_numpy() * b_mhz * 1000 / hop,
                                         gg.t2_ms.to_numpy() * b_mhz * 1000 / hop, k, tile)
                    whole = full_f & full_t
                    display_q4[gi] |= good_display
                    intact[gi] |= whole
                    intact_display[gi] |= whole & good_display
            for sp in ["train", "test"]:
                sm = s.split.to_numpy() == sp
                dm = d.split.to_numpy() == sp
                cl = pd.DataFrame({"class": d.loc[dm, "class"].to_numpy(), "ok": conservative[dm]}).groupby("class").ok.mean()
                rows.append({
                    "record": "overall", "scheme": name, "split": sp, "tile_size": tile, "model_size": model,
                    "samples": int(sm.sum()), "instances": int(dm.sum()),
                    "native_q4_pct": 100 * conservative[dm].mean(),
                    "native_macro_pct": 100 * cl.mean(), "native_worst_class_pct": 100 * cl.min(),
                    "display_q4_model_pct": 100 * display_q4[dm].mean(),
                    "intact_context_pct": 100 * intact[dm].mean(),
                    "intact_context_and_display_q4_pct": 100 * intact_display[dm].mean(),
                    "calls_mean": calls[sm].mean(), "calls_max": int(calls[sm].max()), "calls_total": int(calls[sm].sum()),
                    "square_input_MP_mean": square_pixels[sm].mean() / 1e6,
                    "dynamic_rect_input_MP_mean": rect_pixels[sm].mean() / 1e6,
                    "stft_cells_MP_mean": native_cells[sm].mean() / 1e6,
                    "stft_cells_MP_max": native_cells[sm].max() / 1e6,
                    "fft_NlogM_proxy_mean": fft_proxy[sm].mean(),
                })
            for (sp, b, t), sg in s.groupby(["split", "bandwidth_mhz", "T_ms"]):
                ids = sg.sample_index.to_numpy()
                dm = (d.split == sp) & (d.bandwidth_mhz == b) & (d.T_ms == t)
                f0, h0, k0 = shape(b, int(round(b * t * 1000)), taus[0])
                desc = "|".join(f"{shape(b, int(round(b*t*1000)), z)[0]}x{shape(b, int(round(b*t*1000)), z)[2]}" for z in taus)
                rows.append({
                    "record": "BT", "scheme": name, "split": sp, "B_MHz": b, "T_ms": t,
                    "tile_size": tile, "model_size": model, "samples": len(sg), "instances": int(dm.sum()),
                    "native_q4_pct": 100 * conservative[dm].mean(),
                    "display_q4_model_pct": 100 * display_q4[dm].mean(),
                    "intact_context_pct": 100 * intact[dm].mean(),
                    "calls_mean": calls[ids].mean(), "calls_max": int(calls[ids].max()),
                    "square_input_MP_mean": square_pixels[ids].mean() / 1e6,
                    "dynamic_rect_input_MP_mean": rect_pixels[ids].mean() / 1e6,
                    "stft_cells_MP_mean": native_cells[ids].mean() / 1e6,
                    "design_FxK": desc,
                })
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    assert all(hashlib.sha256((SOURCE / n).read_bytes()).hexdigest() == v for n, v in hashes.items())
    out = pd.DataFrame(rows)
    z = out[(out.record == "overall") & (out.split == "train") & (out.tile_size == 640)]
    print(z[["scheme", "native_q4_pct", "native_worst_class_pct", "calls_mean", "calls_max",
             "dynamic_rect_input_MP_mean", "square_input_MP_mean", "intact_context_pct"]].to_string(index=False))
    print("Rows:", len(out), "Source hashes:", hashes)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pd.read_csv(OUT)
    main = d[(d.record == "overall") & (d.split == "train") & (d.tile_size == 640)].copy()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 10, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    names = {"short125": "0.125 ms", "single250": "0.25 ms", "single500": "0.5 ms",
             "single667": "0.667 ms", "single700": "0.7 ms", "single1000": "1 ms",
             "dual125_700": "0.125 + 0.7 ms"}
    selected = ["short125", "single250", "single500", "single700", "single1000", "dual125_700"]
    g = main.set_index("scheme").loc[selected]
    x = np.arange(len(g)); ax = axes[0]
    ax.plot(x, g.native_q4_pct, "o-", color="#245B85", label="All instances")
    ax.plot(x, g.native_worst_class_pct, "s--", color="#AC7439", label="Worst class")
    ax.axhline(95, color="#85898E", ls=":", lw=1)
    ax.set(ylim=(60, 101), ylabel="Native geometric coverage (%)",
           xticks=x, xticklabels=[names[z] for z in selected])
    ax.tick_params(axis="x", rotation=30); ax.legend(loc="lower right", frameon=False)
    ax.text(-.12, 1.05, "a", transform=ax.transAxes, weight="bold", fontsize=13)
    ax = axes[1]
    ax.bar(x - .18, g.square_input_MP_mean, width=.36, color="#BCD4E6", label="Fixed square tensors")
    ax.bar(x + .18, g.dynamic_rect_input_MP_mean, width=.36, color="#416F92", label="Dynamic rectangle tensors")
    ax.set(ylabel="Input megapixels / source sample", xticks=x,
           xticklabels=[names[z] for z in selected])
    ax.tick_params(axis="x", rotation=30); ax.legend(frameon=False)
    ax.text(-.12, 1.05, "b", transform=ax.transAxes, weight="bold", fontsize=13)
    for ext in ["png", "svg"]:
        fig.savefig(ROOT / f"SAHI覆盖与成本.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved coverage/workload figure.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["analyze", "plot"], default="analyze")
    a = p.parse_args()
    analyze() if a.mode == "analyze" else plot()
