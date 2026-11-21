#!/usr/bin/env python3
"""Phase B' stage-1 acceptance: histogram quality battery.

A. Synthetic end-to-end: BPSK/QPSK/16QAM/64QAM through the FULL phase-B'
   frontend (embed -> extract -> sync -> strip -> histogram). The amp/phase
   marginal structure must match the official HF reference signatures
   (rows=amp pile for constant envelope; phase cols at the theory
   positions under the locked +pi-shift convention).
B. Real occupancy: per-family mean per-bin magnitude profile over sync-pass
   instances vs the legacy 52/104 non-null masks (E4 adoption gate).
C. Real quality set: per class, clean + overlap + a sync-fail case:
   sync metrics, histogram validity (shape/finiteness/max-norm), and
   per-class representative PNG images.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.baseline3b_ofdm_frontend import (
    gen_ofdm_baseband, embed_capture, cp_scan_full, strip_symbols,
    filter_symbol_rows, extract_band, legacy_masks,
)
from scripts.baseline3b_histogram import (
    block_histogram, histogram_for_training, consecutive_runs, BLOCK,
)
from scripts.baseline3_dsp import demod_to_center, band_limit_fft

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports/baseline3_wifi_expert"
CACHE = EXP / "cache/symbols"
PASS_METRIC, PASS_RATIO = 0.34, 15.0

RESULTS = {"A_synthetic": {}, "B_occupancy": {}, "C_quality": []}


def theory_phase_cols(order):
    """Expected phase-difference col positions (locked convention)."""
    if order == 4:      # BPSK: dphi in {0, +-pi}
        vals = [0.0, np.pi, -np.pi]
    elif order == 16:   # QPSK: {0, +-pi/2, pi}
        vals = [0.0, np.pi / 2, -np.pi / 2, np.pi, -np.pi]
    else:               # 16/64QAM: dense; just check col spread
        return None
    cols = sorted(set(int(np.clip(np.ceil((v + np.pi) / (2 * np.pi) * 50) - 1,
                                      0, 49)) for v in vals))
    return cols


def synth_pipeline(bw, order, rng, n_sym=800, B=40.0, off_frac=0.5):
    xb, meta = gen_ofdm_baseband(bw * 1e6, order, n_sym, rng)
    fs_cap = B * 1e6
    max_off = B / 2 - bw / 2
    off = max_off * off_frac
    x, em = embed_capture(xb, bw * 1e6, fs_cap, off * 1e6, rng,
                          start_samples=int(rng.integers(0, 4000)))
    off_abs = 2400.0 + B / 2 + off
    xd = demod_to_center(x, fs_cap, off_abs * 1e6, 2400e6)
    xd = band_limit_fft(xd, fs_cap, bw * 1e6 * 1.02, edge_frac=0.05)
    res = cp_scan_full(xd, fs_cap)
    b = res["best"]
    X = strip_symbols(xd, fs_cap, b["tu_us"], b["cp_us"], b["phi"],
                      cfo_cycles=-b["cfo_cycles"])
    occ, pilots, data = legacy_masks(X.shape[1])
    X, keep_idx, n_kept, _ = filter_symbol_rows(X, occ)
    Hs = []
    for a, e in consecutive_runs(keep_idx):
        run = X[a:e]
        for k in range(len(run) // BLOCK):
            H = block_histogram(run[k * BLOCK:(k + 1) * BLOCK], occ)
            if H is not None:
                Hs.append(H)
    return Hs, b


def main():
    rng = np.random.default_rng(20261121)

    # ---------- A: synthetic signatures ----------
    print("A: synthetic end-to-end histogram signatures")
    from scripts.baseline3_dsp import qam_symbols

    def gen_bpsk(bw_hz, n_sym, rng):
        """BPSK (+-1) on all occupied bins, same layout as the generator."""
        xb_qam, meta = gen_ofdm_baseband(bw_hz, 4, n_sym, rng)
        n_fft, cp = meta["n_fft"], meta["cp"]
        occ = meta["occ"]
        grid = np.zeros((n_fft, n_sym), dtype=np.complex128)
        grid[occ] = rng.choice([-1.0 + 0j, 1.0 + 0j], size=(len(occ), n_sym))
        syms = np.fft.ifft(grid, axis=0) * np.sqrt(n_fft)
        xb = np.concatenate([syms[-cp:], syms], axis=0).T.reshape(-1)
        return xb.astype(np.complex64), meta

    def run_frontend(xb, bw_hz, B_mhz, rng, n_fft_sig):
        fs_cap = B_mhz * 1e6
        max_off = B_mhz / 2 - bw_hz / 1e6 / 2
        off = max_off * 0.5
        x, em = embed_capture(xb, bw_hz, fs_cap, off * 1e6, rng,
                              start_samples=int(rng.integers(0, 4000)))
        off_abs = 2400.0 + B_mhz / 2 + off
        xd = demod_to_center(x, fs_cap, off_abs * 1e6, 2400e6)
        xd = band_limit_fft(xd, fs_cap, bw_hz * 1.02, edge_frac=0.05)
        res = cp_scan_full(xd, fs_cap)
        b = res["best"]
        X = strip_symbols(xd, fs_cap, b["tu_us"], b["cp_us"], b["phi"],
                          cfo_cycles=-b["cfo_cycles"])
        # mask by TARGET bandwidth (bin indices valid at any capture rate)
        occ, pilots, data = legacy_masks(n_fft_sig)
        X, keep_idx, n_kept, _ = filter_symbol_rows(X, data)
        Hs = []
        for a, e in consecutive_runs(keep_idx):
            run = X[a:e]
            for k in range(len(run) // BLOCK):
                H = block_histogram(run[k * BLOCK:(k + 1) * BLOCK], data)
                if H is not None:
                    Hs.append(H)
        return Hs, b

    for name, order in [("BPSK", None), ("QPSK", 4), ("16QAM", 16), ("64QAM", 64)]:
        if name == "BPSK":
            xb, meta = gen_bpsk(20e6, 800, rng)
        else:
            xb, meta = gen_ofdm_baseband(20e6, order, 800, rng)
        Hs, b = run_frontend(xb, 20e6, 40.0, rng, meta["n_fft"])
        H = Hs[0]
        rm = H.sum(axis=1); cm = H.sum(axis=0)
        top_rows = [int(i) for i in np.argsort(rm)[::-1][:5]]
        top_cols = [int(i) for i in np.argsort(cm)[::-1][:8]]
        row_top_mass = float(rm[max(top_rows)] / rm.sum())
        RESULTS["A_synthetic"][name] = {
            "tu_cp": [b["tu_us"], b["cp_us"]], "metric": round(b["metric"], 3),
            "top_rows": sorted(top_rows), "top_cols": sorted(top_cols),
            "row_top_mass": round(row_top_mass, 3),
            "expected_cols": theory_phase_cols(order) if order in (2, 4, None) else None,
        }
        print(f"  {name}: rows={sorted(top_rows)} cols={sorted(top_cols)} "
              f"rowtop={row_top_mass:.2f} metric={b['metric']:.2f}")

    # ---------- B: real occupancy profiles ----------
    print("B: real per-bin occupancy vs legacy masks")
    man = json.loads((EXP / "dataset_manifest.json").read_text())
    cache_sum = json.loads((EXP / "symbol_cache_summary.json").read_text())
    ok_ids = {r["instance_id"] for r in cache_sum["results"]
              if r.get("status") == "ok"}
    byfam = {"W20": [], "W40": []}
    for inst in man["instances"]:
        if inst["instance_id"] in ok_ids:
            byfam[inst["family"]].append(inst["instance_id"])
    for fam, ids in byfam.items():
        n_fft_sig = 64 if fam == "W20" else 128
        occ, pilots, data = legacy_masks(n_fft_sig)
        occ_set = set(int(b) for b in occ)
        # common bin-offset axis: bin offset b in [-n_fft_i/2, n_fft_i/2)
        PAD = 160
        prof = np.zeros(2 * PAD + 1)
        cnt = 0
        for iid in ids[:200]:
            with np.load(CACHE / f"{iid:05d}.npz") as z:
                X = z["X"]
            n_fft_i = X.shape[1]
            p = np.mean(np.abs(X) ** 2, axis=0)
            j = np.arange(n_fft_i)
            offs = np.where(j < n_fft_i // 2, j, j - n_fft_i)  # FFT wrap
            for off_b, val in zip(offs, p):
                prof[int(off_b) + PAD] += val
            cnt += 1
        prof /= max(cnt, 1)
        pm = prof / prof.max()
        in_v = [pm[b + PAD] for b in occ_set if abs(b) <= PAD]
        out_v = [pm[b + PAD] for b in range(-PAD, PAD + 1)
                 if b not in occ_set]
        inmask = float(np.mean(in_v))
        outmask = float(np.mean(out_v))
        RESULTS["B_occupancy"][fam] = {
            "n_frames": cnt, "n_fft_sig": n_fft_sig,
            "n_occupied": len(occ_set),
            "mean_in_mask_rel": round(inmask, 4),
            "mean_out_mask_rel": round(outmask, 4),
            "ratio_db": round(10 * np.log10(max(inmask, 1e-9) / max(outmask, 1e-9)), 1),
        }
        print(f"  {fam}: in-mask {inmask:.3f} out-mask {outmask:.4f} "
              f"ratio {RESULTS['B_occupancy'][fam]['ratio_db']} dB (n={cnt})")

    # ---------- C: real quality set + images ----------
    print("C: real quality set")
    img_dir = EXP / "histogram_images"
    img_dir.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = []
    for fam, classes in (("W20", [0, 1, 2]), ("W40", [3, 4, 5])):
        for c in classes:
            clean = [i for i in man["instances"]
                     if i["class"] == c and i["instance_id"] in ok_ids
                     and i["overlap_any"] == 0]
            ov = [i for i in man["instances"]
                  if i["class"] == c and i["instance_id"] in ok_ids
                  and i["overlap_any"] == 1]
            sel.append(clean[0]); sel.append(clean[1]); sel.append(ov[0])
    fails = [i for i in man["instances"]
             if i["instance_id"] not in ok_ids][:2]
    for inst in sel:
        iid = inst["instance_id"]
        with np.load(CACHE / f"{iid:05d}.npz") as z:
            X = z["X"]; data = z["data_bins"]
            meta = json.loads(str(z["meta"]))
        res = instance_hist = None
        from scripts.baseline3b_histogram import instance_histograms
        r = instance_histograms(iid)
        H = r["blocks"][0] if r and r["blocks"] else None
        valid = H is not None and np.all(np.isfinite(H)) and H.sum() > 0
        entry = {
            "instance_id": iid, "sample_id": inst["sample_id"],
            "class": inst["class"], "stratum": "overlap" if inst["overlap_any"] else "clean",
            "B_mhz": inst["B_mhz"],
            "sync": {"tu_us": meta["tu_us"], "cp_us": meta["cp_us"],
                     "metric": meta["metric"], "ratio": meta["ratio"],
                     "n_kept": meta["n_kept"]},
            "n_blocks": len(r["blocks"]) if r else 0,
            "histogram_valid": bool(valid),
        }
        if valid:
            Ht = histogram_for_training(H)
            fig, ax = plt.subplots(figsize=(2.6, 2.6))
            ax.imshow(Ht[0], origin="lower", aspect="auto", cmap="viridis")
            ax.set_title(f"c{inst['class']} {inst['sample_id'].split('/')[1]}"
                         f"{' ov' if inst['overlap_any'] else ''}", fontsize=6)
            ax.axis("off")
            fig.savefig(img_dir / f"class{inst['class']}_iid{iid}"
                        f"{'_ov' if inst['overlap_any'] else ''}.png",
                        dpi=110, bbox_inches="tight")
            plt.close(fig)
            rm = H.sum(axis=1); cm = H.sum(axis=0)
            entry["top_rows"] = sorted(int(i) for i in np.argsort(rm)[::-1][:5])
            entry["top_cols"] = sorted(int(i) for i in np.argsort(cm)[::-1][:8])
        RESULTS["C_quality"].append(entry)
        print(f"  c{inst['class']} {inst['sample_id']} "
              f"{'ov' if inst['overlap_any'] else 'cl'} B={inst['B_mhz']:.0f} "
              f"sync={meta['metric']:.2f}/{meta['ratio']:.0f} "
              f"blocks={entry['n_blocks']} valid={valid}")
    for inst in fails:
        iid = inst["instance_id"]
        cache_res = next(r for r in cache_sum["results"]
                         if r["instance_id"] == iid)
        RESULTS["C_quality"].append({
            "instance_id": iid, "sample_id": inst["sample_id"],
            "class": inst["class"], "stratum": "sync_fail",
            "sync": {"status": cache_res.get("status"),
                     "metric": cache_res.get("metric"),
                     "ratio": cache_res.get("ratio")},
            "histogram_valid": False,
        })
        print(f"  c{inst['class']} {inst['sample_id']} SYNC-FAIL "
              f"{cache_res.get('status')} m={cache_res.get('metric')}")

    out = EXP / "histogram_battery.json"
    out.write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
