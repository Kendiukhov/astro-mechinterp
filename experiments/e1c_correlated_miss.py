"""
E1c — Does the tomographic bias depend on WHICH objects the pipeline misses?

E1b propagated the measured 3.68% detection-miss rate by drawing the missed objects uniformly
at random. That is the assumption a survey referee attacks first, and correctly: detection
misses are not a random subsample. Faint, small, low-surface-brightness and blended objects are
preferentially missed, and because apparent magnitude correlates with redshift, a
magnitude-dependent miss rate becomes a redshift-dependent one -- which is exactly the kind of
coherent effect a tomographic mean-redshift requirement is written to catch.

So this experiment replaces the uniform draw with one measured from the data, and asks how much
the answer moves.

Design
------
PART A measures the miss probability rather than assuming it. Over the same 5000-cutout scan
E1b used, we record for every object whether the pipeline placed a segment on its position, and
its r-band magnitude. Binning in magnitude gives an empirical p(miss | mag) with no functional
form imposed. We also report how the miss population differs from the detected population in
magnitude and in redshift, which is what makes the effect possible at all.

PART B holds everything else fixed against E1b -- same 960 galaxies, same transplant of a real
measured miss field, same tomographic bins on true redshift -- and varies only WHICH objects are
designated as missed:

    uniform     : n_miss objects drawn with equal probability   (E1b's assumption)
    correlated  : n_miss objects drawn with probability proportional to the measured
                  p(miss | mag), renormalised so the expected rate is still 3.68%

Both arms miss the SAME NUMBER of objects, so the comparison isolates the assignment rule from
the count. Both use the faithful transplant representation, so it also isolates it from the
representation question E1b settled.

Two further weaknesses of E1b are fixed here. First, E1b reported a single realisation of the
miss assignment; this runs an ensemble of draws and reports the spread, so the reader can see
whether a difference between arms exceeds the noise in either. Second, the ensemble makes the
uniform arm a distribution rather than a point, which is the honest object to compare against.

Run:  python experiments/e1c_correlated_miss.py [N_SCAN] [N_IMPACT] [N_DRAWS]
Out:  results/e1c_correlated_miss.json
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import astron_mi as A
from astron_mi import physics as P

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
CKPT = RESULTS / "e1c_correlated_miss_partial.json"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
CROP = 96
BATCH = 16
N_SCAN = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
N_IMPACT = int(sys.argv[2]) if len(sys.argv) > 2 else 960
N_DRAWS = int(sys.argv[3]) if len(sys.argv) > 3 else 12
N_BOOT = 500
LSST_REQ = 0.003
TOMO_EDGES = [0.0, 0.15, 0.25, 0.35, 1.0]
MAG_BINS = 8                      # deciles are too noisy at 184 misses; 8 bins keeps >15 each
ARMS = ("uniform", "correlated")


def centre_crop(x, n=CROP):
    h, w = x.shape[-2:]
    r, c = (h - n) // 2, (w - n) // 2
    return x[..., r:r + n, c:c + n]


def main():
    from aion.modalities import LegacySurveySegmentationMap, Z

    rng = np.random.default_rng(SEED)
    cm = A.load_codecs(DEVICE)
    model = A.load_model("base", device=DEVICE)
    T = P.tables()

    # ------------------------------------------------------------------ PART A
    print(f"PART A - measuring p(miss | mag) over {N_SCAN} cutouts", flush=True)
    miss_flag, mag_r, zspec, miss_bank = [], [], [], []
    with A.ProvabgsReader() as r:
        rows = np.sort(rng.choice(r.n_rows(), min(N_SCAN, r.n_rows()), replace=False))
        for s in range(0, len(rows), BATCH):
            b = r.aion_batch(rows[s:s + BATCH], with_segmap=True,
                             with_image=False, with_spectrum=False)
            seg = centre_crop(b["segmap"])
            binary = (b["segmap"] > 0).float()
            m = np.asarray(b["labels"].get("mag_r"), dtype="float64")
            zz = np.asarray(b["labels"].get("z"), dtype="float64")
            for i in range(len(seg)):
                is_miss = float(seg[i, CROP // 2, CROP // 2]) <= 0
                miss_flag.append(is_miss); mag_r.append(m[i]); zspec.append(zz[i])
                if is_miss:
                    miss_bank.append(binary[i].clone())
            if (s // BATCH) % 60 == 0:
                print(f"  {min(s + BATCH, len(rows))}/{len(rows)}", flush=True)

    miss_flag = np.array(miss_flag, bool)
    mag_r = np.array(mag_r); zspec = np.array(zspec)
    good = np.isfinite(mag_r)
    p_miss = miss_flag.mean()
    print(f"  miss rate {100 * p_miss:.2f}%  ({miss_flag.sum()}/{len(miss_flag)})", flush=True)

    # empirical p(miss | mag), no functional form imposed
    edges = np.quantile(mag_r[good], np.linspace(0, 1, MAG_BINS + 1))
    edges[0] -= 1e-6; edges[-1] += 1e-6
    idx = np.clip(np.digitize(mag_r, edges) - 1, 0, MAG_BINS - 1)
    n_bin = np.array([(good & (idx == k)).sum() for k in range(MAG_BINS)])
    k_bin = np.array([miss_flag[good & (idx == k)].sum() for k in range(MAG_BINS)])
    p_raw = np.divide(k_bin, np.maximum(n_bin, 1), dtype="float64")
    ALPHA = 25.0                       # pseudo-counts, shrinking sparse bins to the global rate
    p_bin = (k_bin + ALPHA * p_miss) / (n_bin + ALPHA)
    print("  p(miss | mag_r):", flush=True)
    for k in range(MAG_BINS):
        print(f"    mag {edges[k]:5.2f}-{edges[k+1]:5.2f}  n={n_bin[k]:4d}  "
              f"raw={100 * p_raw[k]:5.2f}%  smoothed={100 * p_bin[k]:5.2f}%", flush=True)

    from scipy import stats
    md, mm = mag_r[good & ~miss_flag], mag_r[good & miss_flag]
    u_stat, u_p = stats.mannwhitneyu(mm, md, alternative="greater")   # missed fainter?
    lo = np.log(np.clip(p_bin, 1e-6, 1 - 1e-6))
    ctr = 0.5 * (edges[:-1] + edges[1:])
    slope = float(np.polyfit(ctr, lo, 1)[0])                          # log-odds per magnitude
    zd, zm = zspec[~miss_flag], zspec[miss_flag]
    contrast = {
        "mag_r_detected_median": float(np.nanmedian(md)),
        "mag_r_missed_median": float(np.nanmedian(mm)),
        "z_detected_median": float(np.nanmedian(zd)),
        "z_missed_median": float(np.nanmedian(zm)),
        "p_miss_faintest_bin": float(p_bin[-1]),
        "p_miss_brightest_bin": float(p_bin[0]),
        "faint_over_bright_ratio": float(p_bin[-1] / p_bin[0]) if p_bin[0] > 0 else None,
        "mannwhitney_u_p_missed_fainter": float(u_p),
        "log_p_miss_slope_per_mag": slope,
    }
    print(f"  missed fainter than detected: Mann-Whitney p = {u_p:.2e}; "
          f"log p(miss) slope = {slope:+.2f} per mag", flush=True)
    print(f"  missed are {contrast['mag_r_missed_median'] - contrast['mag_r_detected_median']:+.2f} mag "
          f"fainter and sit at z {contrast['z_missed_median']:.3f} vs "
          f"{contrast['z_detected_median']:.3f}", flush=True)
    bank = torch.stack(miss_bank)
    print(f"  banked {len(bank)} real miss fields", flush=True)

    part_a = {"n_scanned": int(len(miss_flag)), "p_miss": float(p_miss),
              "mag_bin_edges": [float(x) for x in edges],
              "p_miss_by_mag_bin_smoothed": [float(x) for x in p_bin],
              "p_miss_by_mag_bin_raw": [float(x) for x in p_raw],
              "n_by_mag_bin": [int(x) for x in n_bin],
              "contrast": contrast, "n_misses_banked": int(len(bank))}

    # ------------------------------------------------------------------ PART B
    print(f"\nPART B - {N_IMPACT} galaxies, {N_DRAWS} draws x {len(ARMS)} arms", flush=True)

    def seg_tok(binary):
        t = cm.encode(LegacySurveySegmentationMap(field=binary.clone().to(DEVICE)))
        return {k: v.reshape(v.shape[0], -1) for k, v in t.items()}

    def zmed(tok):
        o = model.forward({k: v.to(DEVICE) for k, v in tok.items()}, target_modality=[Z])
        lg = o["tok_z"][:, 0, :].detach().float().cpu().numpy()
        p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        pm = p[:, :1024] / p[:, :1024].sum(1, keepdims=True)
        return T.index_to_value("tok_z")[np.argmax(np.cumsum(pm, 1) >= 0.5, 1)]

    # cache the sample once: tokens, true masks, labels, and the miss weight per object
    rng_b = np.random.default_rng(SEED)
    cache, zt_l, mag_l = [], [], []
    with A.ProvabgsReader() as r:
        rows = np.sort(rng_b.choice(r.n_rows(), N_IMPACT, replace=False))
        for s in range(0, N_IMPACT, BATCH):
            b = r.aion_batch(rows[s:s + BATCH], with_segmap=True)
            ti = A.encode_image(cm, b["image"]["flux"], b["image"]["bands"], device=DEVICE)
            ts = A.encode_spectrum(cm, **b["spectrum"], survey="desi", device=DEVICE)
            binary = (b["segmap"] > 0).float()
            cache.append({"ti": {k: v.cpu() for k, v in ti.items()},
                          "ts": {k: v.cpu() for k, v in ts.items()},
                          "binary": binary.cpu(),
                          "st": {k: v.cpu() for k, v in seg_tok(binary).items()}})
            zt_l.append(np.asarray(b["labels"]["z"], dtype="float64"))
            mag_l.append(np.asarray(b["labels"]["mag_r"], dtype="float64"))
            print(f"  cached {min(s + BATCH, N_IMPACT)}/{N_IMPACT}", flush=True)

    zt_all = np.concatenate(zt_l); mag_all = np.concatenate(mag_l)
    ok = np.isfinite(zt_all)
    n_miss = int(round(p_miss * N_IMPACT))
    # weight each object by the measured p(miss | its magnitude)
    w = np.array([p_bin[int(np.clip(np.digitize(m, edges) - 1, 0, MAG_BINS - 1))]
                  if np.isfinite(m) else p_miss for m in mag_all])
    w = w / w.sum()
    print(f"\n  {n_miss} of {N_IMPACT} objects missed per draw; weight ratio "
          f"max/min = {w.max() / w[w > 0].min():.1f}", flush=True)

    # true-mask baseline: identical across draws, so compute once
    z_true = {c: [] for c in ("img", "img_spec")}
    for e in cache:
        bases = {"img": dict(e["ti"]), "img_spec": {**e["ti"], **e["ts"]}}
        for c in ("img", "img_spec"):
            z_true[c].append(zmed({**bases[c], **e["st"]}))
    z_true = {c: np.concatenate(v) for c, v in z_true.items()}
    print("  true-mask baseline done", flush=True)

    def tomo(z0, z1):
        zt = zt_all[ok]; out = []
        for lo, hi in zip(TOMO_EDGES[:-1], TOMO_EDGES[1:]):
            m = (zt >= lo) & (zt < hi)
            if m.sum() < 10:
                continue
            mt, mp = float(z0[ok][m].mean()), float(z1[ok][m].mean())
            out.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                        "shift_over_1pz": abs(mp - mt) / (1 + 0.5 * (lo + hi))})
        return out

    prior = {a: [] for a in ARMS}
    out_path = RESULTS / "e1c_correlated_miss.json"
    if out_path.exists():
        try:
            old_j = json.loads(out_path.read_text())
            if old_j.get("n_impact") == N_IMPACT and old_j.get("n_scan") == N_SCAN:
                prior = {a: old_j["draws"].get(a, []) for a in ARMS}
                print(f"  extending existing ensemble: {len(prior['uniform'])} draws on disk",
                      flush=True)
        except (json.JSONDecodeError, KeyError):
            pass
    n_prior = len(prior["uniform"])
    draws = {a: list(prior[a]) for a in ARMS}
    t0 = time.time()
    for d in range(n_prior, n_prior + N_DRAWS):
        rng_d = np.random.default_rng(70000 + d)   # per-draw seed: independent and resumable
        for arm in ARMS:
            pick = (rng_d.choice(N_IMPACT, n_miss, replace=False) if arm == "uniform"
                    else rng_d.choice(N_IMPACT, n_miss, replace=False, p=w))
            sel = np.zeros(N_IMPACT, bool); sel[pick] = True
            zp = {c: [] for c in ("img", "img_spec")}
            off = 0
            for e in cache:
                nb = len(e["binary"])
                pert = e["binary"].clone()
                for i in range(nb):
                    if sel[off + i]:
                        pert[i] = bank[int(rng_d.integers(0, len(bank)))]
                stk = seg_tok(pert)
                bases = {"img": dict(e["ti"]), "img_spec": {**e["ti"], **e["ts"]}}
                for c in ("img", "img_spec"):
                    zp[c].append(zmed({**bases[c], **stk}))
                off += nb
            rec = {"draw": d, "arm": arm,
                   "missed_mag_r_median": float(np.nanmedian(mag_all[sel])),
                   "missed_z_median": float(np.nanmedian(zt_all[sel]))}
            for c in ("img", "img_spec"):
                bins = tomo(z_true[c], np.concatenate(zp[c]))
                rec[c] = {"bins": bins,
                          "worst": max(b["shift_over_1pz"] for b in bins),
                          "n_exceeding": sum(b["shift_over_1pz"] > LSST_REQ for b in bins)}
            draws[arm].append(rec)
            print(f"  draw {d+1} (+{N_DRAWS} this run) {arm:11s} worst(img) "
                  f"{rec['img']['worst']:.5f} = {rec['img']['worst']/LSST_REQ:5.2f}x  "
                  f"missed mag {rec['missed_mag_r_median']:.2f} z {rec['missed_z_median']:.3f}"
                  f"  ({(time.time()-t0)/60:.0f}m)", flush=True)
        CKPT.write_text(json.dumps({"part_a": part_a, "draws": draws}, indent=1))

    # ------------------------------------------------------------------ verdict
    summ = {}
    for arm in ARMS:
        w_img = np.array([d["img"]["worst"] for d in draws[arm]])
        w_spec = np.array([d["img_spec"]["worst"] for d in draws[arm]])
        summ[arm] = {
            "img_worst_median": float(np.median(w_img)),
            "img_worst_p16": float(np.percentile(w_img, 16)),
            "img_worst_p84": float(np.percentile(w_img, 84)),
            "img_worst_over_req_median": float(np.median(w_img) / LSST_REQ),
            "img_frac_draws_exceeding": float(np.mean([d["img"]["n_exceeding"] > 0
                                                       for d in draws[arm]])),
            "img_spec_worst_median": float(np.median(w_spec)),
            "missed_mag_r_median": float(np.median([d["missed_mag_r_median"]
                                                    for d in draws[arm]])),
            "missed_z_median": float(np.median([d["missed_z_median"] for d in draws[arm]])),
        }
    ratio = (summ["correlated"]["img_worst_median"] / summ["uniform"]["img_worst_median"]
             if summ["uniform"]["img_worst_median"] > 0 else None)
    out = {"n_scan": N_SCAN, "n_impact": N_IMPACT, "n_draws": len(draws["uniform"]), "n_miss_per_draw": n_miss,
           "lsst_req": LSST_REQ, "part_a": part_a, "draws": draws, "summary": summ,
           "correlated_over_uniform": ratio,
           "verdict": (f"drawing misses by measured p(miss|mag) instead of uniformly changes the "
                       f"worst-bin bias by {ratio:.2f}x "
                       f"({summ['uniform']['img_worst_over_req_median']:.2f}x -> "
                       f"{summ['correlated']['img_worst_over_req_median']:.2f}x the requirement)"
                       if ratio else "uniform arm gave zero bias")}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "e1c_correlated_miss.json").write_text(json.dumps(out, indent=1))
    print("\n" + out["verdict"], flush=True)
    print("wrote results/e1c_correlated_miss.json", flush=True)


if __name__ == "__main__":
    main()
