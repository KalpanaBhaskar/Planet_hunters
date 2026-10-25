"""
v3 cache builder.

This script assumes the v2 prepared cache already exists:
    v2/cache/X_train_v2.pkl, X_test_v2.pkl
    v2/cache/hp_train.npy, hp_test.npy
    v2/cache/rel_train.npy, rel_test.npy
    v2/cache/cen_train.npy, cen_test.npy
    v2/cache/sigma_train.npy, sigma_test.npy
    v2/cache/t_train.npy, t_test.npy
    v2/cache/q_train.npy, q_test.npy

It creates v3-specific features and tensors:
    v3/cache/fold_{split}.npy       (N, 4 periods, 4 channels, 32 phase bins)
    v3/cache/X_{split}_tab.pkl      tabular features for LightGBM/ExtraTrees

The four folded periods are:
    0: best BLS period
    1: second BLS period
    2: third BLS period
    3: dominant low-frequency FFT period (rotation / starspots)

The four folded channels are:
    high-pass flux, low-pass flux, centroid x, centroid y

The extra tabular families are:
    - folded-profile shape statistics
    - single-transit / box matched-filter statistics
    - starspot / rotational harmonic fits
    - physically motivated transit priors
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings("ignore")

ROOT = Path(r"C:\Users\Kalpana\Documents\GitHub\Planet_hunters")
V2 = ROOT / "v2" / "cache"
OUT = ROOT / "v3" / "cache"
OUT.mkdir(parents=True, exist_ok=True)

NBINS = 32
PERIOD_COLS = ("b0_period", "b1_period", "b2_period", "fft_low_p1")
PERIOD_NAMES = ("b0", "b1", "b2", "low")
CHANNEL_NAMES = ("hp", "rel", "cx", "cy")

# Box widths are in cadences. 30-minute cadence means:
#   2 = 1 hour, 4 = 2 hours, 8 = 4 hours, 16 = 8 hours, 32 = 16 hours.
MF_WIDTHS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


def as_float_array(values, fill=0.0):
    """Convert a Series/array to float and remove infinities."""
    a = np.asarray(values, dtype=np.float64)
    a = np.nan_to_num(a, nan=fill, posinf=fill, neginf=fill)
    return a


def safe_periods(x: pd.DataFrame) -> np.ndarray:
    """Return a finite, physically sensible period matrix for folding."""
    P = np.column_stack([as_float_array(x[c], 27.0) for c in PERIOD_COLS])
    P = np.where(P <= 0.0, 27.0, P)
    return np.clip(P, 0.40, 27.0)


def binned_profile(t, values, period, nbins=NBINS, scale=1.0):
    """Phase-fold values at one period and return a mean profile.

    The profile is divided by a per-star noise/scale estimate so that stars
    with different brightnesses are comparable.
    """
    phase = (t % period) / period
    bins = np.minimum((phase * nbins).astype(np.int32), nbins - 1)
    counts = np.bincount(bins, minlength=nbins).astype(np.float64)
    sums = np.bincount(bins, weights=np.asarray(values, dtype=np.float64), minlength=nbins)
    profile = sums / np.maximum(counts, 1.0)
    profile = profile / max(float(scale), 1e-6)
    return profile.astype(np.float32), counts


def profile_stats(profile, prefix):
    """Compact shape statistics for a folded profile."""
    p = np.asarray(profile, dtype=np.float64)
    nb = len(p)
    ptp = float(p.max() - p.min())
    half = nb // 2
    # Flatness near the minimum: fraction of bins close to the deepest bin.
    flat = float(np.mean(p <= p.min() + 0.20 * ptp + 1e-12))
    # A secondary eclipse, if present, should appear near phase 0.5.
    secondary = float(np.median(p[half - 1:half + 2]) - p.min())
    return {
        f"{prefix}_min": float(p.min()),
        f"{prefix}_max": float(p.max()),
        f"{prefix}_ptp": ptp,
        f"{prefix}_std": float(p.std()),
        f"{prefix}_argmin": int(np.argmin(p)),
        f"{prefix}_argmax": int(np.argmax(p)),
        f"{prefix}_flat": flat,
        f"{prefix}_secondary": secondary,
        f"{prefix}_h0": float(p[:half].mean()),
        f"{prefix}_h1": float(p[half:].mean()),
        f"{prefix}_odd": float(p[::2].mean() - p[1::2].mean()),
        f"{prefix}_sym": float(p[:half].mean() - p[half:].mean()),
    }


def fold_split(split: str):
    """Create folded tensors and folded-profile tabular features."""
    x = pd.read_pickle(V2 / f"X_{split}_v2.pkl").reset_index(drop=True)
    hp = np.load(V2 / f"hp_{split}.npy", mmap_mode="r")
    rel = np.load(V2 / f"rel_{split}.npy", mmap_mode="r")
    cen = np.load(V2 / f"cen_{split}.npy", mmap_mode="r")
    sigma = np.maximum(np.load(V2 / f"sigma_{split}.npy"), 1e-6)
    t = np.load(V2 / f"t_{split}.npy")

    n = len(x)
    P = safe_periods(x)
    fold = np.zeros((n, len(PERIOD_COLS), 4, NBINS), dtype=np.float32)
    records = []

    # Per-star centroid scale, in the same units as the centroid arrays.
    cscale = 1.4826 * np.median(np.abs(cen - np.median(cen, axis=2, keepdims=True)), axis=2)
    cscale = np.maximum(cscale, 1e-6)

    t0 = time.time()
    for i in range(n):
        rec = {}
        hp_i = np.asarray(hp[i], dtype=np.float64)
        rel_i = np.asarray(rel[i], dtype=np.float64)
        cx_i = np.asarray(cen[i, 0], dtype=np.float64)
        cy_i = np.asarray(cen[i, 1], dtype=np.float64)

        hp_i = hp_i - np.median(hp_i)
        rel_i = rel_i - np.median(rel_i)
        cx_i = cx_i - np.median(cx_i)
        cy_i = cy_i - np.median(cy_i)

        for k, p in enumerate(P[i]):
            hp_prof, _ = binned_profile(t, hp_i, p, NBINS, sigma[i])
            rel_prof, _ = binned_profile(t, rel_i, p, NBINS, sigma[i])
            cx_prof, _ = binned_profile(t, cx_i, p, NBINS, cscale[i, 0])
            cy_prof, _ = binned_profile(t, cy_i, p, NBINS, cscale[i, 1])

            # Align transit/eclipse profiles on the high-pass minimum, but
            # align the rotation/starspot profile on the low-pass minimum.
            align = int(np.argmin(hp_prof if k < 3 else rel_prof))
            profiles = [hp_prof, rel_prof, cx_prof, cy_prof]
            for ch, prof in enumerate(profiles):
                prof = np.roll(prof, -align)
                fold[i, k, ch] = prof

            rec[f"fold{k}_align"] = align
            rec[f"fold{k}_premin"] = float(hp_prof.min())
            for ch_name, prof in zip(CHANNEL_NAMES, profiles):
                rec.update(profile_stats(np.roll(prof, -align), f"fold{k}_{ch_name}"))

        # Simple aggregate statistics across the four folded periods.
        hp_depths = [-rec[f"fold{k}_hp_min"] for k in range(4)]
        rel_amps = [rec[f"fold{k}_rel_ptp"] for k in range(4)]
        rec["fold_hp_depth_mean"] = float(np.mean(hp_depths))
        rec["fold_hp_depth_max"] = float(np.max(hp_depths))
        rec["fold_hp_depth_std"] = float(np.std(hp_depths))
        rec["fold_rel_amp_mean"] = float(np.mean(rel_amps))
        rec["fold_rel_amp_max"] = float(np.max(rel_amps))
        rec["fold_cen_shift_mean"] = float(np.mean([rec[f"fold{k}_cx_ptp"] + rec[f"fold{k}_cy_ptp"] for k in range(4)]))

        records.append(rec)
        if i % 4000 == 0:
            print(f"{split} fold {i}/{n} sec={time.time() - t0:.1f}", flush=True)

    F = pd.DataFrame.from_records(records)
    for c in F.columns:
        F[c] = F[c].replace([np.inf, -np.inf], np.nan)
    # Keep the CNN tensor numerically bounded.  A single bad centroid bin can
    # otherwise dominate the batch norm statistics.
    fold = np.clip(fold, -20.0, 20.0)
    np.save(OUT / f"fold_{split}.npy", fold)
    print(split, "fold tensor", fold.shape, "stats", F.shape, "sec", round(time.time() - t0, 1), flush=True)
    return x, F, sigma


def matched_filter_one(hp, q, cen, sigma, cscale):
    """Single-event / box matched-filter features.

    Unlike the BLS search, this does not require a period. It scans all times
    with box widths from 1 hour to 32 hours. This is important for planets
    with periods longer than the 27-day sector, where only one transit occurs.
    """
    hp = np.asarray(hp, dtype=np.float32)
    n = len(hp)
    out = {}
    best_score = -1e30
    best = None

    for w in MF_WIDTHS:
        box = uniform_filter1d(hp, size=w, mode="nearest")
        base = uniform_filter1d(hp, size=max(4 * w, 9), mode="nearest")
        depth = base - box  # positive for a dip
        score = depth * np.sqrt(w) / sigma
        j = int(np.argmax(score))
        val = float(score[j])
        out[f"mf_score_{w}"] = val
        out[f"mf_depth_{w}"] = float(depth[j])
        # Flare branch: positive excess rather than a dip.
        flare = (box - base) * np.sqrt(w) / sigma
        out[f"mf_flare_{w}"] = float(np.max(flare))
        if val > best_score:
            best_score = val
            best = (w, j, float(depth[j]))

    if best is None:
        return out

    w, j, depth = best
    half = max(1, w // 2)
    lo = max(0, j - half)
    hi = min(n, j + half + 1)
    seg = hp[lo:hi]
    seg_min = float(seg.min())
    seg_max = float(seg.max())
    seg_ptp = seg_max - seg_min + 1e-12
    out["mf_best_score"] = best_score
    out["mf_best_depth"] = depth
    out["mf_best_width"] = int(w)
    out["mf_best_pos"] = float(j) / max(n - 1, 1)
    out["mf_best_flat"] = float(np.mean(seg <= seg_min + 0.20 * seg_ptp))
    if len(seg) > 2:
        left = float(seg[:len(seg) // 2].mean())
        right = float(seg[len(seg) // 2:].mean())
        out["mf_best_sym"] = abs(left - right) / (abs(depth) + 1e-12)
    else:
        out["mf_best_sym"] = 0.0

    mask = np.zeros(n, dtype=bool)
    mask[lo:hi] = True
    if mask.sum() > 1 and (~mask).sum() > 1:
        out["mf_best_cx"] = float((np.median(cen[0, ~mask]) - np.median(cen[0, mask])) / cscale[0])
        out["mf_best_cy"] = float((np.median(cen[1, ~mask]) - np.median(cen[1, mask])) / cscale[1])
        out["mf_best_q1"] = float(np.mean((q[mask] & 1) > 0))
        out["mf_best_q4"] = float(np.mean((q[mask] & 4) > 0))
    else:
        out["mf_best_cx"] = 0.0
        out["mf_best_cy"] = 0.0
        out["mf_best_q1"] = 0.0
        out["mf_best_q4"] = 0.0
    out["mf_best_cm"] = float(np.hypot(out["mf_best_cx"], out["mf_best_cy"]))
    return out


def harmonic_fit(t, y, period, n_harm=3):
    """Fit a sinusoid plus harmonics and return amplitudes and fit quality."""
    phase = 2.0 * np.pi * t / period
    cols = [np.ones_like(t)]
    for h in range(1, n_harm + 1):
        cols.append(np.cos(h * phase))
        cols.append(np.sin(h * phase))
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    amps = []
    phases = []
    for h in range(n_harm):
        a = coef[1 + 2 * h]
        b = coef[2 + 2 * h]
        amps.append(float(np.hypot(a, b)))
        phases.append(float(np.arctan2(b, a)))
    return r2, amps, phases, float(np.sqrt(np.mean((y - pred) ** 2)))


def acf_one(y, lags):
    """Normalised autocorrelation at physically meaningful lags.

    Long lags (24-384 cadences = 0.5-8 days) capture rotational modulation
    from starspots.  Short lags capture transit/eclipse width and flare decay.
    """
    y = np.asarray(y, dtype=np.float64)
    y = y - np.mean(y)
    denom = float(np.dot(y, y)) + 1e-12
    out = {}
    for lag in lags:
        lag = int(lag)
        if 0 < lag < len(y):
            out[f"acf_lag_{lag}"] = float(np.dot(y[:-lag], y[lag:]) / denom)
        else:
            out[f"acf_lag_{lag}"] = 0.0
    return out


def spot_one(rel, t, sigma, period):
    """Starspot / rotational modulation features from harmonic fits."""
    y = np.asarray(rel, dtype=np.float64)
    y = y - np.median(y)
    period = float(np.clip(period if np.isfinite(period) else 5.0, 0.5, 27.0))
    r2, amps, phases, resid = harmonic_fit(t, y, period)

    # A half-sector fit checks whether the rotational signal is coherent.
    mid = np.median(t)
    m0 = t < mid
    m1 = ~m0
    r20, amps0, phases0, _ = harmonic_fit(t[m0], y[m0], period)
    r21, amps1, phases1, _ = harmonic_fit(t[m1], y[m1], period)

    # A period/2 fit captures the two-spot harmonic.
    r2p2, amps_p2, _, _ = harmonic_fit(t, y, period / 2.0)

    amp1 = amps[0]
    out = {
        "spot_period": period,
        "spot_r2": float(r2),
        "spot_r2_half0": float(r20),
        "spot_r2_half1": float(r21),
        "spot_r2_half_diff": float(abs(r20 - r21)),
        "spot_r2_p2": float(r2p2),
        "spot_amp1": float(amp1),
        "spot_amp2": float(amps[1]),
        "spot_amp3": float(amps[2]),
        "spot_amp_over_sigma": float(amp1 / max(float(sigma), 1e-6)),
        "spot_h21": float(amps[1] / (amp1 + 1e-12)),
        "spot_h31": float(amps[2] / (amp1 + 1e-12)),
        "spot_harmonic": float((amps[1] ** 2 + amps[2] ** 2) / (amp1 ** 2 + 1e-12)),
        "spot_amp_ratio_halves": float(amps0[0] / (amps1[0] + 1e-12)),
        "spot_phase_diff": float(abs(phases0[0] - phases1[0])),
        "spot_resid_over_sigma": float(resid / max(float(sigma), 1e-6)),
        "spot_p2_amp_ratio": float(amps_p2[0] / (amp1 + 1e-12)),
    }
    return out


def add_extra_features(x, sigma):
    """Add matched-filter, spot, physical-prior and interaction features."""
    hp = np.load(V2 / f"hp_{'train' if len(x) == 18000 else 'test'}.npy", mmap_mode="r")
    rel = np.load(V2 / f"rel_{'train' if len(x) == 18000 else 'test'}.npy", mmap_mode="r")
    cen = np.load(V2 / f"cen_{'train' if len(x) == 18000 else 'test'}.npy", mmap_mode="r")
    t = np.load(V2 / f"t_{'train' if len(x) == 18000 else 'test'}.npy")
    q = np.load(V2 / f"q_{'train' if len(x) == 18000 else 'test'}.npy")

    cscale = 1.4826 * np.median(np.abs(cen - np.median(cen, axis=2, keepdims=True)), axis=2)
    cscale = np.maximum(cscale, 1e-6)

    mf_records = []
    spot_records = []
    acf_records = []
    P_low = as_float_array(x["fft_low_p1"], 5.0)
    P_low = np.clip(np.where(P_low <= 0.0, 5.0, P_low), 0.5, 27.0)

    t0 = time.time()
    for i in range(len(x)):
        mf_records.append(matched_filter_one(hp[i], q, cen[i], sigma[i], cscale[i]))
        spot_records.append(spot_one(rel[i], t, sigma[i], P_low[i]))
        a = acf_one(rel[i], (24, 48, 96, 192, 384))
        a.update(acf_one(hp[i], (2, 3, 5, 8, 12, 20, 40)))
        acf_records.append(a)
        if i % 4000 == 0:
            print("extra", i, "/", len(x), "sec", round(time.time() - t0, 1), flush=True)

    MF = pd.DataFrame.from_records(mf_records)
    SP = pd.DataFrame.from_records(spot_records)
    AC = pd.DataFrame.from_records(acf_records)
    # Long-lag ACF peak summarises rotational modulation in one feature.
    long_cols = [f"acf_lag_{lag}" for lag in (24, 48, 96, 192, 384)]
    AC["acf_long_peak"] = AC[long_cols].max(axis=1)
    AC["acf_long_peak_lag"] = AC[long_cols].idxmax(axis=1).str.extract(r"(\d+)")[0].astype(float)
    AC["acf_long_absmax"] = AC[long_cols].abs().max(axis=1)
    x = pd.concat([x.reset_index(drop=True), MF, SP, AC], axis=1)

    # Period-consistency and BLS-peak features.
    p = [as_float_array(x[c], 27.0) for c in ("b0_period", "b1_period", "b2_period")]
    s = [as_float_array(x[c], 0.0) for c in ("b0_score", "b1_score", "b2_score")]
    d = [as_float_array(x[c], 0.0) for c in ("b0_depth", "b1_depth", "b2_depth")]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        x[f"b_period_ratio_{a}{b}"] = p[a] / np.maximum(p[b], 1e-6)
        x[f"b_period_diff_{a}{b}"] = np.abs(np.log(np.maximum(p[a], 1e-6) / np.maximum(p[b], 1e-6)))
        x[f"b_score_ratio_{a}{b}"] = s[a] / np.maximum(s[b], 1e-6)
        x[f"b_depth_ratio_{a}{b}"] = d[a] / np.maximum(d[b], 1e-9)
    x["b_score_sum"] = s[0] + s[1] + s[2]
    x["b_depth_sum"] = d[0] + d[1] + d[2]
    x["b_ntrans_max"] = np.maximum.reduce([
        as_float_array(x["b0_ntrans"], 0.0),
        as_float_array(x["b1_ntrans"], 0.0),
        as_float_array(x["b2_ntrans"], 0.0),
    ])
    x["b_score_per_transit"] = s[0] / np.maximum(as_float_array(x["b0_ntrans"], 0.0), 1.0)

    # Physically motivated transit priors.
    radius = np.maximum(as_float_array(x["radius"], 1.0), 0.05)
    contam = np.clip(as_float_array(x["contamination"], 0.0), 0.0, 0.99)
    density = np.maximum(as_float_array(x["density"], 0.01), 1e-3)
    for k in range(3):
        depth = np.clip(d[k], 0.0, None)
        period = p[k]
        dur = as_float_array(x[f"b{k}_dur"], 0.1)
        # Rp/Rs = sqrt(depth); convert solar radii to Earth radii.
        x[f"b{k}_rp_earth"] = np.sqrt(depth) * radius * 109.2
        x[f"b{k}_rp_over_radius"] = np.sqrt(depth)
        x[f"b{k}_depth_contam"] = depth / np.maximum(1.0 - contam, 1e-3)
        expected_dur = (period ** (1.0 / 3.0)) * radius / np.maximum(density ** (1.0 / 3.0), 1e-6)
        x[f"b{k}_dur_expected"] = expected_dur
        x[f"b{k}_dur_ratio"] = dur / np.maximum(expected_dur, 1e-6)
        x[f"b{k}_score_over_radius"] = s[k] / (radius + 1e-6)
        x[f"b{k}_score_x_depth"] = s[k] * depth
        x[f"b{k}_depth_per_transit"] = depth / np.maximum(as_float_array(x[f"b{k}_ntrans"], 0.0), 1.0)
        x[f"b{k}_planet_prior"] = s[k] * np.exp(-x[f"b{k}_rp_earth"] / 12.0)

    # Explicit interactions that were repeatedly useful in v1/v2 analysis.
    x["log_sigma"] = np.log10(np.maximum(sigma, 1e-9))
    x["sigma"] = sigma
    x["transit_like"] = s[0] * d[0] / np.maximum(1.0 + contam, 1e-3)
    x["eb_like"] = s[0] * d[0] * np.log1p(np.maximum(as_float_array(x["b0_ntrans"], 0.0), 0.0))
    x["spot_like"] = as_float_array(x["sin_low_r2"], 0.0) * as_float_array(x["low_std"], 0.0) / np.maximum(sigma, 1e-6)
    x["quiet_like"] = as_float_array(x["high_acf1"], 0.0) * as_float_array(x["low_smooth"], 0.0)
    x["fold_cen_over_depth"] = x["fold_cen_shift_mean"] / np.maximum(np.abs(x["b0_depth"]), 1e-6)
    x["fold_cen_over_depth"] = x["fold_cen_over_depth"].clip(-50, 50)

    return x.replace([np.inf, -np.inf], np.nan)


if __name__ == "__main__":
    for split in ("train", "test"):
        x_base, fold_stats, sigma = fold_split(split)
        x_tab = pd.concat([x_base, fold_stats], axis=1)
        x_tab = add_extra_features(x_tab, sigma)
        x_tab.to_pickle(OUT / f"X_{split}_tab.pkl")
        print(split, "tabular", x_tab.shape, flush=True)

    print("v3 cache complete", flush=True)
