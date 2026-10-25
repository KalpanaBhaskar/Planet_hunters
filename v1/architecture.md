# Planet Hunters - Architecture and Trial 1 Plan

**Competition:** ML Challenge Hack4Hack 2026, Round 1 (24-hour ranking round)
**Window:** 22 Oct 2026, 10:00 IST -> 23 Oct 2026, 10:00 IST
**Task:** 8-class classification of TESS-like stellar light curves
**Metric:** Macro-averaged F1 over 8 classes
**Target machine:** Ryzen 5 5600H (6 cores / 12 threads), 8 GB RAM target (15.3 GB detected), RTX 3050 Laptop GPU (4 GB VRAM), Windows, Python 3.13.5

---

## 0. TL;DR - the best Trial 1 plan

1. **Do not start with deep learning.** With 18,000 labelled stars, 8 classes, and strong physical structure in the data, a well-engineered feature set + LightGBM is the fastest path to a high macro-F1 on this laptop.
2. Drop the **53 shared all-NaN cadences** (they are the same in train and test), leaving **1,262 valid cadences** per star.
3. Build **two views of each light curve**: a low-frequency view for spots/trends and a high-pass view for transits/pulsations. Remove the per-`camera`/`ccd` common-mode trend in each split separately.
4. Extract **~250-350 features** in 7 families: statistics, periodicity, transit/BLS, centroid, flares, spots/pulsations, instrument/quality flags.
5. Train a **5-fold stratified LightGBM** multiclass model with class weights, then **tune per-class decision thresholds/priors on out-of-fold predictions** to directly maximise macro-F1.
6. Add a **priority-aware one-vs-rest cascade** as a second model. The label rule is explicitly hierarchical, so this matches the data-generating process better than a flat classifier.
7. Use the **6 physical cores** for 6 feature-extraction processes. Keep each worker single-threaded (`OMP_NUM_THREADS=1`) to avoid thread thrash.
8. Use the **GPU only for an optional 1D CNN** late in the day. The CPU + LightGBM path is the primary model; the CNN is an ensemble-diversity member, not the backbone.
9. Submit early and often, but only 3 of the 5 daily uploads. **Select the final model by CV, not by public LB** (public is only 30% of 7,000 stars; private is 70%).
10. Expected realistic outcome: starter notebook is ~0.43 CV. A good feature + LightGBM pipeline should reach **~0.70-0.80 CV**, with the centroid/transit features deciding the hardest classes (`planet` vs `background_binary`, `eclipsing_binary` vs `instrument_glitch`).

---

## 1. Problem understanding

### 1.1 What a star can show

| Label | Physical cause | Discriminating signatures |
|---|---|---|
| `planet` | Planet transiting the star | Small, regular, flat-bottomed (U-shaped) dips; equal depth each transit; no secondary eclipse at phase 0.5; **centroid does not move during the dip**; not aligned with quality flags |
| `eclipsing_binary` | Two stars eclipsing each other | Deeper dips, often V-shaped; secondary eclipse at phase 0.5; odd/even differences possible; ellipsoidal modulation at P/2; centroid may move |
| `background_binary` | Faint eclipsing pair behind/next to the target, blended in the aperture | Shallow, planet-sized dips; V-shape; odd/even depth differences; **centroid moves during the dip**; high `contamination` helps |
| `starspots` | Dark spots on a rotating star | Smooth quasi-periodic dimming over days; evolves slowly; low high-frequency power; sinusoidal at rotation period |
| `pulsating_star` | Stellar pulsation (delta Scuti, gamma Dor, RR Lyrae) | Regular oscillations, often fast (hours); high-frequency periodogram peaks; can be aliased by 30-min sampling |
| `flare_star` | Magnetic flares, usually cool red dwarfs | Sudden bright spikes: fast rise, slower decay; multiple events; `teff` low, radius small; centroid usually stable |
| `instrument_glitch` | Telescope/detector misbehaviour | Sudden drops/recoveries, jumps, short sharp dips with big centroid jumps, ramps, stripes in one orbit; aligned with momentum dumps or scattered light |
| `quiet_star` | Nothing astrophysical | Noise and common camera trends only |

### 1.2 Label priority (critical)

If several phenomena are present, the label is the **highest-priority** one:

```text
planet > eclipsing_binary > background_binary > pulsating_star
       > flare_star > starspots > instrument_glitch > quiet_star
```

This is not a detail: it means a spotted star with a planet is `planet`, and a flaring spotted star is `flare_star`. The architecture should encode this rule explicitly in the decision layer, and can also model it with a cascade.

### 1.3 Evaluation

- Macro-F1 = mean of the 8 per-class F1 scores. Every class counts equally.
- Predicting `quiet_star` for everything scores near zero.
- **Accuracy and argmax are not the objective.** A model can have high accuracy and poor macro-F1 if it ignores rare/hard classes. Use class weights, per-class thresholds, and monitor per-class F1.
- Public LB = 30% of test (about 2,100 stars), private = 70% (about 4,900 stars). The final Round 1 ranking uses the private 70%, based on each team's best public upload. Public-LB noise is therefore large; **CV is the primary selection signal**.

### 1.4 Submission

- Exactly `id,label`, 7,000 rows, one row per test star, exact lowercase label spellings.
- `sample_submission.csv` is already in the same order as `test_stars.csv` (verified), so write predictions in that order.
- Up to 5 uploads per day, reset at midnight IST.
- Invalid files do not consume an upload, so validate before submitting.

---

## 2. Data audit (verified on this machine)

### 2.1 Files and shapes

| File | Shape | Dtype | Notes |
|---|---|---|---|
| `train_flux.npy` | 18,000 x 1,315 | float32 | Raw brightness, electrons/s; about 90 MB on disk |
| `train_centroid.npy` | 18,000 x 2 x 1,315 | float16 | Column/row offsets, millipixels |
| `test_flux.npy` | 7,000 x 1,315 | float32 | |
| `test_centroid.npy` | 7,000 x 2 x 1,315 | float16 | |
| `train_stars.csv` | 18,000 rows | - | id, tmag, teff, logg, radius, contamination, camera, ccd, label |
| `test_stars.csv` | 7,000 rows | - | same without label |
| `train_cadences.csv` | 1,315 rows | - | cadence, time_btjd, quality |
| `test_cadences.csv` | 1,315 rows | - | same |
| `sample_submission.csv` | 7,000 rows | - | all `quiet_star` |
| `submission.csv` | 7,000 rows | - | existing starter baseline output |

The full train set is only about 190 MB in memory, so RAM is not the bottleneck. CPU time for period searches is.

### 2.2 Findings that change the design

1. **The gap is global and identical in train and test.** Exactly 53 cadences (4.03%) are all-NaN for every star, in both flux and centroid, and they match `quality == 4`. Valid cadences = **1,262**. Drop the gap once and use a shared mask.
2. **One cadence every 30 minutes.** `diff(time_btjd) = 0.020833` days. Sector length 27.4 days.
3. **Train and test are adjacent in time**: train 2104.272-2131.647, test 2131.877-2159.252. They are still different sectors/pointings, so treat them as a domain shift, but the cadence layout is nearly identical.
4. **Quality flags are almost identical between splits**: train `{0:1221, 4:53, 2:33, 1:8}`, test `{0:1219, 4:53, 2:33, 1:10}`. In this dataset the flags are effectively mutually exclusive, but your code should still use bitwise tests (`quality & 1`, etc.) because the problem statement defines them as bits.
5. **Classes are reasonably balanced**: `quiet_star` 3,624; `starspots` 2,882; `eclipsing_binary` 2,247; `planet` 2,145; `pulsating_star` 1,973; `flare_star` 1,804; `instrument_glitch` 1,721; `background_binary` 1,604. Imbalance is mild (2.26x), but macro-F1 still rewards per-class care.
6. **Metadata distributions are matched** between train and test (`tmag`, `teff`, `logg`, `radius`, `contamination`), and classes are spread evenly across all 16 `camera`/`ccd` combinations. Metadata is a useful prior, not a sector fingerprint.
7. **No obvious leakage found**: no `id` overlap; no exact duplicate normalized light curves in the audited sample (first 3,000 rows per split); no exact metadata-key overlap. Do not rely on ID/order/time leaks; build real features.
8. **Basic global variability barely separates the classes.** Median MAD of relative flux (first 2,000 stars): `quiet_star` 0.000681, `planet` 0.000805, `background_binary` 0.000855, `flare_star` 0.001023, `starspots` 0.001059, `instrument_glitch` 0.001077, `pulsating_star` 0.001166, `eclipsing_binary` 0.001327. Global statistics alone will not solve `planet` vs `background_binary`. Shape, periodicity, and centroid features are mandatory.
9. **The common-mode trend is as large as the astrophysical signals.** Per-camera common-mode standard deviation is 0.00084-0.00125, comparable to the dips. Remove it per `camera`/`ccd` and per split.
10. **Centroid information is real and discriminating, but only if measured during the dip.** Median absolute flux-centroid correlation: `eclipsing_binary` 0.076, `background_binary` 0.059, versus `planet` 0.037 and `quiet_star` 0.033. Global centroid scatter is also elevated for `instrument_glitch` (5.03) and `eclipsing_binary` (5.15) versus `quiet_star` (4.43). A global correlation is weak; a **dip-localized centroid shift** is the decisive feature.
11. **Naive flare detection fails.** Counting raw >5 sigma positive outliers gives a median of 10 for `quiet_star` but only 6 for `flare_star`, because faint stars produce many noise outliers. Flares must be detected by **shape**: fast rise, multi-cadence, slower decay, and not a single-point cosmic ray.

### 2.3 Direct implications

- Remove 53 gap cadences and never let them influence statistics.
- Normalise every star by its own median; never feed raw electrons/s levels.
- Subtract the per-CCD common-mode computed **within each split** (using test stars for the test common-mode is allowed; no labels are used).
- Build separate low-pass and high-pass views. Transits need the high-pass view; spots need the low-pass view.
- The centroid must be analysed relative to detected dips, not just globally.
- Flare features must be shape-aware, not threshold-counting.
- Avoid features that encode absolute time, cadence index, row order, or sector identity.

---

## 3. Hardware reality and what it means

| Resource | Detected | Plan |
|---|---|---|
| CPU | AMD Ryzen 5 5600H, 6 cores / 12 threads | Use **6 worker processes** for feature extraction; benchmark LightGBM at 6 vs 12 threads |
| RAM | 15.3 GB installed, user target 8 GB | Keep peak under 4 GB; use float32, mmap, chunks, free arrays |
| GPU | RTX 3050 Laptop, 4 GB VRAM, CUDA UMD 13.4 | Only for a small 1D CNN, mixed precision, batch 32-64 |
| Disk | ~90 GB free | Cache features/probabilities; mmap the `.npy` files |
| OS | Windows | `spawn` multiprocessing; guard with `if __name__ == "__main__"` |
| Python | 3.13.5 in `myenv` | numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, sklearn 1.9.1; lightgbm/astropy/torch not installed yet |

**Key hardware conclusions**

- The CPU is the workhorse. The dataset is small enough that a 6-core CPU with a good feature set beats a rushed deep-learning model.
- Do not use all 12 logical threads for feature extraction. Six physical cores with one thread each is faster and uses less RAM than 12 oversubscribed threads.
- The 4 GB GPU is enough for a compact 1D CNN on 1,262-length curves, but **not** for large transformers or 2D image models. Keep the CNN small; if installation or VRAM becomes a problem, skip it without regret.
- The actual RAM is 15.3 GB, but design for 8 GB. This is easy: the data is ~190 MB and the feature matrix is only tens of MB.

---

## 4. Architecture at a glance

```text
                         +-----------------------------+
                         |  dataset_and_problem_...    |
                         |  flux / centroid / csv      |
                         +--------------+--------------+
                                        |
                          mmap read, float32, shared 53-cadence gap mask
                                        |
                                        v
                 +---------------------------------------------+
                 |  STAGE 1: PREPROCESSING                     |
                 |  - median normalise each star               |
                 |  - per-camera/ccd common-mode subtraction   |
                 |  - low-pass view (spots/trends)             |
                 |  - high-pass view (transits/pulsations)     |
                 |  - centroid median/common-mode removal      |
                 |  - per-star robust noise sigma              |
                 +----------------------+----------------------+
                                        |
                                        v
     +-----------------------------------------------------------------------+
     |  STAGE 2: PARALLEL FEATURE EXTRACTION (6 processes, 1 thread each)   |
     |  A. basic statistics       B. periodicity (LS/autocorrelation)       |
     |  C. transit/BLS + folded curves   D. centroid/dip-localised shifts   |
     |  E. flare shape            F. spots/pulsations                       |
     |  G. instrument/quality flags + metadata                              |
     +----------------------+------------------------------------------------+
                            |
                            v
                 +-----------------------------+
                 |  STAGE 3: FEATURE STORE     |
                 |  float32 .npy + names.json  |
                 |  versioned, cached, resumable |
                 +--------------+--------------+
                                |
              +-----------------+------------------+
              |                                    |
              v                                    v
   +----------------------+            +----------------------+
   | STAGE 4A: LightGBM   |            | STAGE 4B: optional   |
   | flat 8-class + OVR   |            | 1D CNN on GPU        |
   | 5-fold CV            |            | 5-fold CV            |
   +----------+-----------+            +----------+-----------+
              |                                    |
              +-----------------+------------------+
                                |
                                v
                 +-----------------------------+
                 |  STAGE 5: DECISION LAYER    |
                 |  - calibrate/blend probs    |
                 |  - tune class thresholds    |
                 |  - enforce priority rule    |
                 +--------------+--------------+
                                |
                                v
                 +-----------------------------+
                 |  STAGE 6: SUBMISSION        |
                 |  validate 7,000 ids/labels  |
                 |  write id,label in order    |
                 +-----------------------------+
```

### Build order (strictly value-first)

1. Preprocess + basic statistics + metadata -> LightGBM -> first valid submission.
2. Add periodicity + transit + centroid features -> retrain -> second submission.
3. Add flare + instrument + spot/pulsation features and threshold tuning -> third submission.
4. Add priority-aware OVR/cascade and ensemble -> fourth submission.
5. Optional CNN and stacking -> fifth submission, only if CV improves.

This ordering guarantees a valid, respectable submission even if later stages fail.

---

## 5. Preprocessing recipe

Do this once per split. Never mix train and test statistics.

### 5.1 Load efficiently

```python
import numpy as np, pandas as pd

D = "dataset_and_problem_statement"
flux = np.load(f"{D}/train_flux.npy", mmap_mode="r")       # (N, 1315) float32
cen  = np.load(f"{D}/train_centroid.npy", mmap_mode="r")   # (N, 2, 1315) float16
stars = pd.read_csv(f"{D}/train_stars.csv")
cad = pd.read_csv(f"{D}/train_cadences.csv")
```

Process in chunks of 1,000-2,000 stars. Never call `np.asarray()` on the whole mmap as float64.

### 5.2 Shared masks

```python
gap = np.isnan(np.asarray(flux[:64])).all(axis=0)   # 53 True entries, same in test
t = cad.time_btjd.values[~gap]
q = cad.quality.values[~gap]

dump = (q & 1).astype(bool)          # momentum dumps
scat = (q & 2).astype(bool)          # scattered light
good = ~(dump | scat)                # "clean" cadences for transit search
```

### 5.3 Robust normalisation and views

For each star `i`:

```python
f = flux[i, ~gap].astype(np.float32)
med = np.nanmedian(f)
rel = f / med - 1.0                       # relative flux, dimensionless

# per-star noise: robust sigma of the high-pass component
from scipy.ndimage import median_filter
hp = rel - median_filter(np.nan_to_num(rel, nan=0.0), size=49, mode="nearest")
sigma = 1.4826 * np.nanmedian(np.abs(hp - np.nanmedian(hp))) + 1e-12

# low-pass view: preserves spots/trends (no high-pass)
low = rel

# high-pass view: preserves transit/pulsation shapes
high = hp

# optional interpolated view for FFT/LS; only fill the 53 gap cadences
# (do not interpolate over real NaN inside a star; there are none beyond the shared gap)
```

Then subtract the common-mode per `camera`/`ccd` **within the split**:

```python
ccd_key = stars.camera.astype(str) + "_" + stars.ccd.astype(str)
for key, idx in ccd_key.groupby(ccd_key).groups.items():
    group = rel[idx]                       # (n_group, 1262)
    common = np.nanmedian(group, axis=0)   # robust per-cadence trend
    group = group - common
    rel[idx] = group
```

Do the same for `high` and for centroid columns. For centroid, first subtract each star's own median per axis, then subtract the per-CCD common-mode. Keep centroid in millipixels for physical interpretability; also divide by the star's centroid scatter for shift-significance features.

**Do not** remove a per-star running median from the low-pass view; that would erase starspots. Use the high-pass view for transits and pulsations.

### 5.4 Quality-aware handling

- Transit search: use `good` cadences (exclude momentum dumps and scattered-light cadences), but keep a separate feature for what happens at those cadences.
- Instrument features: deliberately compare dump/scattered cadences to their local neighbours.
- Never drop a cadence permanently from the data layout; store masks and use them per feature family.

---

## 6. Feature engineering - the core of the solution

Aim for **250-350 features**. Start with a high-value subset (~120), get a submission, then expand. Every feature must be computable identically for train and test.

### 6.1 Feature families

| Family | Count | What it captures | Most useful for |
|---|---|---|---|
| A. Global statistics | 30-40 | Spread, extremes, shape, autocorrelation, structure function | All classes; separates `quiet_star`, `eclipsing_binary`, `pulsating_star` |
| B. Periodicity (LS/ACF) | 40-60 | Periods, powers, harmonic structure, frequency bands | `starspots`, `pulsating_star`, `eclipsing_binary`, rotation |
| C. Transit/BLS + folded curve | 60-90 | Period, depth, duration, SNR, U/V shape, odd-even, secondary eclipse | `planet`, `eclipsing_binary`, `background_binary` |
| D. Centroid | 25-40 | Centroid scatter, flux-centroid correlation, **dip-localised shift** | `background_binary`, `instrument_glitch`, `eclipsing_binary` |
| E. Flare shape | 20-30 | Fast-rise/slow-decay events, rates, energies, cosmic-ray rejection | `flare_star` |
| F. Spots/pulsations | 20-30 | Slow vs fast power, smoothness, coherence, harmonics | `starspots`, `pulsating_star` |
| G. Instrument/quality | 30-40 | Dump/scatter alignment, jumps, ramps, per-orbit changes | `instrument_glitch` |
| H. Metadata/priors | 10-15 | `tmag`, `teff`, `logg`, `radius`, `contamination`, derived | All classes; breaks degeneracies |
| I. Interactions | 10-20 | Depth/SNR with `tmag`, centroid with `contamination`, period with radius | Hard look-alikes |

### 6.2 Family A - global statistics

On both `rel` and `high`:

- mean, std, MAD, IQR, range, min, max
- percentiles p1, p2, p5, p10, p25, p75, p90, p95, p98, p99
- skewness, kurtosis
- RMS, mean absolute deviation
- fraction of points below -1%, -2%, -5%, and above +1%, +2%, +5%
- von Neumann eta, lag-1 autocorrelation, lag-5/10/20 autocorrelation
- first-difference std and MAD
- structure function at lags 1, 2, 5, 10, 20, 50 cadences
- Stetson J and K (use flux and error proxies)
- number of zero crossings of the high-pass signal
- per-orbit std (split the 27.4-day sector into two ~13.7-day orbits) and orbit ratio/difference

### 6.3 Family B - periodicity

Use:
- **Lomb-Scargle** on `low` for rotation/spots (periods ~1-20 days).
- **Lomb-Scargle** on `high` for pulsations/aliases (periods ~0.02-1 day).
- **Autocorrelation** for a cheap, robust period candidate.

Concrete features:

- top 5 periodogram peaks: period, power, false-alarm probability, peak width
- number of peaks with FAP < 0.01 and < 0.001
- power ratio top1/top2 and top1/top3
- harmonic relations: P2/P1 close to 0.5 (secondary eclipse/ellipsoidal) or 2.0
- band powers: period < 0.1 d, 0.1-0.5 d, 0.5-1 d, 1-3 d, 3-7 d, 7-15 d, >15 d
- fraction of total power in each band
- periodogram entropy
- autocorrelation peak lag and height; first zero crossing
- sinusoidal fit at the best period: amplitude, phase, R^2, residual std
- number of significant frequencies after pre-whitening (fit top peak, subtract, repeat 3 times)

Implementation notes:

- Use `scipy.signal.lombscargle` for the full frequency grid (fast, C implementation), and `astropy.timeseries.LombScargle` only for the top peaks if you need false-alarm probabilities.
- Frequency grid: 0.02-24 cycles/day (period 0.0417-50 d), 5-10 samples per peak. About 5,000-10,000 frequencies is enough.
- Use weights `1/sigma^2` and subtract the mean. For the high-pass view, the mean is already near zero.

### 6.4 Family C - transit search and folded-curve vetting

This is the highest-value family for the `planet` / `background_binary` / `eclipsing_binary` triangle.

**Full BLS (best score)**:

```python
from astropy.timeseries import BoxLeastSquares

periods = np.exp(np.linspace(np.log(0.3), np.log(15.0), 3000))
durations = np.linspace(0.03, 0.30, 6)      # days, ~0.7-7.2 h
bls = BoxLeastSquares(t[good], high[good], dy=np.full(good.sum(), sigma))
res = bls.power(periods, durations)
best = np.argmax(res.power)
```

Features from the best BLS solution:

- best period, BLS power, depth, depth uncertainty, depth SNR, duration
- number of transits in the sector (compute from period and time baseline)
- transit duty cycle
- BLS power ratio best/second-best distinct period
- period consistency: refit the transit times, check residuals
- depth consistency across individual transits (std of per-transit depths)

**Folded-curve features** (fold `high` at the best period, 201 bins):

- depth at phase 0, depth at phase 0.5 (secondary eclipse)
- FWHM of the dip, total width, ingress/egress durations
- flat-bottom fraction: fraction of in-transit bins within 10% of maximum depth
- U-vs-V score: fit a trapezoid and a triangle; compare residuals, or use the second derivative near the dip bottom
- odd-even depth difference and its significance
- out-of-eclipse variability (std of bins outside the dip)
- ellipsoidal modulation amplitude at P/2
- number of distinct dips per period (one vs two)
- phase coverage and number of points in the dip

**Event-based features** (independent of BLS; important for single transits and irregular glitches):

- detect negative excursions below -3 sigma, -5 sigma, -8 sigma
- cluster consecutive cadences into events
- count, deepest depth, median depth, duration distribution
- rise time, fall time, symmetry, flat-bottom fraction per event
- spacing regularity: autocorrelation of event times, best period from event times
- number of events, fraction of events aligned with quality flags
- deepest single event SNR; is it a single cadence or multi-cadence?

**Speed trick**: if full BLS is too slow, run a two-stage search:

1. get the top 30-50 candidate periods from Lomb-Scargle/autocorrelation;
2. fold on each candidate and compute binned dip metrics (depth, SNR, U/V, odd-even, secondary).

This is much cheaper than a full BLS grid and captures most of the signal.

### 6.5 Family D - centroid (the background-binary killer)

Compute on the common-mode-removed centroid `(cx, cy)`:

Global:
- std, MAD, range of `cx` and `cy`
- correlation between `cx` and `cy`
- Pearson and Spearman correlation between flux and each centroid axis
- centroid jump statistics: max |delta|, number of >3 sigma jumps, jump alignment with flux jumps
- centroid difference at momentum-dump cadences vs neighbours
- pointing drift slope over the sector
- per-orbit centroid std and orbit difference

Dip-localised (most important):
- for every detected event (or the best BLS transit), compare the median centroid inside the event to the median outside
- `shift_x = (median_cx_in - median_cx_out) / centroid_sigma_x`
- `shift_y` analogously
- combined shift magnitude and direction angle
- centroid-flux covariance restricted to in-dip cadences
- shift significance relative to the depth: `|shift| / depth`
- same metrics for positive events (flares) to reject cosmic rays

Why it works: a true planet dims the target star only, so the centroid stays fixed. A blended background binary dims a neighbouring star, so the centroid moves toward the neighbour during the dip. An instrument glitch often produces a large, abrupt centroid jump.

### 6.6 Family E - flares

Do not count raw outliers. Use shape.

For each positive excursion above 3-5 sigma in `high`:

- require a fast rise (peak reached within 1-2 cadences)
- require a decay over at least 2-3 cadences
- fit an exponential decay to the post-peak points; keep the event if the fit is good and the decay time is positive
- reject single-cadence cosmic rays (no decay) and events with a simultaneous centroid jump

Features:
- number of flare candidates per day, per sector, per orbit
- maximum amplitude, median amplitude, total energy proxy (sum of positive excess)
- rise time, decay time, decay-time distribution, asymmetry ratio
- fraction of events that are single-cadence vs multi-cadence
- flare rate vs rotation period (if a period was found)
- flare amplitude vs `teff`/`radius` (cool, small stars flare more)
- number of events with a stable centroid (real flare) vs centroid jump (cosmic ray/glitch)

### 6.7 Family F - spots vs pulsations

These two are often confused. Separate them by frequency and smoothness.

- low-frequency power (periods 1-20 d) vs high-frequency power (periods < 0.5 d)
- ratio of low-band to high-band power
- smoothness: std of the second derivative, high-frequency residual after a low-pass filter
- sinusoidal fit quality at the rotation period; number of harmonics needed
- amplitude modulation over the sector (spots evolve)
- phase coherence: split the sector in half, compare the fitted period/phase
- number of significant high-frequency peaks and their harmonic structure
- zero-crossing rate; peaks per day
- alias pattern: compare power at f and 1/day - f (the 30-min cadence and the 1-day Earth alias)

### 6.8 Family G - instrument and quality flags

- fraction of cadences in each quality class
- flux offset at momentum dumps (`quality & 1`) relative to local neighbours
- flux offset at scattered-light cadences (`quality & 2`)
- correlation of flux with the per-CCD common-mode (after subtraction it should be low; residual correlation indicates a star-specific instrument response)
- number of sudden jumps: |first difference| > 5 sigma, > 10 sigma
- segment means before/after jumps; recovery time; ramp slopes
- number of single-cadence negative spikes with a simultaneous centroid jump
- fraction of detected dips aligned with a quality flag
- per-orbit statistics: std, min, max, number of outliers in orbit 1 vs orbit 2
- "stripe" features: difference in median flux between the two orbits, difference in noise, difference in trend slope
- number of cadences with quality != 0

### 6.9 Family H - metadata and derived priors

Raw: `tmag`, `teff`, `logg`, `radius`, `contamination`, `camera`, `ccd`.

Derived:
- `log_radius = log10(radius)`
- luminosity proxy: `radius^2 * (teff/5772)^4`
- density proxy: `logg - log10(radius)` (or `logg` plus `log_radius`)
- expected photometric noise proxy from `tmag`
- `is_m_dwarf = teff < 4000`
- `is_giant = logg < 3.5`
- `contamination * depth` and `contamination * centroid_shift` interactions
- expected transit depth for a Jupiter/Sun-like planet: `(R_planet/R_star)^2` scaled by radius
- expected transit duration proxy: `period^(1/3) * radius` (or use the detected period)

Use `camera` and `ccd` as categorical features in LightGBM, but verify with adversarial validation and a leave-one-CCD-out check. They may carry sector-specific quirks.

### 6.10 Family I - interactions

Trees can learn interactions, but explicit ratios help with small data:

- `depth / sigma`, `depth / tmag`, `depth * contamination`
- `centroid_shift / depth`, `centroid_shift * contamination`
- `period / radius`, `duration / radius`
- `low_band_power / high_band_power`
- `odd_even_difference / depth`
- `secondary_depth / primary_depth`
- `flare_rate / rotation_period`

### 6.11 Feature extraction strategy

Process **train and test with the same code path**. Do not compute a feature only for one split.

Recommended order:

1. Vectorised global statistics for all stars (fast).
2. Common-mode removal (vectorised per CCD).
3. Per-star loop for periodicity, BLS, centroid events, flares, instrument features.
4. Save feature matrix + names + version.

---

## 7. Modeling strategy for Trial 1

### 7.1 Phase A (must-have): flat LightGBM

Why LightGBM first:

- Handles NaN/inf natively.
- Fast on 18k x 300 on a 6-core CPU.
- Robust to feature scaling and outliers.
- Strong tabular baseline; no GPU needed.
- Easy to inspect feature importance and debug.

Configuration starting point:

```python
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

params = dict(
    objective="multiclass",
    num_class=8,
    metric="multi_logloss",
    learning_rate=0.03,
    num_leaves=63,
    max_depth=-1,
    min_data_in_leaf=40,
    feature_fraction=0.75,
    bagging_fraction=0.85,
    bagging_freq=1,
    lambda_l2=5.0,
    n_estimators=4000,
    n_jobs=6,
    verbosity=-1,
    random_state=42,
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

- Use early stopping with `callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)]` (or the estimator's `early_stopping_rounds` parameter, depending on your LightGBM version).
- Use `sample_weight=compute_sample_weight("balanced", y_train)` (mild balancing; also test without weights).
- Save `model.classes_` and use that exact order everywhere.
- Average test probabilities across the 5 folds.
- Repeat CV with 2-3 seeds and average, because labels have ~2% noise. This stabilises the OOF estimates.

Fallback if LightGBM cannot be installed: `sklearn.ensemble.HistGradientBoostingClassifier` (already installed) with `class_weight="balanced"`, `max_iter=1000`, `learning_rate=0.05`, `max_leaf_nodes=63`, `l2_regularization=1.0`, `early_stopping=True`. It is slower but works.

### 7.2 Phase B: priority-aware one-vs-rest / cascade

The label rule is hierarchical, so a flat classifier is not the only option. Build a second model:

**Option 1 - OVR detectors:**

- Train 7 binary LightGBM detectors: `planet` vs rest, `eclipsing_binary` vs rest, ..., `instrument_glitch` vs rest.
- At prediction, for each star, walk the priority list from highest to lowest. If a detector's calibrated probability exceeds its tuned threshold, assign that label. If none fire, assign `quiet_star`.
- Tune the 7 thresholds on OOF to maximise global macro-F1.

**Option 2 - cascade:**

- Stage 1: `planet` vs all. If positive, stop.
- Stage 2: among the rest, `eclipsing_binary` vs all. If positive, stop.
- Continue through the priority list.
- Each stage is trained only on stars that reached it, so lower-priority classes are learned among the stars that are actually eligible for them.
- Tune stage thresholds on OOF, again for global macro-F1.

**Which to use?** Build both if time allows; compare OOF macro-F1 and confusion matrices. The flat multiclass model and the priority-aware model often make different mistakes; blending their probabilities plus priority decoding is a strong final layer.

### 7.3 Phase C (optional): 1D CNN on GPU

Only start this after Phase A/B have a valid submission and CV is recorded. The CNN is for diversity, not as a replacement.

Input channels (all length 1,262 or downsampled to 631):

1. high-pass flux
2. low-pass relative flux (or raw relative flux)
3. centroid column (common-mode removed, robustly scaled)
4. centroid row
5. quality/gap mask (optional)

Architecture (small, 4 GB-safe):

- Conv1d(4, 32, kernel 7) -> BatchNorm -> ReLU -> MaxPool(2)
- 3 residual blocks: 32 -> 64 -> 128 channels, kernels 5, dilations 1/2/4
- global average + global max pooling
- dropout 0.3 -> Linear(256) -> 8 logits
- about 0.5-1.5 M parameters

Training:

- 5-fold stratified, same folds as LightGBM.
- AdamW, lr 1e-3, weight decay 1e-4, cosine schedule.
- batch 64, mixed precision (`torch.cuda.amp.autocast` + `GradScaler`).
- label smoothing 0.05 to handle the ~2% label noise.
- early stopping on fold macro-F1.
- augmentation: random time shift, Gaussian noise scaled to per-star sigma, small amplitude jitter, random cadence masking.
- keep data on CPU and transfer batches; do not load all channels onto the 4 GB GPU at once.

If installation or VRAM is a problem, skip the CNN. The LightGBM ensemble is the safe path.

### 7.4 Handling the ~2% label noise

- Use repeated stratified CV and average predictions; do not trust a single split.
- Use class weights, early stopping, and regularisation to avoid memorising noisy labels.
- Later, use OOF confident learning: if a star is predicted as a look-alike class with very high confidence and the pair is a known confusion, downweight it or relabel it in a second-stage model. Do this only after the baseline is stable; it can help or hurt.
- For the CNN, label smoothing is the simplest robust choice.

### 7.5 Validation

- Primary: 5-fold stratified CV, repeated with 2-3 seeds.
- Report macro-F1 and **per-class F1**, plus the full confusion matrix.
- Focus on the known confusions:
  - `planet` vs `background_binary` (centroid shift, odd-even, V-shape)
  - `planet` vs `eclipsing_binary` (depth, secondary eclipse, V-shape)
  - `planet` vs `instrument_glitch` (quality flags, single event vs periodic)
  - `starspots` vs `pulsating_star` (frequency band, smoothness)
  - `flare_star` vs `quiet_star` / cosmic rays (decay shape, centroid)
- Robustness check: leave-one-CCD-out CV (train on 12 CCDs, validate on 4). If macro-F1 collapses, reduce CCD-specific features.
- Adversarial validation: train a classifier to distinguish train vs test using the features. If AUC is high, inspect feature importance and remove/normalise the drifting features. With matched metadata this should be modest, but sector-specific trend features can leak.
- Do not use a random row split as the only evidence; the test sector is different.

### 7.6 Threshold tuning for macro-F1

Argmax of multiclass probabilities maximises accuracy, not macro-F1. Add a decision layer:

```python
# OOF probabilities P: (n, 8), LABELS fixed order
# Find class multipliers w >= 0 such that
# pred_i = argmax_c (w_c * P[i, c])
# maximises macro-F1 on OOF.
```

- Optimise `w` with coordinate ascent or Nelder-Mead on OOF macro-F1.
- Constrain `w` to a sensible range (e.g., 0.5-2.0) and smooth the solution to avoid overfitting.
- Validate the tuning with nested CV: tune on 3 folds, evaluate on the other 2.
- With fairly balanced classes the gain may be a few points, but it is one of the cheapest gains available.
- For the priority cascade, tune the 7 stage thresholds the same way.

### 7.7 Ensembling

Start simple, then add diversity:

1. **LightGBM seeds**: 3-5 seeds, average probabilities. Cheap and reliable.
2. **LightGBM + XGBoost + HistGradientBoosting**: different inductive biases. Use sample weights/class weights consistently.
3. **ExtraTrees/RandomForest** (`class_weight="balanced_subsample"`, 500 trees, `max_features="sqrt"`): fast diversity; often helps the hard classes.
4. **CNN**: add only if its OOF macro-F1 is within ~0.02-0.03 of LightGBM.
5. **Blend**: weighted average of probabilities, weights proportional to OOF macro-F1 or optimised on OOF with a holdout. Avoid a complex stacker on 18k rows; a regularised logistic regression on OOF probabilities is the most complex option worth trying.

Calibrate before blending if the models have very different probability scales (temperature scaling for the CNN; isotonic per class for trees, optional).

### 7.8 Optional: pseudo-labeling / transductive learning

- You may use the unlabelled test stars for **feature normalisation and common-mode estimation**; that is transductive but label-free and allowed.
- Pseudo-labeling is optional: train the Phase A/B model, take test stars with max probability > 0.95 and a margin > 0.1, add them with a sample weight of 0.3-0.5, retrain. Validate the idea by simulating on train (train on 80%, pseudo-label the held-out 20%) before trusting it. It can adapt to the test sector, but confirmation bias is real.
- Never use outside labelled data or hand-label the test set.

---

## 8. Parallelisation and memory plan

The goal is maximum throughput without oversubscription. On a 6-core/12-thread laptop, more processes is not always faster.

### 8.1 Process model

| Task | Parallelism | Why |
|---|---|---|
| Data loading | mmap, chunked | Avoids loading 190 MB copies repeatedly |
| Global/vectorised features | NumPy on the main process, 1-6 threads | Fast and memory-light |
| Per-star features (LS/BLS/flares) | **6 worker processes, 1 thread each** | CPU-bound; physical cores |
| LightGBM CV | 1 model at a time, `n_jobs=6` (benchmark 12) | LightGBM already threads; nested parallelism hurts |
| Hyperparameter search | 6 trials in parallel, each `n_jobs=1` | Better than one trial with 6 threads for small data |
| CNN | GPU, 2 dataloader workers max | CPU must stay free for features/GBM |
| Feature extraction + CNN | Overlap only after features are cached | Avoid CPU contention |

### 8.2 Thread environment

Set these **before importing numpy** in every worker:

```python
import os
for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[v] = "1"
```

For the main LightGBM process, set `OMP_NUM_THREADS=6` (or 12) and benchmark. For BLS/LS workers, keep 1.

### 8.3 Chunked, resumable extraction

```python
# extract_features.py
import os
for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[v] = "1"

import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

def process_chunk(split, lo, hi):
    flux = np.load(f"dataset_and_problem_statement/{split}_flux.npy", mmap_mode="r")
    cen  = np.load(f"dataset_and_problem_statement/{split}_centroid.npy", mmap_mode="r")
    stars = pd.read_csv(f"dataset_and_problem_statement/{split}_stars.csv")
    cad = pd.read_csv(f"dataset_and_problem_statement/{split}_cadences.csv")

    feats = []
    for i in range(lo, hi):
        feats.append(compute_features(flux[i], cen[i], stars.iloc[i], cad))
    X = np.asarray(feats, dtype=np.float32)
    np.save(f"artifacts/features/{split}_{lo:05d}_{hi:05d}.npy", X)
    return hi - lo

if __name__ == "__main__":
    chunks = [(split, lo, min(lo + 250, n)) for split in ("train", "test")
              for lo in range(0, n, 250)]
    with ProcessPoolExecutor(max_workers=6) as ex:
        for done in ex.map(lambda a: process_chunk(*a), chunks):
            print("chunk done", done)
```

Rules:

- Pass **file paths and index ranges**, not arrays, to workers.
- Each worker opens its own mmap.
- Write one `.npy` per chunk; concatenate in the main process.
- If a chunk file exists, skip it (resumable).
- Use `if __name__ == "__main__"` on Windows.
- Avoid nested pools; use one pool of 6.

### 8.4 Memory budget (target under 4 GB)

| Item | Approx. size |
|---|---|
| One flux mmap page cache | OS-managed |
| One centroid mmap | OS-managed |
| 6 feature workers | < 1 GB total |
| Feature matrix 25k x 300 float32 | ~30 MB |
| LightGBM dataset | tens of MB |
| CNN batch (64 x 5 x 1,262 float32) | ~1.6 MB input; activations < 1 GB |
| OOF/test probability arrays | < 5 MB |

- Use float32 for features and probabilities.
- Do not keep float64 copies of full curves.
- `del` large arrays and call `gc.collect()` between stages.
- For the CNN, keep curves on CPU; transfer batches. If VRAM OOM: batch 32, input 631, channels 32/64/128.

### 8.5 What to parallelise, in priority order

1. Feature extraction across stars (biggest win).
2. Train and test extraction overlap? No: run them together in one pool, since per-star work is independent.
3. Hyperparameter trials across configs (6 x 1 thread).
4. Model diversity: train LightGBM/XGBoost/ExtraTrees in parallel when benchmarking, but do not oversubscribe while LightGBM is training.
5. GPU CNN while CPU tunes thresholds or extracts a new feature family (only if cores are free; otherwise serialise).
6. I/O: cache features once; never recompute after a model-only change.

### 8.6 Expected runtimes on this laptop

| Stage | Estimate |
|---|---|
| Load/audit + preprocess | 5-15 min |
| Global + metadata features | 1-3 min |
| Full Lomb-Scargle (all stars, 6 workers) | 10-20 min |
| Full BLS (all stars, 6 workers, 3,000 periods) | 20-45 min |
| Two-stage period search (instead of full BLS) | 5-10 min |
| Centroid/flare/instrument features | 5-10 min |
| LightGBM 5-fold | 3-10 min |
| 20-config hyperparameter search | 20-60 min |
| CNN 5-fold (if used) | 20-60 min |
| Ensemble/threshold tuning | 10-20 min |

These are estimates; benchmark on 100 stars first and multiply.

---

## 9. Execution plan

### 9.1 If the full 24 hours are available

| Time | Work | Deliverable |
|---|---|---|
| 0:00-0:30 | Environment, data audit, reproduce starter | Baseline CV ~0.43 |
| 0:30-1:30 | Plot 3-5 stars per class; inspect dips, centroids, flares | Intuition + feature shortlist |
| 1:30-3:30 | Preprocess + ~120 high-value features + LightGBM 5-fold | **Submission 1** |
| 3:30-6:00 | Add LS/periodicity + BLS/folded + centroid features | **Submission 2** |
| 6:00-9:00 | Add flare/instrument/spot features; feature selection; tune LightGBM | **Submission 3** |
| 9:00-12:00 | Priority OVR/cascade + threshold tuning + ensemble | **Submission 4** |
| 12:00-15:00 | Optional CNN (GPU) + stacking; error analysis on top confusions | Candidate for **Submission 5** |
| 15:00-18:00 | Targeted features for remaining confusions; final retrain | Final candidate |
| 18:00-22:00 | Buffer, sleep, verify artifacts, choose best CV model | Final submission |
| 22:00-24:00 | Safety margin; do not make risky last-minute changes | Done |

### 9.2 If only ~12 hours remain (likely on the evening of 22 Oct)

Prioritise ruthlessly:

| Time | Work | Deliverable |
|---|---|---|
| 0:00-0:30 | Install packages in the background; audit; shared masks | Environment ready |
| 0:30-2:30 | Preprocess + 100-150 features + LightGBM | **Submission 1** |
| 2:30-5:00 | Add BLS/transit + centroid features | **Submission 2** |
| 5:00-7:30 | Add flare/periodicity/instrument features + thresholds | **Submission 3** |
| 7:30-9:30 | OVR/cascade + ensemble; error analysis | **Submission 4** |
| 9:30-11:00 | Final retrain, validation, best-CV selection | **Submission 5** if it improves CV |
| 11:00-12:00 | Buffer and sleep before the deadline | Done |

If the CNN is not running by hour 8, skip it. A strong LightGBM ensemble is worth more than a rushed neural network.

### 9.3 First 90 minutes checklist

1. Start the package install in a separate terminal:
   ```powershell
   .\myenv\Scripts\python.exe -m pip install --only-binary=:all: --timeout 120 --retries 3 lightgbm astropy tqdm
   ```
2. Write `src/io_data.py`: load mmaps, shared gap mask, cadence arrays.
3. Write `src/preprocess.py`: median normalise, per-CCD common-mode, high/low views.
4. Write `src/features/basic.py`: 60-80 statistics + metadata features.
5. Run extraction on 500 stars, inspect for NaNs/infinities/shape.
6. Train a quick 3-fold LightGBM, check CV.
7. Run full extraction, train 5-fold, write and validate `submission_001.csv`.
8. Upload Submission 1.

Do not wait for perfection. A valid submission on the board is worth more than a perfect pipeline that misses the deadline.

---

## 10. Submission and leaderboard strategy

### 10.1 Validate every submission

```python
import pandas as pd
valid = {"planet","eclipsing_binary","background_binary","pulsating_star",
         "flare_star","starspots","instrument_glitch","quiet_star"}
test = pd.read_csv("dataset_and_problem_statement/test_stars.csv")
sub = pd.read_csv("artifacts/submissions/submission_001.csv")

assert list(sub.columns) == ["id", "label"]
assert len(sub) == 7000
assert sub.id.is_unique
assert set(sub.id) == set(test.id)
assert sub.label.isin(valid).all()
assert (sub.id.values == test.id.values).all()   # same order as test_stars
```

### 10.2 Upload discipline

- Use 2-3 of the 5 daily uploads, not all 5. Save uploads for genuine CV improvements.
- Keep a log: timestamp, feature version, model, CV macro-F1, per-class F1, public LB, notes.
- The public LB is 30% of test (~2,100 stars). Macro-F1 noise is roughly +/-0.01-0.02. Do not chase small public gains.
- Final Round 1 ranking uses the private 70%, based on your best public upload. Choose the model with the best **CV + robustness**, not the one that squeezed the public LB.
- If CV and public LB disagree strongly, suspect sector shift or a feature that does not transfer. Run adversarial validation and leave-one-CCD-out.

### 10.3 Experiment tracking

Create `artifacts/experiments.csv` with columns:

```text
timestamp, feature_version, model, params_hash, cv_macro_f1,
f1_planet, f1_eb, f1_bb, f1_pulsating, f1_flare, f1_spots, f1_glitch, f1_quiet,
public_lb, submission_file, notes
```

Save OOF and test probabilities as `.npy` with a fixed label order. This makes ensembling and threshold tuning reproducible.

---

## 11. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| BLS too slow | Misses deadline | Two-stage candidate-period search; 3,000 periods, 6 workers; benchmark on 100 stars first |
| Package install fails/times out | No LightGBM/astropy | Use sklearn HistGradientBoosting; implement simplified folding on LS/ACF candidates; install in background early |
| GPU OOM / driver issue | CNN unusable | Skip CNN; use CPU ensemble; batch 32, input 631, smaller channels |
| Windows multiprocessing bugs | Extraction hangs | `if __name__ == "__main__"`, pass paths not arrays, spawn, one pool |
| RAM pressure | Swapping | float32, chunks, mmap, free arrays, 6 workers not 12 |
| Sector shift | CV-public gap | Per-star normalisation, per-split common-mode, avoid absolute/time features, adversarial validation, leave-one-CCD-out |
| Label noise (~2%) | Overfitting | Repeated CV, regularisation, label smoothing for CNN, optional confident learning |
| Public-LB overfitting | Poor private score | CV-first selection; limit uploads; robust features |
| Threshold overfitting | CV gain does not transfer | Nested CV for threshold tuning; smooth multipliers; compare with argmax |
| Class confusion `planet` vs `background_binary` | Low macro-F1 | Dip-localised centroid shift, odd-even, V-shape, contamination interactions |
| Class confusion `eclipsing_binary` vs `instrument_glitch` | Low macro-F1 | Secondary eclipse, P/2 modulation, quality-flag alignment, centroid jumps |
| Flare false positives from noise | Low flare F1 | Shape-based detection (rise/decay, multi-cadence), centroid stability, per-star sigma |
| Time | Incomplete pipeline | Follow the strict build order; submit early; CNN optional |

---

## 12. Proposed code layout

```text
Planet_hunters/
├── architecture.md
├── dataset_and_problem_statement/
│   ├── train_flux.npy
│   ├── train_centroid.npy
│   ├── ...
│   └── starter_notebook.ipynb
├── myenv/
├── src/
│   ├── config.py                 # paths, LABELS order, seeds, fold definitions
│   ├── io_data.py                # mmap loaders, shared gap mask, cadence arrays
│   ├── preprocess.py             # normalise, common-mode, high/low views
│   ├── features/
│   │   ├── __init__.py
│   │   ├── basic.py              # statistics + metadata
│   │   ├── periodicity.py        # LS, ACF, band powers
│   │   ├── transit.py            # BLS, folded curve, events
│   │   ├── centroid.py           # global + dip-localised shifts
│   │   ├── flares.py             # shape-based flare detection
│   │   ├── instrument.py         # quality flags, jumps, per-orbit
│   │   └── spots_pulsations.py   # low/high frequency separation
│   ├── extract_features.py       # 6-process parallel driver, resumable
│   ├── train_lgbm.py             # flat multiclass + OOF/test probs
│   ├── train_cascade.py          # priority OVR/cascade
│   ├── train_cnn.py              # optional GPU model
│   ├── tune_thresholds.py        # macro-F1 decision layer
│   ├── ensemble.py               # blend + stack
│   └── make_submission.py        # validation + CSV writer
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_error_analysis.ipynb
└── artifacts/
    ├── features/
    ├── oof/
    ├── models/
    ├── submissions/
    └── experiments.csv
```

Only `architecture.md` exists now; create the rest as you implement. Keep `LABELS` in one place:

```python
LABELS = [
    "planet", "eclipsing_binary", "background_binary",
    "pulsating_star", "flare_star", "starspots",
    "instrument_glitch", "quiet_star",
]
```

This is priority order, which is also convenient for the decision layer. LightGBM may sort classes alphabetically internally; always use `model.classes_` when mapping probabilities back, and save it.

---

## 13. Environment setup

The current environment already has numpy, pandas, scipy, scikit-learn, matplotlib, and joblib. Add only what is needed.

```powershell
cd C:\Users\Kalpana\Documents\GitHub\Planet_hunters
.\myenv\Scripts\Activate.ps1

# Core additions
python -m pip install --only-binary=:all: --timeout 120 --retries 3 lightgbm astropy tqdm

# Nice-to-have (install only if time permits)
python -m pip install --only-binary=:all: --timeout 120 --retries 3 xgboost catboost pyarrow

# GPU PyTorch (large download; only if you will actually use the CNN)
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```powershell
python -c "import lightgbm, astropy; print(lightgbm.__version__, astropy.__version__)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Fallbacks:

- If pip is slow or times out (observed during the audit), do not burn competition time. Use `sklearn.ensemble.HistGradientBoostingClassifier` and a simplified two-stage period search based on Lomb-Scargle/autocorrelation. Install LightGBM later in the background.
- Python 3.13 wheel availability: LightGBM 4.6+, astropy 7.x, and torch 2.7+ should have cp313 wheels. If a package has no wheel, create a separate Python 3.11/3.12 venv for it rather than compiling. Do not rebuild the working environment unless necessary.
- `numba` is optional. Do not install it just for BLS; astropy or the two-stage search is enough.
- `lightkurve` and `celerite` are not needed. Avoid extra large dependencies during the round.

---

## 14. Definition of done for Trial 1

A Trial 1 run is complete when:

- A valid 7,000-row submission is uploaded.
- 5-fold CV macro-F1 is recorded, with per-class F1 and a confusion matrix.
- The best model is selected by CV, not by public LB alone.
- The top two confusion pairs have been inspected and at least one targeted feature has been added for each.
- OOF and test probabilities are saved for future ensembling.
- The pipeline is reproducible from raw `.npy` files with a fixed seed and a fixed feature version.
- Peak memory stayed under the 8 GB target.
- There is time left to sleep before the deadline.

**Success targets**

| Metric | Minimum | Good | Excellent |
|---|---|---|---|
| CV macro-F1 | 0.65 | 0.72 | 0.78+ |
| Worst per-class F1 | 0.40 | 0.50 | 0.60+ |
| Valid submission | Yes | Yes | Yes |
| Public-private gap | < 0.05 | < 0.03 | < 0.02 |

The starter notebook is ~0.43. The jump from 0.43 to 0.70+ comes almost entirely from **transit shape, periodicity, and dip-localised centroid features**, not from model tuning. Build those first.

---

## Appendix A - Concrete high-value feature checklist

Use this as the first implementation list. Each item is a feature or a small group.

**Preprocessing**

- [ ] Drop the 53 shared gap cadences.
- [ ] Per-star median normalisation.
- [ ] Per-CCD common-mode subtraction (train and test separately).
- [ ] Low-pass and high-pass views.
- [ ] Per-star robust sigma.
- [ ] Centroid median and per-CCD common-mode removal.

**Statistics**

- [ ] std, MAD, IQR, min, max, range
- [ ] p1, p5, p25, p75, p95, p99
- [ ] skew, kurtosis
- [ ] fraction below -1%, -2%, -5%
- [ ] fraction above +1%, +2%, +5%
- [ ] lag-1 autocorrelation
- [ ] first-difference std
- [ ] structure function at lags 1, 5, 20
- [ ] per-orbit std and orbit difference

**Periodicity**

- [ ] LS top-5 periods and powers (low-pass view)
- [ ] LS top-5 periods and powers (high-pass view)
- [ ] number of peaks with FAP < 0.01
- [ ] power in 7 period bands
- [ ] top1/top2 power ratio
- [ ] harmonic relation P2/P1
- [ ] autocorrelation peak lag and height
- [ ] sinusoid fit R^2 at best period
- [ ] number of significant frequencies after pre-whitening

**Transit**

- [ ] BLS best period, power, depth, depth SNR, duration
- [ ] number of transits
- [ ] folded depth at phase 0 and 0.5
- [ ] FWHM, ingress/egress duration
- [ ] flat-bottom fraction (U vs V)
- [ ] odd-even depth difference
- [ ] secondary/primary depth ratio
- [ ] ellipsoidal amplitude at P/2
- [ ] event count, deepest event SNR, event durations
- [ ] event spacing regularity
- [ ] fraction of events aligned with quality flags

**Centroid**

- [ ] centroid std/MAD/range (x and y)
- [ ] flux-centroid correlation (global)
- [ ] centroid jump count and max jump
- [ ] **in-dip vs out-of-dip centroid shift (x, y, magnitude, significance)**
- [ ] centroid shift / depth
- [ ] centroid shift at momentum dumps
- [ ] centroid stability during flares

**Flares**

- [ ] shape-based flare count (fast rise + decay)
- [ ] flare rate per day
- [ ] max/median amplitude
- [ ] rise/decay times and asymmetry
- [ ] single-cadence vs multi-cadence fraction
- [ ] total flare energy proxy
- [ ] flare rate vs rotation period
- [ ] flare candidate centroid stability

**Spots/pulsations**

- [ ] low-band / high-band power ratio
- [ ] smoothness (second-derivative std)
- [ ] rotation period from low-pass LS
- [ ] number of harmonics
- [ ] phase coherence across sector halves
- [ ] high-frequency peak count
- [ ] alias pattern features
- [ ] zero-crossing rate

**Instrument**

- [ ] quality-class fractions
- [ ] flux offset at dumps
- [ ] flux offset at scattered-light cadences
- [ ] residual common-mode correlation
- [ ] jump count at 5 sigma and 10 sigma
- [ ] segment mean changes and recovery times
- [ ] single-cadence negative spikes with centroid jumps
- [ ] per-orbit std/min/max/outlier count
- [ ] orbit-to-orbit median flux difference

**Metadata**

- [ ] tmag, teff, logg, radius, contamination
- [ ] camera, ccd (categorical)
- [ ] log_radius, luminosity proxy, density proxy
- [ ] M-dwarf and giant flags
- [ ] contamination x depth, contamination x centroid shift

---

## Appendix B - Priority-aware decision pseudocode

```python
PRIORITY = [
    "planet", "eclipsing_binary", "background_binary",
    "pulsating_star", "flare_star", "starspots",
    "instrument_glitch", "quiet_star",
]

def decode(probs, thresholds):
    # probs: dict or array in PRIORITY order
    for j, label in enumerate(PRIORITY[:-1]):
        if probs[j] >= thresholds[label]:
            return label
    return "quiet_star"
```

For a flat multiclass model, compare this decode against plain `argmax`. Tune `thresholds` on OOF to maximise macro-F1. For a cascade, replace `probs` with each stage's calibrated probability.

---

## Appendix C - Useful implementation notes

**Common-mode subtraction**

```python
def remove_common_mode(rel, camera, ccd):
    out = rel.copy()
    key = camera.astype(str) + "_" + ccd.astype(str)
    for _, idx in pd.Series(range(len(key))).groupby(key):
        rows = idx.values
        common = np.nanmedian(rel[rows], axis=0)
        out[rows] -= common
    return out
```

**Centroid shift during a dip**

```python
def centroid_shift(cx, cy, in_dip, out_dip):
    sx = np.nanmedian(cx[in_dip]) - np.nanmedian(cx[out_dip])
    sy = np.nanmedian(cy[in_dip]) - np.nanmedian(cy[out_dip])
    # robust significance
    sigx = 1.4826 * np.nanmedian(np.abs(cx[out_dip] - np.nanmedian(cx[out_dip]))) + 1e-9
    sigy = 1.4826 * np.nanmedian(np.abs(cy[out_dip] - np.nanmedian(cy[out_dip]))) + 1e-9
    return sx, sy, sx / sigx, sy / sigy, np.hypot(sx / sigx, sy / sigy)
```

**Macro-F1 threshold search (coordinate ascent sketch)**

```python
from sklearn.metrics import f1_score
import numpy as np

def tune_multipliers(P, y, labels, n_rounds=3):
    w = np.ones(P.shape[1])
    best = f1_score(y, np.array(labels)[P.argmax(1)], average="macro")
    for _ in range(n_rounds):
        for c in range(P.shape[1]):
            for cand in np.linspace(0.5, 2.0, 31):
                w2 = w.copy(); w2[c] = cand
                score = f1_score(y, np.array(labels)[(P * w2).argmax(1)], average="macro")
                if score > best:
                    best, w = score, w2
    return w, best
```

Validate this on held-out folds; do not tune and report on the same OOF predictions without a nested check.

**Benchmark before scaling**

Run 100 stars through the full feature extractor and time it. Multiply by 250 (for 25,000 stars) and divide by 6 (workers); if the estimate is over 45 minutes, cut the BLS grid or use the two-stage search. Do this before launching the full extraction.

---

## Appendix D - Why this architecture wins on this hardware

- **Physics-aware features beat raw deep learning** at 18k labelled samples. The problem gives explicit discriminants (U vs V shape, odd-even, secondary eclipse, centroid shift, quality flags); encode them directly.
- **LightGBM is CPU-efficient** and handles the mixed feature types, NaNs, and outliers without fuss. It trains in minutes, leaving time for feature iteration, which is where the score is won.
- **Macro-F1 is a decision problem**, not just a classification problem. Class weights plus OOF threshold/prior tuning are cheap and directly target the metric.
- **The priority rule is part of the label definition.** A cascade or priority-aware decode matches it; a flat argmax can systematically mislabel mixed-signal stars.
- **The sector shift is manageable** because per-star normalisation and per-split common-mode remove most instrument differences. Avoid absolute flux/time features and validate with leave-one-CCD-out and adversarial validation.
- **The GPU is optional** because 4 GB is too small for large models and the data is too small to need them. A compact CNN can add diversity, but the CPU feature + GBM pipeline is the reliable backbone.

Build the backbone first, submit early, then improve. That is the most efficient route to the best Trial 1 result on this laptop.
