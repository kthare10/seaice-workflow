# Code vs. Paper Comparison

**Paper:** Iqrah, Koo, Wang, Xie, Prasad. *Scalable Higher Resolution Polar Sea Ice
Classification and Freeboard Calculation from ICESat-2 ATL03 Data.* arXiv:2502.02700v1
(IPDPSW 2025). <https://arxiv.org/abs/2502.02700>

**Code reviewed:** `bin/*.py` and `workflow_generator.py` at commit `bf39cee`
(2026-09-16).

**Method:** every stage of the code was compared against the corresponding section of
the paper. Where the paper is silent, this document says so rather than filling the gap.
Two facts were checked against the real labeled data (`IS2_Corrected_data`, 6 files,
139,335 segments) rather than inferred:

- Segment spacing is exactly **2.0 m** in every file (the `_10m_` in the file names is
  the Sentinel-2 resolution the paper mentions, not the segment size).
- The files carry `dac, geoid, tide, mss, fpb_corr, h_cor_mean, h_cor_med` columns,
  confirming the paper's pipeline applied those corrections.

Items marked ■ change the science, not just the numbers.

**Status (2026-09-16, fix pass):** every finding below carries a status line.
✅ = implemented as the paper states; 🔶 = implemented with a documented choice where
the paper is silent; ❌ = not reproducible / out of scope, kept as a known gap.
Section 9 records the reproduction numbers from a full cluster run on the authors'
labeled data.

---

## Summary

The workflow reproduces the paper's **stage structure** faithfully (paper Fig. 1) and
matches several specifics exactly: 2 m sampling, three classes, LSTM(16, ELU),
Adam @ 0.003, focal loss, 80/20 split, 20 epochs, 5 km sea-surface window radius, MLP
shape. There are substantive divergences at nearly every stage.

**What is faithful:** stage decomposition, three-class scheme, 2 m sampling, LSTM cell
configuration, MLP configuration, optimizer / learning rate / loss family, split ratio,
epochs, 5 km window radius, along-track-profile and confusion-matrix figure styles,
EPSG:3976 intent, November 2019 default dates.

**Highest-impact divergences:** shuffled-sequence LSTM training (§4), along-track
distance misread and missing geophysical corrections (§2), and the sea-surface estimator
that is labeled "NASA formula" but is not (§6).

---

## 1. Region & data selection

| | Paper | Code |
|---|---|---|
| Ross Sea extent | lon −180 to **−140**, lat −78 to **−70** (§III.A.1) | lon −180 to **−150**, lat −78 to **−60** — `download_atl03.py:31`, `download_sentinel2.py:33` |
| ATL03 release | **006** (§III.A.2) | not pinned — `earthaccess.search_data(short_name="ATL03")`, `download_atl03.py:170` |
| Tracks | 8 specific IS2/S2 pairs, Table I | whatever CMR returns in bbox + date range, truncated by `--max-granules`; `--granule-id` is a single substring filter |

> **Status:** ✅ Ross Sea extent now −180…−140 / −78…−70 in both download scripts. ✅ `version="006"` pinned in the CMR search. ❌ Table I's specific 8 tracks are still not targeted automatically; `--granule-id` selects one, or use `--local-atl03-dir` / `--labeled-csv-dir` with the exact files.

## 2. Preprocessing (§III.A.2)

**Matches:** 2 m bins (`preprocess_atl03.py:30`); mean / median / std height; photon
count; sea-ice surface-type confidence column (`signal_conf_ph[:, 2]`, line 69).

■ **Strong beams are hardcoded to the left beams.** `STRONG_BEAMS = ['gt1l','gt2l','gt3l']`
(line 29). The paper says "only the three strong beams." Which beams are strong depends on
spacecraft orientation, which the code never reads (`orbit_info/sc_orient` is copied into
the merged HDF5 by `download_atl03.py` but is unused). The paper's own figures (Figs. 6–11)
and the labeled data are all `gt1r` / `gt2r` tracks, which the download path would silently
skip.

> **Status:** ✅ Strong beams are read from each beam group's `atlas_beam_type` attribute, falling back to `orbit_info/sc_orient` (0 → left, 1 → right); the merge exits rather than guessing if neither is present.

■ **Along-track distance is misread.** `preprocess_atl03.py:77-78` uses
`heights/dist_ph_along` directly as the along-track coordinate. In the ATL03 product that
field is the photon's offset *from its ~20 m segment's reference point*, not a cumulative
distance; the absolute coordinate is `geolocation/segment_dist_x[segment] + dist_ph_along`.
The code copies `segment_dist_x` in the download step but never uses it. On real ATL03,
2 m binning over this field collapses each beam to roughly 10 bins. (The labeled data's
`x_atc` ≈ 2.8×10⁷ m is the correct absolute form.)

> **Status:** ✅ `along_track_dist = segment_dist_x[segment] + dist_ph_along`, photons mapped to their 20 m segment via `ph_index_beg` / `segment_ph_cnt`. Verified on synthetic granules: exactly 2.0 m spacing, planted heights recovered with −0.01 cm bias / 2.1 cm RMS.

■ **No geophysical corrections.** Paper: "apply a geographical correction based on [25]."
Labeled data confirms: `dac, geoid, tide, mss → h_cor_mean`. Code uses raw ellipsoidal
`h_ph` (≈ −60 m in the Ross Sea). `visualize_results.py` hardcodes y-limits of −0.5 to
3.5 m, so the download path's plots would be empty.

> **Status:** 🔶 `h_cor = h − MSS − tide_ocean − FPB`, the formula that reproduces the authors' `h_cor_mean` to 4×10⁻⁶ m. MSS is ATL03 `geophys_corr/dem_h` where `dem_flag == 3` ("MSS") — the likely source of the paper's `mss` column, but unverified against a real granule (no Earthdata credentials were available). Segments without an MSS flag are dropped with a warning.

■ **Background rate is a different quantity.** Paper: "calculate the background factors";
labeled data has `bcnt_*` / `brate_*` (≈ 10⁶ Hz — the ATL03 `bckgrd_atlas` rate). Code
(`preprocess_atl03.py:174-208`) computes a MAD-outlier count divided by height range — a
heuristic in photons/m with no relation to the ATL03 background rate. `download_atl03.py`
does not copy `bckgrd_atlas` at all.

> **Status:** ✅ `bckgrd_atlas/bckgrd_rate` and `bckgrd_counts` are copied and interpolated in time to each 2 m segment (`brate`, `bcnt`); the MAD heuristic is gone.

■ **"First-photon bias correction" is not the ATBD correction.** Paper applies FPB
correction; labeled data has `fpb_corr` (an additive height term, ~0.01–0.02 m). Code
(lines 153-155) drops the lowest photon from any bin with > 5 photons. Different
operation.

> **Status:** 🔶 Implemented per the ATL07 ATBD Appendix G: apparent width (10 %–90 % cumulative height interval as two-way time), strength (signal photons per shot) and the beam's average CAL-42 dead time index the granule's CAL-19 `ffb_corr` table (bilinear in width × strength, nearest dead-time slice); the result (ps → m) is subtracted. Applied per 2 m segment, matching the per-row `fpb_corr` in the labeled data. The CAL-19 layout is from the ATL03 data dictionary; it has not been exercised on a real granule.

- Paper also mentions "remove reflective reference photons." No corresponding step in
  code.
- Paper's confidence wording is "high sea ice surface type"; code threshold is `>= 3`,
  which includes ATL03 medium confidence (3) as well as high (4). Whether the paper meant
  confidence 4 only is not stated.

> **Status:** 🔶 Confidence threshold is now `== 4` (ATL03 "high"). "Remove reflective reference photons" is implemented as dropping photons with `quality_ph != 0` (afterpulse / impulse-response / transmit-echo-path flags) — this reading of the sentence is an interpretation.

## 3. Sentinel-2 acquisition & auto-labeling (§III.A.3, Table I)

■ **No temporal coincidence constraint.** Paper: S2 within an 80-minute window of the IS2
overpass (Table I: 7.7–47.6 min). Code (`download_sentinel2.py:76-100`) searches the entire
`start_date/end_date` range, sorts by cloud cover, takes the 10 clearest. The scenes used
for labeling may be days or weeks from the track.

> **Status:** ✅ Scenes are searched per granule within ±80 min of the overpass (UTC from `delta_time` + `atlas_sdp_gps_epoch` − 18 s leap seconds) and ranked by time difference. Scenes record which granule they were matched to, and labels transfer only to that granule.

■ **Different segmentation method.** Paper: "thin cloud and shadow-filtered color-based
segmentation [5]." Code (`auto_label.py:101-107`): `NIR < 0.1` → water;
`NIR < 0.4 & blue/red > 1.05` → thin ice; else thick. No cloud or shadow filter. The
comment says "empirically derived."

> **Status:** ✅ The published HSV ranges of Iqrah et al. 2023 (Sec. 3.2) are applied to the 8-bit true-color (TCI) asset: V ≥ 205 thick, 31–204 thin, ≤ 30 open water. 🔶 The thin-cloud/shadow filter is described only as a list of OpenCV operations; the L2A Scene Classification Layer (classes 0, 1, 3, 8, 9, 10) is used as an explicit substitute.

- **No image shift / co-registration.** Paper shifts each S2 scene 0–550 m to align with
  IS2 (Table I). Code: none.
- **No manual correction** of cloudy / transition regions (paper does). Inherent to
  automation, but it means the training labels are not equivalent.
- **EPSG:3976 is dead code.** Paper projects both datasets to EPSG:3976. `CRS_ANTARCTIC`
  is defined (`auto_label.py:41`) but never used; the code reprojects ATL03 into each
  scene's native CRS instead. Geometrically equivalent for nearest-pixel lookup, but not
  what the paper states.
- **Height-threshold fallback labeling is not in the paper.** `auto_label.py:172-200,
  260-262`: if no S2 overlaps, labels are assigned by `mean_h > 0.3 / > 0.05`. Given §2
  (uncorrected heights ≈ −60 m), this fallback would label everything thick ice.
- `max_distance=100` parameter in `overlay_labels` is accepted and ignored.

> **Status:** ✅ Label raster reprojected to EPSG:3976 at 10 m and ATL03 transformed to EPSG:3976. ✅ Table I shifts applied by ICESat-2 date (direction interpreted in the EPSG:3976 grid axes). ✅ Height-threshold fallback removed; the job writes an empty output and exits non-zero. ❌ Manual correction of cloudy / transition regions cannot be automated.

## 4. Features & LSTM (§III.B.1)

| | Paper | Code |
|---|---|---|
| Features | **6** per time step: "height/elevation, height std, high-confidence photon, photon rate changes, background photon, background photon rate changes" | **5**: `mean_h, median_h, std_h, photon_count, bg_rate` (`train_model.py:33`). No rate-of-change features; adds median. |
| Context window | point *n* depends on *n−2, n−1, n+1, n+2*; "batch size of 5 … per time step" | `SEQUENCE_LENGTH = 10`, label = **last** element (line 142). One-sided, twice as long. |
| LSTM layer | 16 units, ELU, **dropout 0.2 in the LSTM layer** | 16 units, ELU ✓; no LSTM dropout |
| Dense stack | **7** layers: 32, 96, 32, 16, 112, 48, 64 — **ELU** | **6** layers: 64, 32, 32, 16, 16, 8 — **ReLU**, with Dropout(0.3), Dropout(0.2) inserted (lines 77-84). Docstring says "7 Dense layers." |
| Output | Dense(3, softmax) | ✓ |

> **Status:** ✅ 6 features: `mean_h, std_h, pcnth, d_pcnt, bcnt, d_brate`. 🔶 The paper names the features without defining them; `pcnth` = high-confidence photons per shot, `bcnt` = ATL03 50-shot background counts, and "changes" = first difference along track are this implementation's reading. ✅ Centered 5-segment window (n−2…n+2), label of the center. ✅ LSTM(16, ELU, dropout 0.2) → Dense 32, 96, 32, 16, 112, 48, 64 (ELU) → softmax(3).

■ **Training sequences are built from shuffled rows.** `train_test_split` (line 197,
`shuffle=True` default) randomizes segment order, *then* `prepare_lstm_sequences`
(line 209) forms sliding windows. Each training sequence is therefore 10 spatially
unrelated segments, so the along-track dependency the paper's LSTM is designed to exploit
("progression of data points … simulates the change of sea ice cover") is destroyed in
training. Windows also cross granule / beam boundaries (no grouping) in both training and
inference.

> **Status:** ✅ Windows are built per track (granule + beam) in along-track order *before* the split; the split is over windows, stratified by center label. Inference edge-pads each track so every segment is classified.

## 5. MLP & training setup (§III.B.2, §IV.A)

**Matches:** Dense(32, ReLU) → softmax; dropout 0.2 (§IV.A states both models use 0.2);
Adam lr 0.003; focal loss; 80/20 stratified split; 20 epochs.

- **Batch size:** paper **32** (§IV.A, Table IV); code default **64**
  (`train_model.py:147, 314`).
- **Focal loss parameters** (γ = 2.0, α = 0.25) are not given in the paper —
  unverifiable. Note: the code's α is a scalar applied uniformly to all classes
  (line 54), so it scales the loss but does not reweight minority classes; only γ acts on
  imbalance.
- **Metrics:** paper reports accuracy, precision, recall, F1 (Table III). Code tracks
  accuracy only (line 91); nothing computes precision / recall / F1.
- **StandardScaler** normalization (line 193) — paper does not mention feature scaling
  either way.
- **Distributed training:** paper uses Horovod synchronous data-parallel over 1–8 A100s
  (§III.B.3, Table IV). Code: single-process TensorFlow, one GPU. Likewise the paper's
  PySpark map-reduce for auto-labeling and freeboard (Tables II, V) has no counterpart —
  the workflow's only parallelism is the Pegasus fan-out of `classify_seaice`. The paper's
  scalability results are not reproduced; the workflow parallelizes a different stage by a
  different mechanism.

> **Status:** ✅ Batch size 32. ✅ Precision / recall / F1 (per class, macro, weighted) and the confusion matrix on the held-out 20 % are written to `training_metrics.json`. 🔶 Focal-loss γ = 2, α = 0.25 kept (paper gives none). ❌ Horovod / PySpark scaling is not reproduced; the workflow's parallelism remains the Pegasus classify fan-out.

## 6. Freeboard (§III.D, Eqs. 1–3)

**Matches:** h_f = h_s − h_ref; 5 km radius / 10 km window (`calculate_freeboard.py:31`);
linear interpolation where no open water is found; per-track computation
(`groupby(['granule','beam'])`, line 141).

■ **The sea-surface estimator is not NASA's formula, despite the comment.**
`calculate_freeboard.py:70` says "Using NASA formula: weighted mean with
uncertainty-based weights." Lines 84-88 implement `w = 1 / (|d_i − d| + 1)` —
inverse-*distance* weighting. The paper's Eq. 2 weights by *height*:
`w_i = exp(−((h_i − h_min)/σ_i)²)` per lead, then Eq. 3 combines leads by
`1/σ²_lead`. No `h_min`, no `σ_i`, no per-lead grouping, no variance weighting appears in
the code. The paper explicitly evaluated four methods (minimum, average, nearest-minimum,
NASA) and chose NASA's as primary; the code implements none of the four.

- **Windowing:** paper describes 10 km windows "with a sliding overlap of 5 km"
  (ATL10-style stepped windows). Code evaluates a window centered on *every* segment.
- **Fallbacks not in the paper:** no open water → 5th percentile of heights (lines 67,
  94); open-water freeboard forced to 0 (line 156). Paper is silent on both.
- **No ATL07 / ATL10 comparison.** Figs. 6b–11d compare against ATL07 (Koo method) and
  ATL10. Nothing ingests those products, so those figures cannot be reproduced.

> **Status:** ✅ Eq. 2 per lead (runs of consecutive open-water segments) with `w_i = exp(−((h_i − h_min)/σ_i)²)`, Eq. 3 across leads by inverse variance. 🔶 σ_i² = std_h²/N (standard error of the segment mean), floored at (1 cm)²; the paper does not define σ_i. ✅ 10 km windows stepped 5 km; windows without leads are linearly interpolated. ✅ Fallbacks removed: no open water on a track → freeboard NaN with a warning; open-water freeboard is no longer forced to 0. ❌ ATL07 / ATL10 comparison remains out of scope.

## 7. Evaluation (§IV.C, Fig. 4, Table III)

- Paper's Fig. 4 confusion matrix and Table III are on the **20% held-out set**. The
  workflow's confusion matrix (`visualize_results.py:114-118`) is over **all classified
  segments**, including the 80% used for training. The only held-out number is
  `test_accuracy` in `training_metrics.json`.
- Table III's precision / recall / F1: not computed anywhere.

> **Status:** ✅ `visualize_results.py --metrics-input training_metrics.json` plots the held-out confusion matrix (titled as such) and `summary_statistics.json` carries the held-out accuracy / precision / recall / F1 separately from the all-segment agreement, which is now labeled as including training data.

## 8. Mode-consistency issue introduced by the two input paths

In `--labeled-csv-dir` mode, heights are the paper's corrected `h_cor_*`, `bg_rate` is
the real `brate_mean` (~10⁶), and along-track distance is correct. In the download path,
heights are uncorrected ellipsoidal, `bg_rate` is the MAD heuristic (~0–1), and
along-track distance is wrong (§2). A model trained in one mode cannot be applied in the
other. The labeled data also carries columns that plausibly correspond to the paper's 6
features (`pcnth_*`, `bcnt_*`, `brate_*`); the harmonizer discards them because the model
is fixed to the workflow's 5 features.

> **Status:** ✅ Both `preprocess_atl03.py` and `prepare_labeled_csv.py` emit the same 22-column schema (corrected heights, ATL03 background, photon rates, along-track deltas, `mss` / `tide_ocean` / `fpb_corr` for traceability). A model trained in one mode applies in the other.

---

## 9. Reproduction results (authors' labeled data, 6 tracks, 139,335 segments)

Full Pegasus run on the FABRIC cluster (`run0004`, 33/33 jobs), labeled-CSV mode:
harmonize → train → classify (6-way fan-out) → merge → freeboard → visualize. LSTM,
20 epochs, batch 32, 139,311 centered windows, stratified 80/20 split
(111,448 / 27,863). `train_model` ran on a real GPU (`/physical_device:GPU:0`).

### Classification, held-out 20 % (paper Table III / Fig. 4)

| | Paper | This run |
|---|---|---|
| Accuracy | 96.56 % | **95.26 %** |
| Precision | 97.00 % (averaging unspecified) | 75.8 % macro / 95.1 % weighted |
| Recall | 96.09 % | 68.2 % macro / 95.3 % weighted |
| F1 | 96.54 % | 71.5 % macro / 95.1 % weighted |
| Thick-ice recall | 98.39 % | **98.83 %** |
| Thin-ice recall | 73.80 % | 56.98 % |
| Open-water recall | 60.25 % | 48.80 % |

Over all 139,335 segments the predicted class mix is 93.3 % / 4.8 % / 1.9 %
(thick / thin / water) against a 92.0 % / 5.5 % / 2.5 % truth mix, and agreement is
95.43 %.

Accuracy lands 1.3 points under the paper and thick-ice recall slightly above it; the
shortfall is entirely in the two minority classes. Causes the paper does not settle:
focal-loss parameters (unspecified, and a scalar alpha does not re-weight classes), the
exact feature definitions (Sec. 4), and the manual label corrections the authors applied
in transition and cloudy regions, which an automated pipeline cannot reproduce.

### Freeboard (Eqs. 1-3)

Thick ice mean 0.655 m / median 0.612 m; thin ice mean 0.041 m / median 0.034 m. Thin
ice sitting an order of magnitude below thick ice is the expected physical ordering and
is the main sanity check on the height correction chain.

### Not verified against a real ATL03 granule

No Earthdata credentials were available, so the raw-ATL03 path ran only against synthetic
granules built to the v006 data dictionary: the `atlas_beam_type` attribute,
`dem_flag == 3` as the MSS source, the CAL-19 / CAL-42 table layout, and `bckgrd_atlas`
alignment. The labeled-CSV path above does not exercise any of them.

### A failure mode worth recording

An earlier run of this same configuration (`run0003`) reported 33/33 success while
producing 89.9 % open water and 11 % agreement. `train_model` declared only `model.h5`
and `training_metrics.json` as outputs, so the `model.scaler.npz` written beside the
model never reached the classify jobs; they warned, standardized nothing, and fed raw
features to a model trained on standardized ones. The warning did not fail the job, so
the DAG went green. `classify_seaice.py` now exits non-zero when the scaler is absent.

## Fix priority (original, for reference)

1. **§4** — build LSTM sequences per track in along-track order *before* splitting;
   centered 5-point window; paper's dense stack.
2. **§2** — correct along-track distance (`segment_dist_x + dist_ph_along`); apply
   `geophys_corr` (geoid, ocean tide, DAC) and select strong beams from `sc_orient`;
   read `bckgrd_atlas` for background rate.
3. **§6** — implement Eqs. 2–3 for the sea-surface reference.
4. **§5 / §7** — batch size 32; precision / recall / F1; confusion matrix on the held-out
   split.
5. **§3** — S2 temporal coincidence window; cloud / shadow filtering.
6. **§1** — region extent; ATL03 release 006.
