# Paper Reproduction Audit — Sensors 2026 (representation/CNN/training) + INFOCOM 2024 (OFDM frontend)

**Track:** Baseline-3 Phase B′, Track P (reproduction audit)
**Date:** 2026-11-21
**Purpose:** This document is the fidelity contract for the Wi-Fi OFDM modulation-classification experiment. Later stages (representation rebuild, CNN rebuild, smoke tests, training runs) must conform to the specs, arithmetic, artifact inventory, and documented engineering assumptions below. Where the papers are silent, the "fixed choice" recorded in §6 is binding unless explicitly re-negotiated in a later stage.

**Papers audited**

| # | Paper | Venue / ID | Role in our experiment |
|---|---|---|---|
| P1 | Nelega, R.; Mezei, M.-M.; Polgar, Z.A.; Kovacs, G.; Puschita, E. — *Deep Learning-Based Automatic Modulation Classification for OFDM Signals: From Synthetic Training to OTA Evaluation* | Sensors 2026, 26(10):2945; DOI 10.3390/s26102945; PMID 42197754; PMC13211014; epublish 8 May 2026; CC BY 4.0 | Primary: amplitude-phase 2D-histogram representation, CNN architecture, training config, cross-domain results |
| P2 | Kim, B.; Mecklenbräuker, C.; Gerstoft, P. — *Deep Learning-based Modulation Classification of Practical OFDM Signals for Spectrum Sensing* | IEEE INFOCOM 2024, Vancouver, 20–23 May 2024; arXiv:2403.19292v1 (28 Mar 2024); arXiv license CC BY-NC-SA 4.0 | Frontend: blind OFDM parameter estimation, CP/symbol-start estimation, CFO handling, adjacent-symbol phase-difference mechanism |

**Access log (fetch failures recorded as instructed):**
- `https://www.mdpi.com/1424-8220/26/10/2945` → HTTP 403 (bot-blocked). Retry 1: `…/2945/htm` → 403. Retry 2 (different form): `…/2945/pdf` → 403. **Resolved via** Europe PMC open-access mirror: `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13211014/fullTextXML` (full JATS XML of the article, CC BY). DOI `https://doi.org/10.3390/s26102945` resolves correctly (confirms metadata).
- `https://arxiv.org/abs/2403.19292` and `https://arxiv.org/html/2403.19292v1` → both fetched successfully (full text).
- Hugging Face: `huggingface.co` and `datasets-server.huggingface.co` are **DNS-poisoned in this environment** (resolve to a Facebook IP block; curl HTTP 000; webfetch timeout — retried once each with alternate URL forms, still failed). **Fallback used:** `hf-mirror.com` (full Hub-API mirror) for dataset metadata, file tree, README, and ranged reads of the ZIP artifacts. `datasets-server` `/info` and `/first-rows` remain unreachable; structure was instead verified by reading ZIP central directories via HTTP `Range` requests (bytes fetched: ~2.9 MB total, no bulk download).

---

## 1) Sensors 2026 (P1) — representation, CNN, training spec

### 1a) Amplitude–phase 2D histogram construction (Section 3.1.1; axis semantics from Section 2)

Verified facts (quotes in §7, Q1–Q4):

- **Axes.** One axis is the subcarrier amplitude (normalized), the other is the **phase difference between consecutive OFDM symbols** (normalized). P1 Section 2, describing the inherited representation: *"the horizontal axis corresponds to normalized amplitude, and the vertical axis represents phase difference."*
- **Binning.** 50 × 50 bins over a "1 × 1 grid with a resolution of 1/50 per axis" (Q3). Both axes span exactly one unit.
- **Amplitude axis construction (per 128-symbol block):**
  1. Amplitude outliers exceeding the **99th percentile** are discarded: *"The noisy QAM symbols were then grouped into blocks of 128 OFDM symbols, while amplitude outliers exceeding the 99th percentile were discarded."* (Q2)
  2. Remaining amplitudes are normalized **by the maximum value within the 99% interval**: *"Using only the data subcarriers, the amplitudes were normalized by the maximum value within the 99% interval"* (Q3) — i.e., per-instance normalization `amp = |Y| / max(|Y| ≤ P99)` ∈ [0, 1].
- **Phase axis construction.** *"…the phase difference between consecutive symbols was normalized by 2π"* (Q3). The paper does **not** state the wrap convention ((−π, π] vs [0, 2π)) nor the bin-index rule; P2's official code wraps raw differences to (−π, π] and uses 50 bins over that span (see §2f, §6 E2).
- **Per-instance max normalization (training-time, second stage).** Section 4.2: *"each 2D histogram is independently normalized to the [0, 1] interval via per-instance maximum normalization"* (Q7). **Verified divergence:** the released HF dataset stores histograms that **sum to exactly 1.0** (INFOCOM eq. (14) style `Z′=Z/𝒵`), *not* max-normalized histograms (see §5.6). The per-instance max normalization is therefore a downstream training-input step in P1, applied on load.
- **Why 50×50?** Neither paper justifies the number. P1 Section 1 introduces "50 × 50 amplitude-phase histograms" without ablation; P2 Table VI sets `{S_P, S_Q} = {15, 50}` also without ablation; P2's only rationale is qualitative (histograms capture high-order constellations and are length-invariant, Q20). **Conclusion: 50 is an inherited constant from P2's S_Q=50; treat as fixed, unjustified.**

### 1b) Symbol organization per histogram (Sections 3.1.1, 3.1.3, 5.2)

- Base generation unit: **8-symbol IEEE 802.11 data fields** (one MCS each: BPSK, QPSK, 16QAM, 64QAM = MCS 0, 1, 3, 5) (Q1).
- One histogram aggregates a **block of 128 OFDM symbols = 16 independent groups of 8 symbols**. Section 5.2 is explicit (Q8): *"the 128-symbol amplitude-phase histograms are constructed by aggregating 16 independent groups of 8 OFDM symbols."* The 16 groups are independent channel realizations (coherence argument in Q8: channel "effectively constant within each 8-symbol block, but varies across the 16 blocks").
- **Scope of "consecutive symbols" (paper-ambiguous → binding choice E1).** The paper says only "the phase difference between consecutive symbols" (Q3). Given the 16 × 8 grouping (independent channel/fading per group) and that D5/OTA bursts deliver 8 payload symbols per burst (Q6), phase differences across group boundaries are physically undefined (different channel realizations). **Fixed choice (E1):** compute phase differences only **within** each 8-symbol group (7 differences per group; 16 groups → 112 difference values per subcarrier per histogram). The last symbol of each group contributes no difference.
- **Bursts/gaps (conducted D5 and OTA, Section 3.1.3 / 3.2).** Frames = custom 512-sample multifrequency signalling sequence(s) + **two consecutive OFDM data bursts** sharing (MCS, channel, SNR). Receiver: burst detection (|·| → dual moving-average smoothing → hysteresis comparator), strict length gates (~512 signalling / ~1500 data samples), spectral decode of the signalling, then a hardware synchronization pipeline (symbol timing + coarse/fine CFO), then *"After stripping the frame header, 8 OFDM payload symbols were retained and fed directly into the previously established processing pipeline. They were demodulated, organized into groups of 128 OFDM symbols"* (Q6). Only OFDM payload samples reach the histogram; signalling bursts and inter-burst gaps are discarded.
- **Preamble avoidance (synthetic analog in P2 IV-B):** P2 removes the first 2000 samples of each capture to avoid the Wi-Fi preamble (Q19). P1's synthetic data has no preamble (direct 8-symbol field generation).

### 1c) Subcarrier selection (Section 3.1.1)

- **"20 MHz IEEE 802.11 signals with 52 data subcarriers"** (Q1). Under the 20 MHz legacy/HT IEEE 802.11 grid (64-FFT), the occupied band is −26…−1, +1…+26 = **52 non-null subcarriers = 48 data + 4 pilots**; 12 nulls (incl. DC). The paper's phrase "52 data subcarriers" therefore most plausibly denotes the **52 occupied bins used for the histogram**; whether the 4 pilots are additionally excluded (→ 48) is **not stated** → binding choice **E4** in §6 (we use the 52 occupied non-null bins; alternative reading noted).
- **Null/pilot/DC exclusion method:** P1 uses **known-position selection** ("Using only the data subcarriers", Q3). This *differs* from P2, which removes nulls **blindly** by discarding the `N_null` subcarriers with the smallest average amplitudes (Q17: N_null = 8 for 64-FFT HT, 32 for 256-FFT HE). Both papers' 20 MHz counts are consistent with the 12-null layout (P1 excludes all nulls at known positions; P2's blind N_null=8 is a noisier subset — see §6 E4).
- **Amplitude filtering:** the only amplitude-based filter in P1 is the per-block P99 outlier discard (Q2). No per-subcarrier amplitude screening (no β threshold — that is P2's 5G-only mechanism, Q18).

### 1d) CNN architecture (Section 4.1; parameter verification in §3)

Exact layer table (all convs 3×3, "same" padding — implied by the quoted 50→6 spatial reduction being attributed solely to pooling):

| # | Layer | Out shape | Params (weights + bias) |
|---|---|---|---|
| 0 | Input | 1 × 50 × 50 | — |
| 1 | Conv2d(1→32, 3×3) + BN + ReLU | 32 × 50 × 50 | 288 + 32 = 320 (+BN 64) |
| 2 | MaxPool 2×2 | 32 × 25 × 25 | — |
| 3 | Conv2d(32→64, 3×3) + BN + ReLU | 64 × 25 × 25 | 18,432 + 64 = 18,496 (+BN 128) |
| 4 | MaxPool 2×2 | 64 × 12 × 12 | — |
| 5 | Conv2d(64→128, 3×3) + BN + ReLU | 128 × 12 × 12 | 73,728 + 128 = 73,856 (+BN 256) |
| 6 | MaxPool 2×2 | 128 × 6 × 6 | — |
| 7 | Flatten | 4608 | — |
| 8 | FC(4608→512) | 512 | 2,359,296 + 512 = 2,359,808 |
| 9 | Dropout p = 0.5 | — | — |
| 10 | FC(512→n_classes) | n_classes | 512·n + n |

Paper support: input 1×50×50; "three sequential convolutional blocks utilizing compact 3 × 3 kernels"; filters "progressively doubles (from 32 to 64, and finally to 128)"; "Each convolution is integrated with Batch Normalization … and a ReLU"; "2 × 2 Max-Pooling layers at the end of each block, reducing the spatial resolution from 50 × 50 down to 6 × 6"; "flattens the resulting 128 × 6 × 6 feature map (4608 elements) into a 512-dimensional hidden layer"; "dropout rate of 0.5 … prior to the final linear layer, which maps … to the 4 target modulation schemes" (Q5). VGG-inspired (ref [23] = Simonyan & Zisserman).

### 1e) Training configuration (Section 4.2)

| Item | Value (quoted where load-bearing) |
|---|---|
| Split | 80/20 train/validation ("The datasets are divided into training and validation sets using an 80/20 split", Q7). Stratification not specified → E8. |
| Input normalization | per-instance max normalization to [0,1] (Q7; second-stage, see §1a) |
| Optimizer | Adam |
| Loss | Cross-Entropy |
| Batch size | 1024 |
| Learning rate | 10⁻³ |
| Weight decay | 10⁻⁴ |
| Early stopping | 10-epoch patience on validation loss; "ensuring the selection of the best-performing weights" (Q7) |
| Epochs | not fixed (early-stopped); no LR scheduler mentioned → E9 |
| Data augmentation | none at training time; dataset-level augmentation only (D3's three synchronization-impairment configurations, Table 1) |
| Class balance | not mentioned; datasets are balanced by construction (1562 per (MCS, SNR, channel); per-mod counts in D5 verified ~4.5–4.7K each) |
| Runs | 10 independent trainings per scenario; all reported metrics averaged over the 10 runs (Q7) |
| Hardware/framework | PyTorch, NVIDIA A100-PCIE 40 GB; inference latency measured on RTX 5070 (0.374 ms/sample) (Q7) |

### 1f) Reported results and the conditions under which they hold (Section 5, Tables 3–4)

**Training-set conditions:** 20 MHz IEEE 802.11 (non-HT and HT), 4 classes (BPSK/QPSK/16QAM/64QAM); synthetic sets add TGn indoor channel models A–F and AWGN at 0–36 dB in 3 dB steps (D1: perfect sync; D2/D3/D4: + symbol-clock & CFO impairments per Table 1 below); conducted set D5 adds hardware impairments (BladeRF x40 loopback + two-board), TGn A–F plus ideal channel, noise 0–30 dB in 2 dB steps.

**Impairment configurations injected in D2/D3/D4 (Table 1, "Configuration of impairment scenarios"):**

| Demodulation window | Timing error (SER) | Frequency deviation (CFO) | Max fractional sampling error |
|---|---|---|---|
| Standard (full length) | 2 samples | 5% | 1 × TER |
| Reduced (CP short) | 4 samples | 5% | 1 × TER |
| Reduced (CP short) | 2 samples | 0.25% | 0.2 × TER |

(CFO "applied relative to the subcarrier frequency spacing"; fractional sampling errors `T_ER` shift the sampling instant by a random fraction of the sampling period with resolution `T_ER/1000`; "CP short" = deliberately shortened demodulation window to keep all samples inside the current symbol.)

**Headline OTA results (Table 4; all mean ± std over 10 runs; 95% CI given):**

| Model | Training set | Size | Impairments | OTA accuracy |
|---|---|---|---|---|
| D1 | mid-scale perfectly synchronized synthetic | 487,344 | multipath + AWGN | 91.54% ± 0.29% [91.33, 91.75] |
| D2 | mid-scale sync-impaired synthetic | 487,344 | + sync errors | 92.20% ± 0.54% [91.81, 92.56] |
| D3 | large-scale sync-impaired synthetic | 1,947,117 | + sync errors | **93.36% ± 0.42%** [93.06, 93.66] |
| D4 | small-scale sync-impaired synthetic | 18,591 | + sync errors | 89.89% ± 0.52% [89.52, 90.26] |
| D5 | small-scale conducted hardware | 18,591 | + hardware impairments | 89.38% ± 0.60% [88.95, 89.81] |

**Per-class and SNR structure (D3 model unless noted):**
- BPSK and QPSK: ~100% for **all** training strategies (OTA) — "near-perfect performance (100%) for BPSK and QPSK modulations" (Q10).
- 16QAM is the weak class: D3 peak 80.9%; D2 78.9% vs D1 76.3%. **Confusion pattern:** *"The primary source of error is an asymmetric confusion between the QAM schemes, where 19.0% of 16-QAM signals are misclassified as 64-QAM"* (Q10) — i.e., the 16QAM constellation spreads into the denser 64QAM decision space at low SNR; 64QAM itself holds 92.6% ± 1.5%.
- SNR dependence: 16QAM "sharp degradation below 15 dB"; "Both high-order modulations eventually converge to 100% accuracy above 20 dB" (Q10). Low-SNR (below 10 dB): D3 best; D5 falls below 50% at the lowest SNRs (hardware DC-offset artifacts not present OTA).
- Spatial: corridor position G (24.7 dB) and lab B3 (16.1 dB) → up to 100.0%; corridor position B (5.0 dB) → 76.0% (lowest); per-position σ typically ≤ 1.2% (Q10).
- **Scope/conditions of validity:** indoor static OTA only ("the scope of these conclusions is strictly bounded to controlled indoor environments"); single modulation per detected signal; SISO; 20 MHz; the representation is *argued* (not tested) to extend to mobility because 8-symbol groups (32 µs) sit inside channel coherence time (Q8). No HE/Wi-Fi-6 or 5G formats in P1 (P2 covers those).

### 1g) Items P1 leaves unspecified → engineering assumptions (full list with fixed choices in §6)

See §6 items **E1–E14**. Headline: phase-difference scope within 8-symbol groups (E1); phase wrap/bin convention (E2); histogram-on-load normalization (dataset is sum-normalized, paper trains on max-normalized) (E3); pilot handling in "52 data subcarriers" (E4); P99 computed over what population (E5); .mat axis orientation (E6); SNR-bin label semantics in the released zips (E7); split stratification (E8); epochs/scheduler (E9); CNN padding mode (E10); OTA LTF-based SNR estimator (E11); D3 size arithmetic (1,947,117 vs 4 × 487,344 = 1,949,376, unexplained −2,259) (E12); format inventory across zips (E13); frontend sync algorithms for hardware capture (E14).

---

## 2) INFOCOM 2024 (P2) — frontend spec (Section III; formulas as in arXiv:2403.19292v1)

P2 goal: classify the modulation of symbols on OFDM subcarriers for Wi-Fi 6 (HT/HE payload) and 5G DL (PDSCH), **without** preamble, control information, or known FFT size/CP — i.e., a spectrum sensor. Inputs are resampled to f_RX = 20 MHz; 5G is further resampled to 30.72 MHz.

### 2a) OFDM parameter estimation — symbol period / SCS (Section III-A, Eqs. (2)–(4))

- Cyclic autocorrelation function (CAF): `R_yy(α, τ) = Σₙ R_yy(n, τ) e^{−j2παn}` (Eq. 2).
- Deployed estimator (Eq. 3, a variant of Punchihewa et al. [24] with multi-sample averaging):

  `R̂_yy(α, ℓ) = 1/(L − l − ℓ + 1) · Σₙ₌₀^{L−l−ℓ} { Σᵢ₌₀^{l−1} y[n+i] y*[n+i+ℓ] } · e^{−j2παn}`

  Text sets **l = 8** ("corresponding to the shortest CP length in our scenario"); the evaluation section reports **l = 2, 4 achieve 99% accuracy and outperform l = 1** (Q21); the official code uses `inputLen = 2` (§4). → binding choice **E15**.
- CP-induced repetition at lag ℓ = f_RX/Δf_SCS makes |R̂_yy(0, ℓ)| peak there. IFFT length estimated over the candidate set **ℓ_C = {64, 256, 333, 667, 1333}** (per-SCS candidates at 20 MHz) via Eq. (4): `T_IFFT = ℓ′/f_RX s.t. ℓ′ = argmax_{ℓ∈ℓ_C} |R̂_yy(0, ℓ)|`. Symbol period then follows from CP length.

### 2b) CP length and symbol-start estimation (Section III-B, Eqs. (7)–(8))

- CP-length via cycle-frequency peak (Eq. 5): `T_CP = (1/α′ − ℓ′)/f_RX s.t. α′ = argmax_{α∈α_C^{ℓ′}} |R̂_yy(α, ℓ′)|`, where α_C^{ℓ′} are the candidate cycle frequencies 1/{ℓ′ + f_RX·T_CP} given ℓ′.
- **Symbol-start correlation statistic (Eq. 7):**

  `R_yy(m, N_FFT) = (1/N_CP) Σ_{i=0}^{N_CP−1} y[m+i] · y*[m+i+N_FFT]`

  |R_yy| peaks when m is the first index of a CP.
- **Peak search procedure:** "we search for a sample whose amplitude is larger than both of its neighboring samples while ensuring that the minimum distance between two adjacent peaks is 90% of the OFDM symbol duration (i.e., (256+64)×0.9 = 288-time samples for HE format with 3.2 µs CP), to avoid selecting undesired local peaks" (Q16).
- **Symbol-start estimate (Eq. 8):** `p = Median_i{ mod(p′_i, N_FFT + N_CP) }` over the detected peaks {p′_0 … p′_{S−1}}.
- **Guard against start error:** FFT windows are taken starting at **p + N_CP/2**: "This sequence is entirely within a single OFDM symbol for estimation error of p below N_CP/2" (Q16). (Verified in official code: windows at `firstIndexSym + int(CPLen/2) + iSym·(nSubC+CPLen)`.)
- Reported accuracy: exact start index < 60% at tight tolerance, but **99% within N_CP/4 samples** (Q21).

### 2c) CFO estimation and compensation (Section III-B Eq. (11), Section III-C Eq. (12))

- **Estimator (Eq. 11):** `∠( y[p + N_FFT + i] · y*[p + i] ) = 2π Δf_c / Δf_SCS`, with `i ∈ {⌊N_CP/4⌋, …, ⌈3N_CP/4⌉}` so the CP samples used stay inside the CP even if the p error is below N_CP/4; CFO = the average of Eq. (11) over multiple OFDM symbols. (Official code: mean over symbols of the mean over the i-range, then `CFO = mean phase / (2π·N_FFT) · f_s`, iterated `nIter` times.)
- **Aliasing limit:** "If the absolute value of the CFO is larger than Δf_SCS/2, the CFO cannot be accurately estimated due to aliasing" (Q16).
- **Residual CFO handling (5G, Eq. 12):** after CP-based correction, residual CFO is an integer multiple of Δf_SCS ("The algorithm makes the corrected CFO a multiple of Δf_SCS, not a zero"). Its effect on the feature is a **constant phase offset**:

  `Δ∠Y^s_Δn[k] = ∠Y^{s+1}[k + Δf_c/Δf_SCS] − ∠Y^s[k + Δf_c/Δf_SCS] − 2π Δf_c (1/Δf_SCS + T_CP)`

  i.e., a uniform **vertical shift of the histogram's phase axis**, which the CNN is trained to absorb as same-class variation: "The neural network can be trained to identify histogram movements along the y-axis as a single class" (Q20). Because of the T_CP term, pairs where symbol s+1 carries a **long CP are excluded** from features (5G).

### 2d) KEY MECHANISM — adjacent-symbol phase differences cancel synchronization error (Section III-B, Eqs. (6), (9), (10))

With a sampling offset Δn *inside* an OFDM symbol (Eq. 6):

```
Y^s_Δn[k] ≜ F(y^s[n − Δn]) = Y^s[k] · e^{−j2πΔn k / N_FFT}
```

(phase drift linear in subcarrier index k; amplitude unchanged). Then (Eq. 9):

```
Δ∠Y^s_Δn[k] ≜ ∠Y^{s+1}_Δn[k] − ∠Y^s_Δn[k]
             = ∠{Y^{s+1}[k] e^{−j2πΔnk/N}} − ∠{Y^s[k] e^{−j2πΔnk/N}}
             = ∠Y^{s+1}[k] − ∠Y^s[k].
```

"The feature used to identify the modulation type is (Eq. 10): `Y_f^s[k] ≜ |Y^s_Δn[k]| · e^{j Δ∠Y^s_Δn[k]}`." Constant Δn across the pair is guaranteed by spacing the two FFT windows exactly one OFDM symbol apart. This is the load-bearing cancellation that makes the histogram robust to symbol-level timing error (and is the mechanism P1's representation inherits; P1 Section 2 bullet: "By relying on phase differences between adjacent symbols, identical phase drifts mathematically cancel out", Q9).

### 2e) Null-subcarrier exclusion and data-subcarrier screening

- **Wi-Fi 6 (blind):** "For Wi-Fi 6, the null subcarrier symbols are eliminated by discarding symbols with the N_null smallest average amplitudes." With N_null = 8 (HT, N_FFT=64) and 32 (HE, N_FFT=256) — "N_null is set to 8 and 32 for Wi-Fi HT and HE, respectively" (Q19). Implemented in official code as `sortSubC = argsort(mean(|feat|, axis=1)); keep sortSubC[N_null:]`.
- **5G (energy threshold):** "Only when |Y^s[k]| and |Y^s+1[k]| are higher than β, Y^s[k] is used", with **β = p99/10 per input** (Q18), because unused REs have random phase. (Official code implements the same effect via NaN-masking of non-allocated REs.)
- **Pilot/other-channel contamination:** impossible to remove blindly, but "the distribution of phase differences is still an intrinsic characteristic of PDSCH modulation" since other PHY channels are BPSK/QPSK (Section III-C).

### 2f) Classifier input and reported results (Sections III-D, IV-B, IV-C; official code for the layer detail)

- **Input preprocessing (III-D):** (1) phase = Δ∠ mod π/2 for QAM classifiers (constellations are rotationally symmetric with π/2); the BPSK-vs-QPSK discriminator uses the **original (unmodded) phase** because BPSK/QPSK collapse under mod π/2 → Wi-Fi uses a **two-stage classifier** (Fig. 4a). (2) 2D histogram of amplitude |Y_f|/p99 and phase, **S_Q = 50** bins (QAM/5G), **S_P = 15** bins (PSK stage) (Table VI); bin rule Eq. (13): `u/S ≤ |Y_f|/p99 ≤ (u+1)/S` and `v/S ≤ Δ∠/φ ≤ (v+1)/S` with φ = π/2 (mod) or 2π (original). Outliers above p99 are dropped. (3) Sum normalization Eq. (14): `Z′(u,v) = Z(u,v)/𝒵` with 𝒵 = number of valid features.
- **Histogram advantages (III-D):** length-invariance (handles 5G's unknown-used-RE counts) and CFO-error tolerance (y-axis shift).
- **CNN (Fig. 5 + official code, §4):** 2×2 kernels everywhere; QAM/5G net: Conv(1→2)→Pool→Conv(2→4)→Pool→Conv(4→8)→Pool→Conv(8→16) → FC 1200 → FC 84 → FC N_c (50×50 input → 16×4×4=256 flatten); PSK net: same convs with only two pools and S=15 input → flatten 16 → 1200 → 84 → 3. **N_c = 3 (PSK stage), 4 (Wi-Fi QAM), 5 (5G)**: "For the classifier to identify BPSK and QPSK, the third Maxpool layer is not used, S = S_P, and N_c = 3. The classifier for 5G and for identifying the QAM types for Wi-Fi 6 use N_c = 5, 4, respectively." No BatchNorm; Xavier init on Linear layers.
- **Training (Table V):** batch 32, epochs 200, lr 5·10⁻⁵, cross-entropy loss.
- **Data per instance (IV-B):** Wi-Fi: skip first 2000 samples (preamble), then "an IQ sequence whose length corresponds to 40+2 or 10+2 OFDM symbols … starting with a random sample" (the +2 covers unknown start residue and the extra symbol needed for differences); 5G: 14 OFDM symbols per input; feature-value-input comparison used 2240 (Wi-Fi) / 7900 (5G) samples.
- **Results (IV-C):** SCS/CP estimation 99% (l = 2, 4); symbol-start < 60% exact but 99% within N_CP/4; long-CP localization > 90% except 60 kHz SCS; modulation classification ≥ 97% on OTA whenever SNR meets each modulation's standard-required EVM-derived SNR (Table VII: e.g., Wi-Fi BPSK 5 dB, QPSK 10, 16QAM 16, 64QAM 22, 256QAM 30, 1024QAM 35 dB). AWGN synthetic ≥ same level; histogram input beats raw-feature input, increasingly so for HE/5G.
- **On "OFDM vs single-carrier discrimination":** P2 does **not** address it — P2 classifies modulation order *within* OFDM (its System Objective fixes OFDM inputs; single-carrier-vs-OFDM/SCS joint estimation is the related work [16] cited in P2's Introduction). Recorded so later stages do not attribute a claim P2 never makes.

---

## 3) CNN parameter-count verification (P1 Section 4.3 claim: 2,454,980 trainable parameters)

### 3a) Arithmetic for the 4-class model (P1's reported configuration)

| Layer | Weights | Bias | BN (γ+β) | Subtotal |
|---|---|---|---|---|
| Conv1 3×3, 1→32 | 32·(3·3·1) = 288 | 32 | 32·2 = 64 | **320 + 64 = 384** |
| Conv2 3×3, 32→64 | 64·(3·3·32) = 18,432 | 64 | 64·2 = 128 | **18,496 + 128 = 18,624** |
| Conv3 3×3, 64→128 | 128·(3·3·64) = 73,728 | 128 | 128·2 = 256 | **73,856 + 256 = 74,112** |
| FC1 4608→512 | 4608·512 = 2,359,296 | 512 | — | **2,359,808** |
| FC2 512→4 | 512·4 = 2,048 | 4 | — | **2,052** |
| **Total** | | | | **384 + 18,624 + 74,112 + 2,359,808 + 2,052 = 2,454,980** |

**VERIFIED: the table in §1d yields exactly 2,454,980 trainable parameters — the paper's claim (Q6) is reproduced to the digit.** This also confirms the paper's count includes conv biases and BN affine parameters (weights-only would give 2,454,532; no-bias/no-BN variants cannot reach the claimed number).

Spatial check: 50 →(pool)→ 25 →(pool)→ 12 (floor of 12.5) →(pool)→ 6; flatten 128·6·6 = **4,608** ✓ (matches Q5's "4608 elements").

Memory check: 2,454,980 × 4 B (float32) = 9,819,920 B = **9.37 MiB** ✓ (paper: "approximately 9.37 MB").

MAC check (forward pass): Conv1 32·9·50·50 = 720,000; Conv2 64·288·25·25 = 11,520,000; Conv3 128·576·12·12 = 10,616,832; FC1 2,359,296; FC2 2,048 → **total 25,218,176 ≈ 25.22 M MACs** vs the paper's "25.77 million Multiply–Accumulate operations". The parameter and memory claims match exactly; the MAC claim is ~2% higher than a standard count (possibly includes BN/elementwise ops). Non-blocking discrepancy — noted, not contract-level.

### 3b) Arithmetic for the 3-class variant (not reported by P1; computed for completeness)

FC2 becomes 512→3: weights 1,536 + bias 3 = 1,539. Total = 384 + 18,624 + 74,112 + 2,359,808 + 1,539 = **2,454,467** parameters. (For any other n_classes: total = 2,452,928 + 512·n + n.)

### 3c) P2's CNNs (from official code, for contrast)

- PskNet (S=15, N_c=3): convs 10 + 36 + 72 + 272 = 390; FC 16→1200 (20,400) → 1200→84 (100,884) → 84→3 (255) → **≈ 122,129 params**.
- QamNet (S=50, N_c=4): convs 390; FC 256→1200 (308,400) → 100,884 → 84→4 (340) → **≈ 410,014 params**.
- P1's CNN (2.45 M) is ~6× larger than P2's QAM net — P1 is a deliberately heavier, VGG-style redesign of the same input representation.

---

## 4) Official code / weights audit

Searches performed: MDPI page + full text (for GitHub/URLs), arXiv full text, HF user profile (Raluca13), GitHub repository/code/user searches ("Amp-Phase-2D-Histogram", "Nelega OFDM", "user:byungjunkim12", "getModFeatWifi", "2403.19292", "OFDM modulation classification"), PyPI (a package citing P1), paperswithcode API for both titles, author websites (byungjunkim12.github.io, noiselab.ucsd.edu).

| # | Artifact | URL | License | Contents | Matches | Classification |
|---|---|---|---|---|---|---|
| A | **HF dataset** `Raluca13/Amp-Phase-2D-Histogram-Dataset` | https://huggingface.co/datasets/Raluca13/Amp-Phase-2D-Histogram-Dataset | README states **CC BY-NC 4.0** (no HF license tag; no DOI yet — "DOI will be available soon") | 4 ZIPs of MATLAB `.mat` 50×50 histograms (D1/D3/D5/OTA) + README; **no code, no model weights** | P1 ref [6] verbatim; created 2026-04-03 = P1's "accessed on 3 April 2026"; zip counts match P1's D1/D3/D5/Table 3 sizes **exactly** (§5) | **OFFICIAL** (P1 authors) |
| B | **GitHub repo** `byungjunkim12/modulation_classification_wifi6_5G` | https://github.com/byungjunkim12/modulation_classification_wifi6_5G | **No LICENSE file, no README** (default all-rights-reserved; publicly viewable) | MATLAB + Python feature-extraction code: `OFDMparam.py` (CAF SCS/CP estimation), `CFOcorr.py` (CP-phase CFO), `getModFeat.py` (phase-difference features, null removal, mod π/2), `utilitiesDL.py` (CNN models, datasets, histogram `imgData`), notebooks `tbWifi.ipynb` / `tbNR.ipynb` / `loadRealWifi.ipynb` / `loadRealNR.ipynb`, `matlab/{CFO_corr,OFDM_param,modul_feat,utilities}`. Repo figures `feat_16qam.png`, `imgFeat_16qam.png` are P2's Fig. 6. **No trained weights, no datasets.** | P2 exactly (all Section III mechanics reproduced in code, §2); author = Byungjun Kim (personal site links this GitHub; UCSD PhD → Rutgers); commits Mar 2023 – 13 Apr 2024 (P2 arXiv: 28 Mar 2024) | **OFFICIAL** (P2 first author) |
| C | HF user `Raluca13` profile | https://huggingface.co/Raluca13 | — | full name "Nelega"; 1 dataset, **0 models, 0 spaces, 0 papers** | P1 first author Raluca Nelega | confirms **no released weights** for P1 |
| D | PyPI `rf-modulation-classifier` 0.1.1 | https://pypi.org/project/rf-modulation-classifier/ | MIT | 5-class (BPSK/QPSK/8PSK/AM-DSB/WBFM) classical-feature MLP on RadioML 2016.10a; **cites P1 only as a reference [5]** | unrelated method/dataset | **THIRD-PARTY** (Nishad Suresh), *not* a reproduction of P1 |
| E | paperswithcode | paperswithcode.com API + web search | — | **no entry found for either title** (API returned empty results; searches surfaced only the publisher pages) | — | **NONE** |
| F | GitHub code searches (`getModFeatWifi`, `2403.19292`, `Amp-Phase-2D-Histogram`) | github.com | — | all hits are artifact B itself or unrelated arXiv-ID scrapes | — | **no THIRD-PARTY reproduction found** for either paper |
| G | P2 earlier versions (Globecom 2022, ICASSP-wkshp 2023, MSWiM 2023 by same authors) | listed at noiselab.ucsd.edu/pub.html (#81, #87, #94, #111) | — | no separate code repos found for these | — | context only |

**Binding conclusions for later stages:**
1. **No official training code exists for the Sensors (P1) CNN pipeline.** The released official artifact for P1 is the *dataset* only (HF, artifact A). Any P1 model we build is therefore a **"paper-description faithful rebuild"**, **not** an "official model run" — later reports must use exactly that framing.
2. **Official frontend code exists for P2** (artifact B) and can be used to cross-check our frontend implementation (CAF estimator, CP search, CFO, phase-difference feature, null removal, histogram binning). Note its two deviations from the P2 text: uses `nanmax` (not p99) for the amplitude axis scale and clamps (not drops) the top bin; uses `inputLen = 2` (not l = 8) in the CAF estimator.
3. **No official model weights exist for either paper** (HF: 0 models; repo: no `.pt/.pth`).

---

## 5) Hugging Face dataset structure + minimal smoke plan

### 5.1 Network reachability (recorded per instructions)

- `huggingface.co` / `datasets-server.huggingface.co`: **unreachable from this environment** (DNS poisoned to a Facebook IP block; curl HTTP 000; webfetch timeout; retried once with alternate URL forms). `datasets-server` `/info` and `/first-rows` (DEFAULT_CONFIG/TRAIN) could **not** be consulted; per instructions, the fallback was the file-tree + ranged-ZIP inspection below. If HF becomes reachable later, re-run `/info` + `/first-rows` to confirm the same conclusions.
- **`hf-mirror.com` (Hub-API mirror) is reachable** and was used for: dataset metadata API, tree API, README download, and ranged reads of the LFS-backed ZIPs. Total bytes fetched ≈ 2.9 MB (README 3 KB; ZIP central directories 2.68 MB + 0.28 MB + 2×0.25 MB; two ~2.5 KB sample entries). **The 6.42 GB dataset was NOT downloaded.**

### 5.2 Dataset metadata (Hub API via mirror, 2026-11-21)

`id: Raluca13/Amp-Phase-2D-Histogram-Dataset` · `sha: 6be93ded3620aa0c1dc6427b68db28cf9c465c37` · created 2026-04-03T08:45:54Z (last modified same day) · public, not gated · `usedStorage: 6,422,313,640` bytes (**6.42 GB**) · downloads 30, likes 1 · description: "2D amplitude-phase histograms derived from raw IEEE 802.11 (Wi-Fi) IQ signal captures … Automatic Modulation Classification (AMC) under realistic and challenging indoor wireless propagation environments."

### 5.3 File listing (tree API)

| File | Size (bytes) | ≈ | Content |
|---|---|---|---|
| `Histograms_with_sync_err.zip` | 5,082,527,980 | 5.08 GB | D3 (large-scale sync-impaired synthetic, 1,947,117 histograms) |
| `Histograms_no_sync_error.zip` | 1,183,735,143 | 1.18 GB | D1 (perfectly synchronized synthetic, 487,344 histograms) |
| `Histograms_OTA.zip` | 107,492,493 | 107 MB | OTA testing set (13 positions, 41,844 histograms) |
| **`Histograms_bladeRF.zip`** | **48,558,024** | **48.6 MB** | **D5 conducted hardware set (18,591 histograms) — smallest usable subset** |
| `README.md` | 3,115 | 3 KB | structure/naming/license doc |
| `.gitattributes` | 2,504 | — | LFS pointer bookkeeping |

### 5.4 README-documented structure (fetched in full)

- Directory format: `[Channel_Model]/[SNR_dB]/[Filename].mat` — filterable by channel model / physical scenario and SNR level.
- Filename pattern: `H_{Format}_{Channel}_{Modulation}_{SNR}_{SampleID}.mat`; `{Format}` ∈ {`HT`, `NHT`}.
- **Modulation label = bits per symbol:** `1`=BPSK, `2`=QPSK, `4`=16-QAM, `6`=64-QAM.
- All samples are MATLAB `.mat` (v5/MAT-file, deflate-compressed elements). License statement: cite the associated publication, attribute under **CC BY-NC 4.0**; DOI pending.

### 5.5 ZIP central-directory audit (ranged reads only — no bulk download)

| ZIP | EOCD/ZIP64 | Total entries | `.mat` files | Dir entries | Paper cross-check |
|---|---|---|---|---|---|
| bladeRF | EOCD | 19,264 | **18,591** | 673 | P1 D5 = 18,591 ✓ **exact** |
| OTA | EOCD | 42,054 | 41,844 | 210 | P1 Table 3 sum = 41,844 ✓ **exact** |
| no_sync_error | ZIP64 | 487,429 | 487,344 | 85 | P1 D1 = 487,344 ✓ **exact** |
| with_sync_err | ZIP64 | 1,947,454 | 1,947,117 | 337 | P1 D3 = 1,947,117 ✓ **exact** |

All four zips' `.mat` counts reproduce the paper's dataset sizes to the histogram. This is the strongest available evidence that artifact A is P1's official data release.

**Directory layouts observed (from CD listings):**
- `Histograms_bladeRF/Histograms_{A|B|C|D|E|F|I}_blade{1-1|1-2}[_4p|_baterie|_baterie_4p]/SNR_{1|3|…|29}/H_NHT_{ch}_{mod}_{snr}_{id}.mat` — 42 config dirs (7 channels × 6 hardware variants), 15 SNR bins labeled 1…29 (odd), all files **NHT** in this zip; mod counts 4,675/4,714/4,673/4,529 (labels 1/2/4/6 — balanced); uncompressed `.mat` size ≈ 1.5–2.7 KB each.
- `Histograms_no_sync_error/Histograms_A_ch/SNR_0/H_HT_A_1_0_*.mat` (and with_sync_err same layout) — synthetic zips are **HT** (in sampled channel A), SNR bins labeled `SNR_0, SNR_3, …` (matching P1's 0–36 dB / 3 dB); exactly **1562 BPSK** files observed for (channel A, SNR_0) — matching P1's "1562 histograms for each combination of MCS, SNR, and channel model" (Q4) ✓.
- `Histograms_OTA/Corridor pos.B/SNR_{1|11|13…}/…` — position-named dirs (e.g., "Corridor pos.B") with odd-labeled SNR bins.
- Unexplained naming items (→ E7, E13): odd SNR labels for conducted/OTA vs paper's "0 to 30 dB in 2 dB steps" (interpret as 2-dB bin centers/edges); hardware-variant suffixes (`_4p`, `_baterie`) undocumented; bladeRF zip is NHT-only while synthetic zips sampled are HT-only (P1 says "both non-HT and HT" overall).

### 5.6 Sample value inspection (two `.mat` files extracted via ~2.5 KB ranged reads)

Files: `H_NHT_A_1_1_1.mat` (BPSK, bladeRF, SNR_1) and `H_NHT_A_6_19_2.mat` (64QAM, SNR_19).

- MATLAB v5 MAT-file header: "MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Tue Dec 9 11:01:14 2025".
- Single variable **`Histogram`**, dtype **float64**, shape **(50, 50)**; values ≥ 0.
- **`sum(Histogram) == 1.000000` exactly (sum-normalized, INFOCOM eq. (14) style)**; max bin ≈ 0.0022–0.0030 → the released histograms are **NOT** per-instance max-normalized, contrary to the *training-input* description in P1 §4.2 — on-load re-normalization is required to match the paper's training input (→ E3).
- Nonzero bins: 1,838/2,500 (BPSK) and 1,945/2,500 (64QAM); BPSK sample concentrates in few rows (phase axis) at low SNR with a CFO-shifted peak; 64QAM at 19 dB spreads across the amplitude axis and concentrates in fewer phase rows — consistent with the expected class structure. Axis orientation (rows = phase?) to be pinned by a synthetic zero-CFO BPSK/QPSK sample at smoke time (→ E6).

### 5.7 Minimal smoke plan (no 6.42 GB download)

1. **Ultra-light structural smoke (already demonstrated in this audit):** ranged-fetch a handful of `.mat` entries (≈2.5 KB each) via `hf-mirror.com/.../resolve/main/Histograms_bladeRF.zip` with `Range:` headers; load with `scipy.io.loadmat`; assert variable name `Histogram`, dtype float64, shape (50, 50), sum ≈ 1.0; parse the filename tokens into (format, channel, modulation label ∈ {1,2,4,6}, SNR, sample ID).
2. **Smallest usable subset:** download **`Histograms_bladeRF.zip` (48.6 MB)** through the mirror → 18,591 histograms (D5, balanced 4-class, 15 SNR bins, 42 hardware/channel configs). This is the minimum real-data smoke corpus. (Second-cheapest real subset: `Histograms_OTA.zip`, 107 MB, 41,844 histograms, position-labeled.)
3. **Loader checks against the two extracted samples** (§5.6) as fixed regression fixtures; label semantics = bits/symbol.
4. **Fallback if the mirror also fails:** synthetic-only smoke — generate 8-symbol 802.11 fields per §1, build histograms per the recipe, train the §1d CNN a few steps, and compare synthetic-histogram marginals to the paper's Fig. 1 description; mark real-data validation as blocked with the network failure recorded.

---

## 6) Engineering-assumptions list (paper-unspecified details; fixed choices are binding)

**From P1 unless noted.** Each item: ambiguity → **fixed choice** (documented; change requires a later stage to re-negotiate this contract).

- **E1 — Scope of "consecutive symbols".** P1 never says whether phase differences cross the 16 group boundaries. Groups are independent channel realizations (Q8), and D5 bursts deliver 8 payload symbols (Q6), so cross-group differences are physically undefined. **Choice:** differences computed **within each 8-symbol group only** (7 per group; 16×7 = 112 per subcarrier per histogram).
- **E2 — Phase wrap and bin index.** P1 says only "normalized by 2π" over a 1×1 grid; wrap convention and edge binning unspecified. P2's official code wraps raw differences to **(−π, π]** and bins over that span. **Choice:** wrap Δ∠ to (−π, π]; axis = Δ∠/2π ∈ (−0.5, 0.5]; 50 equal bins; `ceil`-style bin index clamped to [0, 49] (following P2's `imgData`), i.e. effectively 50 bins covering the 1-unit span. (Alternative [0, 2π) wrap changes only an axis shift; record which one the first real sample audit supports — see E6.)
- **E3 — Histogram normalization on load.** Released `.mat` histograms sum to 1.0 (verified, §5.6); P1 §4.2 trains on per-instance **max**-normalized histograms. **Choice:** at load time divide by the max bin value (H/ max H) to reproduce P1's training input; keep the raw sum-normalized artifact for provenance.
- **E4 — Pilot handling in "52 data subcarriers".** 20 MHz legacy/HT grid = 48 data + 4 pilots + 12 nulls. P1 says "52 data subcarriers" and "using only the data subcarriers" without naming pilots. **Choice:** use the **52 occupied non-null bins (48 data + 4 pilots)**, matching the count 52; note the alternative (48, pilots excluded) — pilots add a class-independent BPSK-like phase-difference peak; if smoke tests show the BPSK class contaminated by a pilot-consistent peak for all classes, revisit.
- **E5 — P99 population.** "Amplitude outliers exceeding the 99th percentile were discarded" — over what population? **Choice:** per 128-symbol block, over all retained data-subcarrier symbol amplitudes (the block's (subcarrier × symbol) amplitude matrix, nulls already excluded).
- **E6 — .mat axis orientation.** README gives no orientation. P1 §2: horizontal = amplitude, vertical = phase difference; MATLAB row convention ⇒ rows = phase (vertical), columns = amplitude. **Choice:** rows = phase-difference bins, columns = amplitude bins; pin definitively at smoke time with a synthetic zero-CFO BPSK histogram (expect two row clusters at Δ∠ = 0 and π, i.e., rows 0 and 25 under E2's (−π, π] axis with 50 bins) compared against a real BPSK sample's row-marginal.
- **E7 — SNR-bin label semantics in the released zips.** Conducted/OTA dirs use odd labels `SNR_1 … SNR_29` although the paper says 0–30 dB in 2-dB steps; synthetic dirs use `SNR_0, SNR_3, …`. **Choice:** interpret odd labels as 2-dB bin centers (bin k ⇒ [k−1, k+1) dB); synthetic labels as exact dB values (0–36, 3-dB steps, matching P1 §3.1.1).
- **E8 — 80/20 split method.** Random vs stratified unspecified. **Choice:** stratified by (modulation, SNR, channel) so validation coverage is uniform; no cross-domain leakage concern for within-domain training runs (OTA is a disjoint collection anyway).
- **E9 — Epochs and LR schedule.** Only early stopping (patience 10, on validation loss) is specified; no max-epoch or scheduler. **Choice:** max epochs generous (e.g., 200 as in P2) with P1's early stopping; constant LR (no scheduler).
- **E10 — Conv padding.** "Same" padding is implied (only pooling reduces 50→6). **Choice:** 3×3 convs with padding=1.
- **E11 — OTA "receiver-estimated SNR from the LTF".** Estimator unspecified. **Choice:** LTF-based per-capture noise/power ratio implemented once, documented; only used for OTA SNR binning/analysis, not for training.
- **E12 — D3 size arithmetic.** P1: D3 = 1,947,117 histograms "including both impairment-free and synchronization-impaired data"; but 4 × 487,344 = 1,949,376 (−2,259 unexplained; likely rejections during impaired re-generation). No action needed; recorded so later reconciliation with the ZIP (1,947,117 exact) is not misread as an error.
- **E13 — Format inventory across zips.** P1 claims "non-HT and HT" datasets; bladeRF zip verified 100% NHT (18,591), sampled synthetic zip entries 100% HT. **Choice:** treat format as a filename token (`HT`/`NHT`) carried in the loader; do not assume per-format balance until a full inventory (deferred; requires the full 270 MB CD of with_sync_err or a 48.6 MB/1.18 GB download).
- **E14 — Hardware-capture synchronization algorithms (D5/OTA only).** P1 names the stages (symbol timing; coarse+fine CFO) but not the algorithms. **Choice for any hardware-capture path:** adopt P2's pipeline (§2a–2c) as the documented stand-in; for the synthetic rebuild (D1-style) use perfect demodulation as P1 does.
- **E15 — CAF averaging length l (P2).** P2 text: l = 8; P2 evaluation: l = 2, 4 achieve 99%; official code: `inputLen = 2`. **Choice:** l = 2 (official code + best reported accuracy); note the text's 8 as a discrepancy.
- **E16 — P2 amplitude-axis normalization.** P2 text: |Y|/p99 with >p99 dropped; official code `imgData`: scale = `nanmax(|Y|)`, top bin clamped. P1's own text: "maximum value within the 99% interval". **Choice for our rebuild (P1-faithful):** P1's wording — drop >P99, scale by max of retained values (≈ P99); if we cross-check against P2's code output, expect this known difference.
- **E17 — Class imbalance in D5.** Verified mod counts (1:4,675; 2:4,714; 4:4,673; 6:4,529) are near-balanced; no re-weighting needed. **Choice:** plain CE loss, no class weights (matches P1's silence).

---

## 7) Exact quotes with section numbers (fidelity evidence)

### P1 — Sensors 2026, 26(10):2945 (from the CC BY full text, Europe PMC PMC13211014)

- **Q1 (§3.1.1, synthetic generation):** "To generate the 2D amplitude-phase histograms, 20 MHz IEEE 802.11 signals with 52 data subcarriers were simulated in MATLAB R2025b [20]. First, 8-symbol data fields were generated for BPSK, QPSK, 16QAM, and 64QAM modulations (corresponding to modulation and coding schemes (MCSs) 0, 1, 3, and 5). The resulting signals were passed through indoor TGn multipath fading models (types A–F) [21], compensated for filter delays, and demodulated, with the 16-sample cyclic prefix removed."
- **Q2 (§3.1.1, noise + outlier discard):** "To emulate realistic reception conditions, measured AWGN was added over an SNR range from 0 to 36 dB in 3 dB increments. The noisy QAM symbols were then grouped into blocks of 128 OFDM symbols, while amplitude outliers exceeding the 99th percentile were discarded."
- **Q3 (§3.1.1, axes + binning):** "Using only the data subcarriers, the amplitudes were normalized by the maximum value within the 99% interval, while the phase difference between consecutive symbols was normalized by 2π. These amplitude-phase pairs were mapped onto 2D histograms over a 1 × 1 grid with a resolution of 1/50 per axis, as illustrated in Figure 1."
- **Q4 (§3.1.1, dataset size):** "This process generates 1562 histograms for each combination of MCS, SNR, and channel model, resulting in a total of 487,344 histograms. The resulting dataset represents the perfectly synchronized dataset D1."
- **Q5 (§4.1, architecture):** "Drawing inspiration from the VGG [23] architectural paradigm, the proposed CNN is designed to process an input tensor of size 1 × 50 × 50, representing the 2D amplitude-phase histogram grid. … As the network deepens, the number of convolutional filters progressively doubles (from 32 to 64, and finally to 128) … Each convolution is integrated with Batch Normalization to stabilize training and a ReLU activation function … Systematic spatial downsampling is achieved via 2 × 2 Max-Pooling layers at the end of each block, reducing the spatial resolution from 50 × 50 down to 6 × 6. The architecture then transitions into a fully connected classification head that flattens the resulting 128 × 6 × 6 feature map (4608 elements) into a 512-dimensional hidden layer. To ensure robust generalization, a dropout rate of 0.5 is applied prior to the final linear layer, which maps the high-level abstractions to the 4 target modulation schemes."
- **Q6 (§4.3, complexity):** "Regarding computational complexity, the network contains 2,454,980 trainable parameters with a total memory footprint of approximately 9.37 MB, requiring 25.77 million Multiply–Accumulate operations per forward pass. Baseline latency testing on a standard commercial workstation equipped with an NVIDIA GeForce RTX 5070 GPU yielded an average execution time of 0.374 ms per sample…"
- **Q7 (§4.2, training):** "The datasets are divided into training and validation sets using an 80/20 split. To emphasize the structural topology and geometric signatures of the modulation schemes, rather than absolute sample intensities, each 2D histogram is independently normalized to the [0, 1] interval via per-instance maximum normalization. Model training is performed using the Adaptive Moment Estimation (Adam) optimizer and Cross-Entropy loss function, with a batch size of 1024, a learning rate of 10−3 and a weight decay of 10−4. An early stopping mechanism with a 10-epoch patience window is used to stop training once validation loss stabilizes, ensuring the selection of the best-performing weights. The training is performed using the PyTorch library [24] … on an NVIDIA A100-PCIE GPU [25] with 40 GB of RAM. To ensure statistical reliability and account for random weight initialization, the model is trained 10 times independently for each experimental case."
- **Q8 (§5.2, 16 × 8 symbol grouping):** "…the 128-symbol amplitude-phase histograms are constructed by aggregating 16 independent groups of 8 OFDM symbols. In a highly mobile scenario—such as vehicular speeds of 100 km/h—the channel coherence time is approximately 2 ms in the 2.4 GHz band and 1 ms in the 5.8 GHz band. Because the duration of an 8-symbol group in a 20 MHz bandwidth is only 32 μs, it is much smaller than the coherence time. Consequently, the channel transfer function remains effectively constant within each 8-symbol block, but varies across the 16 blocks that make up a single histogram."
- **Q9 (§2, representation properties inherited from P2):** "…the method first extracts a feature sequence from the normalized amplitude and phase differences between consecutive OFDM symbols. The statistical distribution of these features is then represented as a 2D histogram, where the horizontal axis corresponds to normalized amplitude, and the vertical axis represents phase difference." — with advantages: "**Robustness to symbol-level synchronization errors:** By relying on phase differences between adjacent symbols, identical phase drifts mathematically cancel out. **Invariance to sequence length:** The fixed dimensions of the histogram allow the network to process signals of any arbitrary duration. **Robustness to carrier frequency offset (CFO):** CFO estimation errors simply manifest as a uniform vertical shift along the histogram's phase axis."
- **Q10 (§5.2, results/QAM confusion):** "The confusion matrix highlights high classification fidelity with near-perfect performance (100%) for BPSK and QPSK modulations. The primary source of error is an asymmetric confusion between the QAM schemes, where 19.0% of 16-QAM signals are misclassified as 64-QAM. This suggests that real-world distortions cause the 16-QAM constellation to spread into the denser 64-QAM decision space. Nevertheless, the model maintains a robust 92.6% accuracy for 64-QAM with low statistical variance (±1.5%)…" / "Specifically, 16-QAM exhibits a sharp degradation below 15 dB…" / "Both high-order modulations eventually converge to 100% accuracy above 20 dB…" / "In high-SNR environments such as corridor position G (24.7 dB) and lab room position B3 (16.1 dB), the model maintains near-perfect accuracy (up to 100.0%)…" / "At corridor position B (5.0 dB), the accuracy drops to 76.0%, which is the lowest observed value."
- **Q11 (§5.1, scale conclusion):** "Model D3, trained on the large-scale synchronization-impaired dataset, attains the highest mean accuracy (93.36%) along with a relatively low standard deviation (±0.42%)." / "…the small-scale synchronization-impaired dataset (D4) and the small-scale conducted hardware dataset one (D5). Their comparable mean accuracies (89.89% vs. 89.38%)…"
- **Q12 (§5.1, low SNR):** "…the conducted hardware dataset (D5) declines more sharply, falling below 50% accuracy at the lowest SNR levels."
- **Q13 (§3.1.3, D5 capture pipeline):** "Once isolated and validated, the data frames underwent the hardware synchronization pipeline. This included symbol timing estimation, followed by both coarse and fine carrier frequency offset estimation and correction. After stripping the frame header, 8 OFDM payload symbols were retained and fed directly into the previously established processing pipeline. They were demodulated, organized into groups of 128 OFDM symbols, normalized, and mapped into two-dimensional histograms just as in the simulated datasets. This batch-processing workflow generated a final dataset (D5) of 18,591 histograms."
- **Q14 (§3.2, OTA labels):** "…the final testing dataset is categorized exclusively by the intended MCS and the true receiver-estimated SNR, calculated from the IEEE 802.11 Long Training Field (LTF)."
- **Q15 (Data Availability Statement + ref [6]):** "The generated 2D amplitude-phase histogram datasets supporting the findings of this study are openly available on Hugging Face at [6]." — ref [6]: "Hugging Face Raluca13/Amp-Phase-2D-Histogram-Dataset. Available online: https://huggingface.co/datasets/Raluca13/Amp-Phase-2D-Histogram-Dataset (accessed on 3 April 2026)." — ref [5]: "Kim B., Mecklenbräuker C., Gerstoft P. Deep Learning-Based Modulation Classification of Practical OFDM Signals for Spectrum Sensing. Proceedings of the IEEE INFOCOM 2024—IEEE Conference on Computer Communications, Vancouver, BC, Canada, 20–23 May 2024."

### P2 — INFOCOM 2024 / arXiv:2403.19292v1

- **Q16 (§III-B, correlation + peak search + start):** "Utilizing the knowledge of N_CP and N_FFT, the CP position is determined through autocorrelation analysis, R_yy(m, N_FFT) = (1/N_CP) Σ_{i=0}^{N_CP−1} y[m+i] y*[m+i+N_FFT] … The position of CP is indicated by the peaks in |R_yy(m, N_FFT)| … To locate a peak, we search for a sample whose amplitude is larger than both of its neighboring samples while ensuring that the minimum distance between two adjacent peaks is 90% of the OFDM symbol duration (i.e., (256+64)×0.9 = 288-time samples for HE format with 3.2 μs CP), to avoid selecting undesired local peaks. … p = Median_i{mod(p′_i, N_FFT + N_CP)} … To reliably sample the sequences contained in a single OFDM symbol, we deploy the sequence {y[p + N_CP/2], …, y[p + N_CP/2 + N − 1]}. This sequence is entirely within a single OFDM symbol for estimation error of p below N_CP/2."
- **Q17 (§III-B, null exclusion):** "For Wi-Fi 6, the null subcarrier symbols are eliminated by discarding symbols with the N_null smallest average amplitudes."
- **Q18 (§III-C + §IV-B, 5G screening):** "Only when |Y^s[k]| and |Y^{s+1}[k]| are higher than β, Y^s[k] is used." / "β is set to |Y_f^s[k]|_p99/10 in each input."
- **Q19 (§IV-B, Wi-Fi capture handling):** "First, to avoid using the Wi-Fi preamble, we remove the first 2000 samples from each data. If the estimated T_IFFT corresponds to those of Wi-Fi 6, an IQ sequence whose length corresponds to 40+2 or 10+2 OFDM symbols is deployed to build Y_f^s[k], starting with a random sample. … N_null is set to 8 and 32 for Wi-Fi HT and HE, respectively."
- **Q20 (§III-D, histogram + CFO shift):** "A 2D histogram of the normalized amplitude of the features |Y_f^s[k]|/|Y_f^s[k]|_p99 … and the phases ∠Y_f^s[k]/2π, as an input for the classifier." (Eq. 13 bin rule; φ = π/2 if mod is used, else 2π.) "We normalize histogram value to be classifier input: Z′(u,v) = Z(u,v)/𝒵, where 𝒵 denotes the number of valid Y_f^s[k] in one data. To remove outliers, Y_f^s[k] whose amplitude is larger than |Y_f^s[k]|_p99 was not included in the histogram." / "…the effect of CFO estimation error caused by aliasing (12) is a movement along the y-axis of the histogram… The neural network can be trained to identify histogram movements along the y-axis as a single class."
- **Q21 (§IV-C1, estimation accuracy):** "Using l = 2, 4 achieves 99% accuracy for both Wi-Fi 6 formats and 5G and outperforms l = 1 as used in [24]." / "the estimation accuracy for identifying the starting index of an OFDM symbol falls below 60% for both Wi-Fi 6 formats and 5G. When the tolerance is relaxed to N_CP/4 time samples, the reported estimation accuracy increases to 99%."
- **Q22 (§IV-C2 + Abstract, headline):** "For every modulation with both Wi-Fi 6 formats and 5G, accuracy is at least 97%." / "achieves a minimum accuracy of 97% accuracy with OTA data when SNR is above the value required for data transmission."
- **Q23 (§III-A, CAF estimator setting):** "It differs from the estimator in [24], which corresponds to l = 1 in (3). The increase in the length of a sample sequence y[n+i] aims to make peaks more distinct. We set l = 8 corresponding to the shortest CP length in our scenario." (Note tension with Q21's l = 2, 4; official code uses 2 → E15.)
- **Q24 (§III-B, Eq. (11) CFO):** "the CP in each OFDM symbol is used for CFO estimation Δf_c, i.e., ∠(y[p + N_FFT + i] · y*[p + i]) = 2πΔf_c/Δf_SCS, where y[p + i] is in CP. We use i ∈ {⌊N_CP/4⌋, …, ⌈3N_CP/4⌉} … We determine CFO as the average of Δf_c (11) evaluated over multiple OFDM symbols. If the absolute value of the CFO is larger than Δf_SCS/2, the CFO cannot be accurately estimated due to aliasing."

### Histogram recipe (pseudocode contract — P1 §3.1.1 + §4.2 with binding choices E1–E5)

```text
for each instance:                                        # (MCS, channel A-F, SNR) — P1 §3.1.1
    generate 8-symbol 802.11 data field, MCS in {0,1,3,5} # BPSK/QPSK/16QAM/64QAM
    apply TGn fading; add AWGN (0..36 dB, 3 dB steps); demodulate; strip 16-sample CP
    keep 52 occupied non-null subcarriers -> |Y^s[k]|, angle(Y^s[k])   # E4
collect 16 such groups -> block of 128 symbols            # P1 §5.2 (Q8)
A = all |Y^s[k]| in block over data subcarriers           # E5 population
p99 = percentile(A, 99); A' = A[A <= p99]                 # Q2 outlier discard
amp = A' / max(A')                                        # Q3 amplitude axis in [0,1]
for each group g of 8 symbols:                            # E1: within-group only
    dphi = wrap_to_(-pi, pi](angle(Y^{s+1}) - angle(Y^s)) for s = 1..7
axis_v = dphi / (2*pi)                                    # Q3 phase axis, span 1 (E2)
H = zeros(50, 50)                                         # rows=phase (E6), cols=amplitude
u = clamp(ceil(amp*50) - 1, 0, 49); v = same rule for axis_v  # E2 bin rule
H[v, u] += 1  for every (amp, dphi) pair
H = H / max(H)                                            # P1 §4.2 per-instance max norm (E3;
                                                          #  released HF .mat stores H/sum(H) instead)
label = bits-per-symbol {1, 2, 4, 6}                      # HF README label semantics
```

---

## 8) Summary for downstream stages

1. **Representation contract:** §1a–§1c + the pseudocode in §7 (with E1–E6) — any deviation is a contract change.
2. **Model contract:** §1d table; 4-class params = 2,454,980 (verified exactly, §3a); 3-class = 2,454,467 (§3b). Training: §1e.
3. **Expected behavior:** BPSK/QPSK ≈ 100%; 16QAM weakest (≈81% best case OTA) with 19.0% asymmetric leakage into 64QAM; 64QAM ≈ 92.6%; degradation below ~15 dB, convergence to ~100% above ~20 dB (§1f).
4. **Frontend contract (if blind sync is in scope):** P2 pipeline per §2 with official-code cross-checks (§4 artifact B), noting the three code-vs-text deltas (l=2 vs 8; nanmax-vs-p99; clamp-vs-drop).
5. **Framing:** the CNN pipeline is a **paper-description faithful rebuild** (no official P1 training code exists); the histogram *dataset* is **official** and byte-level-verified against the paper's sizes (§5.5); P2's **frontend code is official**.
6. **Smoke path:** ranged-`.mat` extraction demonstrated; smallest full subset = `Histograms_bladeRF.zip` (48.6 MB); HF itself is DNS-blocked here, `hf-mirror.com` works; synthetic-only fallback documented (§5.7).
