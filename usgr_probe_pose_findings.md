# USGR guidance model — evaluation findings on in-vivo Clarius A4C data

*Prepared 2026-06-09 by Jilei, for discussion with the USGR team.*

## Setup

We integrated the USGR `PoseTransformer` guidance head (frozen DINOv3
ConvNeXt-Base encoder + transformer head) into our cardiac ultrasound guidance
app and evaluated the two EMA checkpoints we have (`ema_9000.pth`,
`ema_21000.pth`, with `normalization_statistics.pth`) on in-vivo Clarius data:

- **"Green" set**: 431 frames from one recording, filtered by an image-only
  traffic-light A4C quality classifier (all confidently good A4C views).
  512×512 grayscale sector fans, plus a per-frame IMU quaternion CSV.
- **Baseline set**: the full 2,247-frame recording the green set was drawn from
  (mixed quality / off-target views).

Evaluation logic: for frames that already ARE a good A4C, the predicted
relative transform to target A4C should be ≈ identity (|T_rel| → 0, rotation
angle → 0), and should be clearly smaller than for baseline frames.

Inference matches `recording/inference_simulation.py::preprocess_image_for_model`
(grayscale /255, 3-channel repeat), conditional one-hot (class_length 12),
denormalize, SVD-project rotation to SO(3). No filtering in these measurements.

## Headline results (green A4C frames, target = A4C)

| Metric | 9k @384 | 21k @384 | 21k @512 | converged should be |
|---|---|---|---|---|
| \|T_rel\| green | 7.57 cm | 2.42 cm | 4.89 cm | → ~0, ≪ baseline |
| \|T_rel\| baseline | 6.98 cm | 2.43 cm | 4.91 cm | |
| R angle green | 56.9° | 47.5° | 27.8° | → ~0°, ≪ baseline |
| R angle baseline | 52.5° | 37.1° | 27.7° | |
| per-frame T std (cm) | [.06,.08,.14] | [.14,.32,.27] | [.17,.44,.48] | large (image-driven) |

Per-view "argmin" test: for each green A4C frame we predict against all 6
target views and ask which view gets the smallest output (a converged model
should say A4C):

| A4C mean rank (0=best, 5=worst) | 9k @384 | 21k @384 | 21k @512 |
|---|---|---|---|
| by rotation angle | 2.44 | 1.02 | **0.29** (argmin 71% of frames) |
| by \|T_rel\| | 5.00 | 0.97 | 2.92 |

## Findings

### 1. The trend is healthy — this looks like undertraining, not a bug

From 9k → 21k every metric moves the right way, and A4C climbs sharply toward
rank 0 in the argmin test. We initially suspected a view-index /
canonical-pose convention mismatch on our side; the monotone improvement with
training steps rules that out. At 21k the head still mostly emits a per-view
constant (tiny per-frame variance; green ≈ baseline), i.e., it barely uses the
image for in-distribution frames yet.

**Ask: a later EMA checkpoint (≥100k steps).** We'll re-run the same
diagnostics on each checkpoint you can share.

### 2. Inference resolution matters a lot — we now run 512

We had been inferring at 384 (the `PoseTransformer` default in `model.py`, and
what `recording/realtime_inference.py` uses). Your training entry point uses
`DinoEncoderConfig.image_size`, default **512**. Switching 384 → 512 on
ema_21000 cut the green-A4C rotation error 47.5° → **27.8°** and made A4C the
clear rotation argmin (rank 1.02 → 0.29). At 384 the translation output was
just the normalizer mean (|T| 2.42 cm ≈ |T_mean| 2.1 cm — regress-to-mean in
normalized space).

**Ask: confirm the training `image_size` (and any other preprocessing in your
config that differs from defaults).** Also note `realtime_inference.py` still
hardcodes 384 + vitl16 — worth updating upstream.

### 3. L-R chirality (your observation) — confirmed real, currently latent

You're right that training images are LV-on-left while ours are LV-on-right.
Your h-flip augmentation (`dataset/augmentation.py::_h_flip`) flips the image
AND right-multiplies the rotation by `F = diag(1,-1,-1)` (180° about probe x),
so a converged model should be flip-**equivariant**. We measured ema_21000's
response to mirroring the input (n=40 green frames, 512):

- angle(R_noflip, R_flip) = **4.9°** → the model currently *ignores* chirality
- angle(R_noflip, F·R_flip) = 175.7° → i.e., NOT yet equivariant
- mirroring changes the green-A4C rotation error by only 0.1° (27.8 → 27.7)

So chirality is **not** the source of today's 27.8° residual — but once the
head converges and becomes equivariant, feeding the wrong chirality will
inject up to ~180°-about-x error. We've added a `mirror_input` switch on our
side (off until a converged checkpoint).

**Ask: confirm the intended correction for LV-on-right input.** Mirror the
input frame only (output then lives in your canonical frame), or mirror input
and also apply `F` to the output transform? Equivalently: are the canonical
poses defined w.r.t. the LV-on-left image convention?

### 4. Classifier-free guidance: tested, doesn't help at this stage

Since training uses `proportion_unconditioned=0.2`, we swept CFG
(`se3_guidance`-style SLERP/linear extrapolation) at w ∈ {1, 1.5, 2, 3}.
w=1.5–2 slightly sharpens *relative* view discrimination (rotation argmin 68% →
75% A4C) but inflates the *absolute* transform (27.8° → 47.9°, 4.9 → 7.9 cm at
w=2), which is what our UI consumes. We run conditional-only (w=1), same as
your inference scripts. Curious whether you use/recommend CFG at convergence.

### 5. The degenerate-rotation edge case (your inference snippet) — measured

We checked your `det(R) <= 0` edge case on this data (ema_21000, 150 baseline
frames × 6 target views):

| | invalid rate | raw det(R): mean per view |
|---|---|---|
| @384 | 1/900 (0.1%) | A4C 0.05, A2C 0.40, PSAX_PAP 0.46, PSAX_MV 0.04, PLAX 0.14, SC4C 0.26 |
| @512 | 7/900 (0.8%) | A4C 0.22, A2C 0.39, PSAX_PAP 0.43, PSAX_MV **0.02**, PLAX 0.21, SC4C 0.39 |

So "pretty rare" confirmed — and the invalids are not random: they concentrate
where the per-view mean det sits at ~0 (PSAX_MV, 6/150 at 512). Two takeaways:

- **det(R_raw) is a free confidence / convergence signal.** A converged head
  should emit det → 1; at 21k it emits 0.02–0.46, i.e., heavily shrunken
  matrices from averaging. (A4C's det rose 0.05 → 0.22 going 384 → 512 —
  consistent with 512 being the trained resolution.) Might be worth logging
  per-view det during training.
- We adopted your fallback (degenerate ⇒ ESKF predict-only extrapolation;
  drop frame only if the filter has no state). One small note on your snippet:
  it feeds the raw, non-orthonormal matrix to the filter as the measurement —
  we SVD-project to SO(3) first, which the ESKF measurement model implicitly
  assumes; may be worth doing upstream too. (We also SVD-project before
  rendering, so visualization can't crash on our side.)

### 6. Open questions about the training setup

1. **Dataset type for these checkpoints**: simulated (`CanonicalViewDataset` /
   MUSiK) or in-vivo (`InVivoReferenceViewDataset`)? This matters twice:
   - If in-vivo: the view-class indices come from the alphanumeric sort of the
     reference labels — what were the labels, so we can verify our index map
     (we assume `0=A4C, 1=A2C, 2=PSAX_PAP, 3=PSAX_MV, 4=PLAX, 5=SC4C` from
     `CANONICAL_VIEW_LIST`)?
   - If simulated (MUSiK): our in-vivo Clarius frames are out-of-domain — do
     you expect them to work without in-vivo fine-tuning?
2. **Normalization stats**: are `normalization_statistics.pth` computed from
   the same run/dataset as these checkpoints? (T mean [2.1, 0.0, −0.04] cm,
   std [2.85, 5.68, 6.08] cm; the rotation-component mean reshapes to a
   near-degenerate matrix, so unconfident outputs need SO(3) projection — is
   that expected?)
3. **Training horizon**: total planned steps, and which checkpoint you'd
   consider "done" for rotation vs translation (translation is converging
   visibly slower for us: rank 2.92 vs 0.29 at 21k/512).

## What we're handing over

- The **green set** (431 frames + `frame.imu.csv` with per-frame IMU
  quaternions, timestamps in ns matching filenames) and the **full 2,247-frame
  source recording**.
- Our diagnostic scripts (standalone, only need torch/cv2/scipy + the model):
  - `scripts/diagnose_probe_pose.py` — green vs baseline |T_rel| / R angle /
    per-frame responsiveness; `--image-size`, `--flip` flags.
  - `scripts/perview_argmin_probe_pose.py` — per-view argmin / rank test.
  - `scripts/cfg_probe_pose.py` — CFG scale sweep.

If you can run `diagnose_probe_pose.py` against your latest checkpoint on this
data (or just send us the checkpoint), that's the fastest way to see whether
the remaining gap is purely training steps or an in-vivo domain gap.
