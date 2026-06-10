# Next session — USGR probe-pose convergence (A4C): waiting on later checkpoint

## Copy-paste prompt for next session

> Continue the USGR probe-pose convergence work in cardiac-ultrasound-guidance.
> The diagnosis is now settled: the model is **undertrained** (hypothesis 1
> confirmed, view-index mismatch ruled out) and CUSG was additionally running it
> at the **wrong resolution** (384 vs the 512 it was trained at — now fixed,
> predictor defaults to 512). The remaining work is gated on a **later
> checkpoint** (>21k steps; USGR trains 100k+). When one lands in
> `/Users/jileihao/data/vigil/__demo_weights/`, run experiments (B') and (A')
> below with the vigil conda env
> (`/Users/jileihao/miniconda3/envs/vigil/bin/python`, torch+MPS). Also confirm
> with Trevor: training dataset type (simulated `CanonicalViewDataset` vs
> `InVivoReferenceViewDataset` — the latter remaps view indices!), the exact
> `image_size` in the training config, and total planned steps.

---

## Status after the 2026-06-09 troubleshooting session

### Resolved

1. **View-index / convention mismatch (old hypothesis 3) — RULED OUT.**
   Per-view argmin test (`scripts/perview_argmin_probe_pose.py`): from ema_9000
   → ema_21000, A4C's mean rank among the 6 target views improved
   - by |T_rel|: rank 5.00 (dead last) → 0.97
   - by R angle: rank 2.44 → 1.02 (A4C becomes plurality argmin)
   The correct view climbing monotonically toward rank 0 with training steps is
   the undertraining signature, not an index bug. (Caveat: if training used
   `InVivoReferenceViewDataset`, indices come from alphanumeric label sort —
   still confirm with Trevor, but the trend says the mapping we use is right.)

2. **Resolution mismatch (old hypothesis 4) — CONFIRMED AND FIXED.**
   USGR training default is `image_size=512` (`DinoEncoderConfig`,
   `usgr/usgr/configs/models.py`); CUSG inferred at 384. At 512 on ema_21000
   (green A4C, n=80):
   - R angle: 47.5° → **27.8°**; argmin-by-R picks A4C 71% (vs 39%), rank 0.29
   - per-frame T std rises to [0.17, 0.44, 0.48] cm (more image-driven)
   - |T_rel| changes 2.42 → 4.89 cm (at 384 the head was emitting ~0 in
     normalized space — 2.42 cm ≈ the normalizer T-mean magnitude, pure
     regress-to-mean; at 512 it actually responds)
   **Fix applied**: `ProbePosePredictor` default `image_size` is now **512**
   (`src/inference/probe_pose/probe_pose_predictor.py`), and all diagnostic
   scripts default to 512. Note: upstream USGR's own
   `recording/realtime_inference.py` still constructs the model at 384 with a
   stale vitl16 encoder — don't trust it as a reference.

3. **L-R chirality mismatch (USGR dev's observation) — REAL BUG, but NOT the
   current residual; latent until convergence.**
   USGR training images are LV-on-left; CUSG frames are LV-on-right (mirror).
   The training h-flip augmentation (`usgr/usgr/dataset/augmentation.py::_h_flip`,
   prob 0.5) flips the image AND right-multiplies the source rotation by
   `F = diag(1,-1,-1)` (180° about probe x). So a *converged* model must be
   flip-**equivariant**: opposite chirality ⇒ prediction rotated ~180° about x.
   Measured on ema_21000 (green A4C, n=40, 512):
   - angle(R_noflip, R_flip) = **4.85°** ⇒ model is flip-INVARIANT (ignores chirality)
   - angle(R_noflip, F·R_flip) = **175.7°** ⇒ NOT equivariant (would be ~0 if it were)
   - |T_noflip − T_flip| = 0.8 cm; |T_noflip − F·T_flip| = 9.4 cm
   So mirroring the green frames changes the rotation error by 0.1° (27.8→27.7) —
   the residual is undertraining, not chirality. The invariance itself is the
   undertraining signature (regress-to-mean head ignores the image ⇒ ignores
   chirality). **Once the head keys on anatomy, feeding LV-on-right will inject
   up to ~180°-about-x error.** Scaffolding added: `ProbePosePredictor(...,
   mirror_input=False)` flips the frame to LV-on-left before inference (no-op
   today). **TODO at convergence**: enable it, and confirm with the USGR dev
   whether the canonical poses want mirror-input-ONLY or mirror-input + applying
   F=diag(1,-1,-1) to the output transform (to land back in CUSG's render
   frame). Test via `scripts/diagnose_probe_pose.py --flip` /
   `perview_argmin_probe_pose.py --flip`.

4. **Classifier-free guidance — TESTED, NOT ADOPTED.**
   The head is trained with conditioning dropout (`proportion_unconditioned=0.2`)
   so CFG is valid. Sweep (`scripts/cfg_probe_pose.py`, w ∈ {1, 1.5, 2, 3}, 512):
   w=1.5–2 sharpens *relative* view discrimination slightly (argmin-by-R 68→75%)
   but inflates *absolute* error (A4C R angle 27.8→47.9°, |T| 4.9→7.9 cm at w=2)
   and never fixes translation ranking. The UI consumes the absolute transform →
   keep conditional-only (w=1). Upstream USGR inference also runs w=1
   (`se3_guidance` in `usgr/usgr/guidance/sampling.py` is unused there).

### Still open (gated on a later checkpoint)

- **Translation doesn't discriminate**: green ≈ baseline |T_rel|, A4C rank ~2.9
  by |T| even at 512 (argmin is PSAX_PAP/PSAX_MV). Rotation converges faster
  than translation; T needs more training steps.
- **Green vs baseline separation**: still none at 21k (both ~same |T| and R).
- 21k of a planned 100k+ steps (`max_iterations=100_000` default in
  `usgr/usgr/configs/guidance.py`).

## Next experiments (when a new checkpoint arrives)

- **(B') Checkpoint sweep at 512** (scripts now default to 512):
  ```bash
  PY=/Users/jileihao/miniconda3/envs/vigil/bin/python
  cd /Users/jileihao/dev/vigil-developer/cardiac-ultrasound-guidance
  $PY scripts/diagnose_probe_pose.py --checkpoint .../ema_NNNNN.pth --view A4C
  ```
  Convergence signals: green |T_rel| → 0 **and below baseline**; green R angle
  → 0 and below baseline; per-frame T std grows.
- **(A') Per-view argmin** (rank trend across checkpoints):
  ```bash
  $PY scripts/perview_argmin_probe_pose.py --checkpoint .../ema_NNNNN.pth
  ```
  Expect A4C rank → 0 by both metrics. By R it's already 0.29 at 21k/512.
- **(C) Confirm with Trevor (USGR side)** — still worth one message:
  1. Dataset type: simulated (`CanonicalViewDataset` / MUSiK) or in-vivo
     (`InVivoReferenceViewDataset`)? If in-vivo: what were the reference labels
     (their alphanumeric sort defines the view-index mapping)?
  2. Training `image_size` (we infer 512 from config default + empirics).
  3. Planned total steps / when a ≥100k EMA checkpoint will be available.
  4. **L-R chirality**: canonical poses assume LV-on-left (their image
     convention). CUSG is LV-on-right. Confirm the fix: mirror the input frame
     only, or mirror + apply F=diag(1,-1,-1) to the output? (See finding 3.)
- **(D) Ground truth** (unchanged, harder): needs a reference A4C pose in the
  IMU frame; only IMU orientation available for the green set.

## Key facts & assets (carried forward)

- **Env**: `/Users/jileihao/miniconda3/envs/vigil/bin/python` (torch 2.9, MPS, +einops, scipy).
- **Checkpoints**: `/Users/jileihao/data/vigil/__demo_weights/ema_9000.pth`,
  `ema_21000.pth` (head-only, `_orig_mod.transformer_head.*`; no config inside —
  checked). Arch: convnext_base, embed_dim 1024, depth 4, num_heads 16, cond_dim 12.
- **Other assets** (same folder): `normalization_statistics.pth` (T mean
  `[2.1, 0.0, -0.04]` cm, std `[2.85, 5.68, 6.08]` cm; rotation-mean reshapes
  near-degenerate → SVD-project to SO(3)),
  `dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth`.
- **Green set**: `/Users/jileihao/data/vigil/2025-nov-demo_2246f_green/` (431 png
  + frame.imu.csv). Baseline: `.../2025-nov-demo_2246f/` (2247 frames).
- **Scripts** (all default to image_size 512 now):
  - `scripts/diagnose_probe_pose.py` — green vs baseline |T_rel| + R angle +
    per-frame responsiveness; `--image-size` flag.
  - `scripts/perview_argmin_probe_pose.py` — experiment A (argmin/rank over views).
  - `scripts/cfg_probe_pose.py` — CFG scale sweep (kept for reference; CFG not used).
  - `scripts/smoke_probe_pose.py` — single forward per view.
- **View index map** (`CANONICAL_VIEW_LIST`): `0=A4C, 1=A2C, 2=PSAX_PAP,
  3=PSAX_MV, 4=PLAX, 5=SC4C`. Canonical poses in
  `usgr/usgr/guidance/datasets.py` (note A4C and A2C share canonical T).
- **usgr repo** checked out at `/Users/jileihao/dev/vigil-developer/usgr/`.

## Reference results (green A4C, target A4C)

| Metric | 9k/384 | 21k/384 | 21k/512 | converged should be |
|---|---|---|---|---|
| \|T_rel\| green | 7.57 cm | 2.42 cm | 4.89 cm | → ~0, ≪ baseline |
| \|T_rel\| baseline | 6.98 cm | 2.43 cm | 4.91 cm | |
| R angle green | 56.9° | 47.5° | 27.8° | → ~0°, ≪ baseline |
| A4C argmin rank by R | 2.44 | 1.02 | **0.29** | 0 |
| A4C argmin rank by \|T\| | 5.00 | 0.97 | 2.92 | 0 |
| per-frame T std (cm) | [.06,.08,.14] | [.14,.32,.27] | [.17,.44,.48] | large |

## Working-tree state

Branch `feature/usgr-integration`. Uncommitted: the Kalman/ESKF filter package
(`src/inference/probe_pose/filter/` + predictor/pipeline/UI/pyproject edits from
the previous session) **plus this session's changes**: predictor default
image_size 384→512, new `mirror_input=False` flag on the predictor (chirality
scaffolding, no-op today), `--image-size` and `--flip` flags on the diagnostic
scripts, new `scripts/perview_argmin_probe_pose.py` and `scripts/cfg_probe_pose.py`,
and degenerate-rotation fallback in `predict()` (matches USGR
realtime_inference.py: det(R_raw)<=0 + filter initialized ⇒ ESKF predict-only
extrapolation instead of discarding the frame; `valid` semantics updated —
False now only means degenerate AND no filter state). Measured invalid rate on
ema_21000 is rare (~0.1% of forwards at 384), but raw det(R) runs ~0.04–0.46
(vs 1.0 for a proper rotation) — det is a useful per-frame
confidence/convergence telemetry; consider surfacing it.
