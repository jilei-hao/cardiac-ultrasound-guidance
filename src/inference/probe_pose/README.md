# probe_pose — standard-view probe-pose guidance

Wraps the USGR `PoseTransformer` (vendored under `usgr_model/`) to predict the
body-centric relative SE(3) transform `[R_rel | T_rel]` from the current
ultrasound frame to a selectable standard TTE view. This adds **translation
guidance** on top of the existing rotation-only `RotationPredictor`, and lets the
operator switch among the **6 standard views**.

## Standard views (one-hot index)
`0=A4C, 1=A2C, 2=PSAX_PAP, 3=PSAX_MV, 4=PLAX, 5=SC4C` (`CANONICAL_VIEW_LIST`).

## Required assets (all three, else the feature auto-disables)
Set in `.env` (see `.env.example`):
- `PROBE_POSE_WEIGHTS` — EMA transformer-head checkpoint, e.g. `ema_9000.pth`.
  Contains **only** `transformer_head.*` (the frozen encoder is excluded); the
  loader strips the `_orig_mod.` prefix left by `torch.compile`.
- `PROBE_POSE_NORM_STATS` — pose `Normalizer` stats (`normalization_statistics.pth`),
  required to denormalize the 12-D output back to metric scale.
- `PROBE_POSE_DINO_WEIGHTS` — the frozen DINOv3 **ConvNeXt-Base** pretrained
  weights (`dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth`). The backbone
  architecture is loaded from the dinov3 hub already bundled at
  `src/inference/dinov3/`.

## Hotkeys (GuidanceUI)
- `1`–`6` — select target standard view (default `1`=A4C).
- `g` — toggle the rotation-arrow source between the USGR model (`usgr`, default)
  and the legacy `RotationPredictor` (`legacy`).
- Translation arrows + the target-probe ghost (`b`) are driven by `T_rel`.

## Coordinate-frame calibration (must tune on real frames)
`T_rel` is in the USGR probe frame (metres); the renderer's translation arrows
use its own IMU body frame. Tune these `GuidanceWidgetRenderer3D` attributes so
the arrows point the physically correct way:
- `usgr_to_body` (3×3 sign/permutation) — maps USGR axes → renderer body axes.
- `translation_amplification`, `translation_arrow_threshold_m` — arrow gain / deadzone.
- `guidance_translation_amplification` — spatial offset gain for the predicted axes
  and the target-probe ghost (raise to exaggerate cm-scale moves for visibility).

If the USGR rotation arrows point opposite to the legacy ones, the relative-
rotation sign convention may need inverting before feeding `R_rel` to the renderer.
