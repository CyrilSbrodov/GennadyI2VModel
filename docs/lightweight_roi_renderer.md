# Lightweight ROI Renderer v1 (Sprint 9)

Sprint 9 adds a minimal local trainable ROI patch renderer. It consumes Sprint 8 `roi_renderer_contract` metadata, trains only from manifest-backed observed ROI pairs, and keeps all output generated/weak rather than observed or authoritative.

## Model architecture

`TinyROIRenderer` is a shallow CPU-compatible CNN encoder/decoder: two local convolutions, one stride-2 convolutional bottleneck, bilinear upsample, and a final sigmoid head. It has no attention, transformer, diffusion process, video model, or external checkpoint dependency. Default width is `base_channels=16`.

## Input channels and conditioning

The default tensor has 16 channels:

- 3 source ROI RGB channels.
- 1 alpha/mask channel.
- 12 stable scalar conditioning planes from Sprint 8 metadata.

Reference ROI payloads are accepted by the builder API for future expansion, but v1 keeps the default trainable tensor at 16 channels by not adding separate reference RGB channels.

Conditioning fields include `render_mode`, `route_decision_type`, `canonical_region_id`, `memory_family`, `memory_authority`, `output_authority`, `identity_locked`, `weak_memory_candidate`, `source_delta_type`, `source_reveal_decision_type`, `action_type`, and `phase_id`.

## Training manifest format

The v3 manifest is JSON with `samples` (or `records`). Each sample must include:

- `roi_before`: array, `.npy`, or image path.
- `roi_after`: observed target array, `.npy`, or image path.
- `target_provenance: "observed"` (default is observed for compatibility).
- `canonical_region_id`, `route_decision_type`, `render_mode`.
- Recommended: `region_id`, `memory_family`, `memory_authority`, `output_authority`, `identity_locked`, `weak_memory_candidate`, `source_delta_type`, `source_reveal_decision_type`, `action_type`, `phase_id`, `alpha`, `alpha_target`, and safety flags.

Generated runtime outputs and bootstrap/self-generated targets are rejected as supervised targets.

## Training command

```bash
python -m training.cli --stage train_roi_renderer --manifest path/to/manifest.json --out artifacts/checkpoints/lightweight_roi.pt --epochs 1 --batch-size 1 --lr 0.001 --device cpu --roi-size 16 --seed 0 --strict
```

## Checkpoint format

Checkpoints are local `torch.save` dictionaries with:

- `model_state_dict`
- `model_config`
- `contract_version: lightweight_roi_renderer_v1`
- `training_manifest_contract_version`
- `roi_renderer_contract_version`
- stable hash/vocab metadata
- safety metadata
- training metrics summary

Raw user images and observed target tensors are not saved in checkpoints.

## Runtime backend config

The default patch backend is unchanged. The optional backend is selected with `patch_backend="lightweight_roi"` (or `learned_roi_renderer`) and `patch_checkpoint_path=<checkpoint>`. If no checkpoint is loaded, fallback is explicit in execution trace and does not claim learned output.

## Sprint 8 contract consumption

`LightweightROIRendererBackend` reads `PatchSynthesisRequest.transition_context["roi_renderer_contract"]`, extracts the first ROI render request if a full contract is provided, validates safe route/output metadata, builds tiled conditioning planes, runs the CNN, and wraps output metadata through the Sprint 8 ROI output validator.

## Safety constraints

The backend rejects private/optional, blocked, diagnostic, unknown/deferred, identity-risk, weak identity reveal, forbidden-operation, and authoritative/reusable/observed-output routes. It never writes memory, mutates the scene graph, creates observed evidence, creates identity memory, removes clothing, renders private anatomy, or generates hidden anatomy.

## Limitations and future work

This is a tiny patch renderer for deterministic tests and local training plumbing. It is not a full video generator, temporal stabilizer, identity recognizer, high-quality diffusion model, or clothing/anatomy reconstruction system.
