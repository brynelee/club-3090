# Patch Attribution Profiles

v0.8.0 Phase A adds three data-only profile files:

- `scripts/lib/profiles/patches.yml` records every local patch bundle and Genesis env-gated patch, what it fixes, how it is delivered, and any known compose coverage gaps.
- `scripts/lib/profiles/arch_patches.yml` is the declarative input for the locked C0 engine-support gate: arch, loadable engine pins, required patches, valid TP flags, trust-remote-code tri-state, and kernel constraints.
- `scripts/lib/profiles/calibration_seed.yml` seeds the calibration backbone from directly measured `BENCHMARKS.md` rows.

These files support the v0.8.x scope: evaluate any safetensors HF repo; pull only vLLM-loadable supported ones, and only when the gates pass (or an explicit override is accepted).

When adding a local patch, add a `patches.yml` entry in the same change. Declare all delivery channels: Dockerfile bake, compose mount/invoke, or Genesis env gate. If the patch gates an architecture or engine path, add or extend the matching `arch_patches.yml` row. If a patch is known to be load-bearing but cannot reach a compose today, record it under `delivery_gaps` rather than silently fixing runtime files in the audit change.

When adding measured results that should seed the cold-start predictor, add a `calibration_seed.yml` anchor only for directly measured configs. `confidence: exact` applies only to capabilities listed in `smoked_capabilities`; everything else stays in `unsmoked_capabilities`.

If no seeded or community calibration anchor matches a future pull target, the verdict must say: `no calibration data for this config class -- prediction is an unvalidated lower bound`. Also surface the boot-fit caveat: static fit does not guarantee stability under accumulated-context workloads; run or request a continuous soak before treating a config as production-stable.

Run the audit with:

```bash
bash scripts/tests/test-patch-attribution.sh
```
