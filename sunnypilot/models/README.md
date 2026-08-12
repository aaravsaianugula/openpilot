# Model Selector Version Compatibility

This document explains the version compatibility mechanism used by the Model Selector system, and the rationale behind certain version constraints and JSON file management strategies.

## Overview

The Model Selector is responsible for selecting and validating model bundles based on their metadata and version constraints. Each model bundle is distributed via a JSON file and includes a `minimumSelectorVersion` field indicating the minimum selector version required to load it.

To ensure robust compatibility and prevent mismatches between model expectations and selector capabilities, the selector enforces two version boundaries:

* **`REQUIRED_MIN_SELECTOR_VERSION`**: the oldest selector version we support.
* **`CURRENT_SELECTOR_VERSION`**: the current version of the selector logic.

## Version Compatibility Check

A model bundle is considered compatible if:

```python
REQUIRED_MIN_SELECTOR_VERSION <= bundle["minimumSelectorVersion"] <= CURRENT_SELECTOR_VERSION
```

This ensures:

* **Old bundles are rejected** if they rely on deprecated selector behavior.
* **Future bundles are ignored** if they expect logic that the current selector doesn't yet implement.

## Handling Breaking Changes

When a deep change in selector behavior requires *all* models to be recompiled (e.g., due to a major architectural update), we:

1. **Create a new JSON file** (e.g., from `models_v4.json` to `models_v5.json`).
2. **Assign updated `minimumSelectorVersion` values** in the new bundles.

This allows older selector versions to continue using the previous JSON file, while newer versions point to the new one, preventing cross-contamination.

## Why `REQUIRED_MIN_SELECTOR_VERSION` Still Matters

Despite using new JSON files to isolate breaking changes, `REQUIRED_MIN_SELECTOR_VERSION` plays a critical role:

### 1. **Cached Bundle Validation**

Model bundles are cached locally (e.g., in-memory or on disk). A user might have previously loaded a now-invalid bundle from an older JSON file.

`REQUIRED_MIN_SELECTOR_VERSION` prevents the selector from reloading or trusting that stale cached bundle, even if the original JSON is gone.

### 2. **Explicit Deprecation Boundary**

By raising `REQUIRED_MIN_SELECTOR_VERSION`, we declare older bundles officially unsupported, even if they technically still exist in a legacy JSON file.

### 3. **Avoiding Race Conditions**

Some clients may have intermittent access to updated JSONs. The runtime check ensures version compatibility is enforced independently of external file state.

## Summary

| Component                       | Purpose                                                               |
| ------------------------------- | --------------------------------------------------------------------- |
| `minimumSelectorVersion`        | Declares the minimum selector version required to load a model bundle |
| `REQUIRED_MIN_SELECTOR_VERSION` | Prevents loading bundles that are too old (e.g., from stale cache)    |
| `CURRENT_SELECTOR_VERSION`      | Prevents loading bundles that are too new or forward-incompatible     |
| JSON file renaming              | Isolates bundles by selector generation to handle full recompiles     |

This layered strategy ensures safe evolution of the model selection system while maintaining backward compatibility and runtime protection against stale or incompatible bundles.

## OpenPilot experimental small models

This branch keeps the live Sunnypilot v18 catalog as its primary model catalog and adds a small local overlay for published official OpenPilot experiments that are missing from that catalog.

The currently exposed small-model lineup is:

* OpenPilot Model16 Deep
* Rebel Legion
* Get Your Hopes Up
* Rebellious Hope
* No-PP
* RDF checkpoints 1, 2, 3, 4, and 5

Model behavior is identified by an exact `commaai/openpilot` commit and source-model SHA-256. Existing v18 artifacts continue to use Sunnypilot's published catalog. No-PP and RDF checkpoints 2 through 5 use QCOM Tinygrad artifacts from the `firestar5683/StarPilot-Resources` `Models` branch because commaai publishes the source ONNX models but not a multi-model download catalog. Those runtime artifacts are accepted only when the SHA-256 of the complete reconstructed file matches the pinned value.

RDF checkpoint numbering is chronological community terminology. Official OpenPilot commit messages identify training checkpoints rather than naming releases "RDF v1" through "RDF v5". No checkpoint is automatically preferred or made the default.

The `nostd` OpenPilot branch is intentionally not exposed because no hash-pinned, comma-ready artifact was found. Adding a source ONNX entry without a compatible runtime artifact would create a selector option that cannot run.

Downloads and model selection are offroad operations. All experiments use the existing v18 combined-model runtime; there is no separate RL runner or Tinygrad checkout.
