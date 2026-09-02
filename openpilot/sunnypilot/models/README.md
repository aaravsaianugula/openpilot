# Model Selector Version Compatibility

This document explains how the Model Selector decides which catalog of driving models to fetch, and which bundles inside it are loadable.

## Overview

Model bundles are published as versioned JSON catalogs on gh-pages. Each bundle carries a `minimum_selector_version` field (`minimumSelectorVersion` once parsed) declaring the selector version required to load it, and each catalog carries a top-level `tinygrad_ref` naming the tinygrad commit its artifacts were compiled against.

Two independent checks apply: one picks the catalog, the other filters the bundles inside it.

## Version Compatibility Check

`is_bundle_version_compatible` (`helpers.py`) is an **exact match**, not a range:

```python
bundle["minimumSelectorVersion"] == REQUIRED_JSON_VERSION
```

`REQUIRED_JSON_VERSION` is what this build of the selector understands. A bundle declaring anything else is dropped at parse time, before it reaches the UI. This is stricter than a range, and deliberately so: a catalog generation is recompiled as a whole, so a bundle from a different generation is never one this selector should load.

This runs on cached bundles too, so a stale bundle from a catalog that no longer exists cannot be revived out of `ModelManager_ModelsCache`.

## Catalog Resolution

`catalog.py` resolves which catalog URL to fetch rather than hardcoding one. Hardcoding strands the device: the eGPU feed was renamed from `driving_models_usbgpu_v*.json` to `driving_models_chestnut_v*.json` and moved on two versions while the device kept requesting the old name, so every model published after the pin was invisible.

Resolution walks upward from the newest catalog this build shipped against (`ONBOARD_FLOOR` / `CHESTNUT_FLOOR`), probing existence with cheap `HEAD` requests across both the current and the legacy filename for that family. Catalogs that exist are then downloaded newest-first, and the first **usable** one wins. Usable means both:

* the catalog's `tinygrad_ref` equals this checkout's `tinygrad_repo` HEAD, and
* at least one bundle passes the version check above.

The `tinygrad_ref` condition is the important one. Artifacts are compiled per tinygrad commit, so a newer catalog built against a tinygrad we do not run lists models this build cannot execute. Following it would download gigabytes of unusable weights. It is refused rather than silently accepted, and it becomes eligible on its own the moment the `tinygrad_repo` submodule is bumped -- no code change.

If nothing resolves (offline, or every published catalog is incompatible), the floor URL is used and the existing expired-cache fallback applies. A failed resolve is not cached, so a device that boots offline retries on the next sync.

## Which Catalog: on-board vs eGPU

There are two families, and they are separate lists rather than groups within one list:

| family | contents |
| --- | --- |
| `driving_models_v{n}.json` | models that run on the device SoC |
| `driving_models_chestnut_v{n}.json` | big models that require the Chestnut external GPU |

The eGPU family is selected when the dock is present (`deviceState.chestnutPresent`) **or** when the `ModelManager_ShowChestnutModels` param is set, so eGPU models can be browsed and downloaded ahead of time with nothing plugged in. The dock is still required to drive on one: `selfdrived` raises `bigModelFailed` if a big model is active without it.

The same boolean drives the cache-key suffix (`ModelManager_ModelsCache_USBGPU`) and `validate_active_bundle`'s two-slot selection stash (`ModelManager_PrevBundle` / `ModelManager_PrevBundle_USBGPU`), so switching context -- by toggle or by plugging the dock in -- parks the current selection and restores the other one.

## Handling Breaking Changes

When a change requires *all* models to be recompiled, a new catalog version is published with updated `minimum_selector_version` values and a new `tinygrad_ref`. Older selectors keep resolving to the previous catalog because the new one fails both checks; a selector updated in step resolves forward on its own.

## Summary

| Component | Purpose |
| --- | --- |
| `minimum_selector_version` | Declares the selector version required to load a bundle |
| `REQUIRED_JSON_VERSION` | The selector generation this build implements; bundles must match it exactly |
| `tinygrad_ref` | Names the tinygrad the catalog's artifacts were compiled against |
| `ONBOARD_FLOOR` / `CHESTNUT_FLOOR` | The catalog this build shipped against; the search floor and the fallback |
| `ModelManager_ShowChestnutModels` | Lists the eGPU catalog without the dock attached |
