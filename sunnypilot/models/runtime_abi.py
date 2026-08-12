"""Tinygrad ABI selection for driving-model artifacts.

A tinygrad driving model ships as a pickle of JIT-compiled closures, so it records
the tinygrad classes it was compiled against. Tinygrad renamed its JIT class
(`TinyJit` -> `_TinyJit`), which makes the two generations of artifact mutually
unloadable: a post-rename pickle unpickled by a pre-rename checkout fails with

    AttributeError: Can't get attribute '_TinyJit' on
    <module 'tinygrad.engine.jit' from '.../tinygrad/engine/jit.py'>

so the two families are served by two pinned checkouts:

  * `tinygrad_repo`     - pre-rename, what the sunnypilot v18 catalog is built against
  * `tinygrad_rdf_repo` - post-rename, what the OpenPilot experimental artifacts use

Which ABI an artifact needs is a property of whoever compiled it, NOT of the
OpenPilot source ref it was trained from. No-PP is the worked example: its source
ref pins a pre-rename tinygrad, yet the published pickle is post-rename because
the publisher compiled it with a newer checkout. So the ABI is declared per bundle
in `openpilot_experimental.py`, established by reading the published pickles rather
than inferred from source refs.

Bundles that reach modeld are capnp `ModelBundle` values, which drop unknown fields,
so selection keys off `ref` - an exact commit SHA that survives the round trip.
"""

import os

from openpilot.common.basedir import BASEDIR

LEGACY_ABI = "legacy"
RDF_ABI = "rdf"

# Directory holding the post-rename tinygrad checkout. It is a package parent, so
# placing it on PYTHONPATH is enough to make `import tinygrad` resolve to it.
RDF_RUNTIME_DIRNAME = "tinygrad_rdf_repo"

_rdf_abi_refs: frozenset[str] | None = None


def rdf_abi_refs() -> frozenset[str]:
  """OpenPilot refs whose published artifacts need the post-rename tinygrad."""
  global _rdf_abi_refs
  if _rdf_abi_refs is None:
    # Imported lazily: openpilot_experimental imports the ABI constants from here.
    from openpilot.sunnypilot.models.openpilot_experimental import OPENPILOT_EXPERIMENTAL_BUNDLES
    _rdf_abi_refs = frozenset(bundle["ref"] for bundle in OPENPILOT_EXPERIMENTAL_BUNDLES
                              if bundle.get("runtime_abi") == RDF_ABI)
  return _rdf_abi_refs


def bundle_runtime_abi(bundle) -> str:
  """Return the tinygrad ABI required by `bundle`, defaulting to the catalog's own."""
  ref = getattr(bundle, "ref", None) if bundle is not None else None
  return RDF_ABI if ref in rdf_abi_refs() else LEGACY_ABI


def runtime_pythonpath_entry(bundle, basedir: str | None = None) -> str | None:
  """Path to prepend to PYTHONPATH so `import tinygrad` picks the right checkout.

  Returns None when the default checkout already on the path is the correct one.
  """
  if bundle_runtime_abi(bundle) != RDF_ABI:
    return None
  return os.path.join(basedir or BASEDIR, RDF_RUNTIME_DIRNAME)
