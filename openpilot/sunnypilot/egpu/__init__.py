"""eGPU vendor support for the comma chestnut dock.

Upstream openpilot assumes the card in the chestnut is the AMD one comma ships. This package
holds everything that stops being true when it is not, so the upstream files we have to touch
stay down to a handful of one-line hooks -- the weekly sync replays those by three-way merge,
and every line of them is a conflict waiting to happen.
"""
