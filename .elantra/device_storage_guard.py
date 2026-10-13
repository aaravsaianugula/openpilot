#!/usr/bin/env python3
"""Read-only device allocation preflight above the native log-deleter floor.

Run BEFORE transfer/expansion, counting all planned native/compressed/temp/output
allocations. It never imports the deleter, stops a service or changes its policy.
Existing inference adapters remain frozen; launchers must call this guard first.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

GIB = 1024**3
MARGIN = 2 * GIB


class StorageError(ValueError):
  pass


def allocation_budget(total, free, additional, min_bytes, min_percent):
  values = (total, free, additional, min_bytes, min_percent)
  if (any(type(v) is not int for v in values) or total <= 0 or not 0 <= free <= total or
      additional < 0 or min_bytes <= 0 or not 0 < min_percent <= 100):
    raise StorageError("invalid integer filesystem/allocation/deleter contract")
  floor = max(min_bytes, (total * min_percent + 99) // 100)
  required = additional + floor + MARGIN
  if free < required:
    raise StorageError(f"allocation risks native deleter: free={free}, planned={additional}, floor={floor}, margin={MARGIN}, required={required}")
  return {"total_bytes": total, "free_before_bytes": free, "planned_additional_bytes": additional,
          "deleter_floor_bytes": floor, "margin_above_deleter_bytes": MARGIN,
          "required_free_before_bytes": required, "predicted_free_after_bytes": free - additional}


def _integer_literal(node):
  if isinstance(node, ast.Constant) and type(node.value) is int:
    return node.value
  if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
    return _integer_literal(node.left) * _integer_literal(node.right)
  raise StorageError("native deleter threshold is not a supported integer literal/product")


def read_policy(path, expected_sha):
  if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
    raise StorageError("explicit expected native deleter source hash required")
  raw = path.read_bytes()
  digest = hashlib.sha256(raw).hexdigest()
  if digest != expected_sha:
    raise StorageError("native deleter source hash changed")
  try:
    tree = ast.parse(raw.decode("utf-8"))
  except (SyntaxError, UnicodeError) as exc:
    raise StorageError("native deleter source cannot be parsed") from exc
  wanted, assignments = {"MIN_BYTES", "MIN_PERCENT"}, {}
  for node in ast.walk(tree):
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in wanted:
      matches = [item for item in tree.body if isinstance(item, ast.Assign) and len(item.targets) == 1 and item.targets[0] is node]
      if len(matches) != 1 or node.id in assignments:
        raise StorageError("ambiguous/nonliteral native deleter threshold assignment")
      assignments[node.id] = _integer_literal(matches[0].value)
  if set(assignments) != wanted:
    raise StorageError("native deleter thresholds missing; no fallback")
  minimum, percent = assignments["MIN_BYTES"], assignments["MIN_PERCENT"]
  if minimum <= 0 or not 0 < percent <= 100:
    raise StorageError("invalid native deleter threshold values")
  return {"min_bytes": minimum, "min_percent": percent, "source": str(path.resolve()), "source_sha256": digest}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--additional-bytes", type=int, required=True)
  parser.add_argument("--expected-deleter-sha256", required=True)
  args = parser.parse_args()
  try:
    if os.name != "posix":
      raise StorageError("live check requires the actual POSIX comma filesystem")
    data, routes = Path("/data"), Path("/data/media/0/realdata")
    if data.stat().st_dev != routes.stat().st_dev:
      raise StorageError("drive logs and planned /data allocations are on different filesystems")
    policy = read_policy(data / "openpilot/openpilot/system/loggerd/deleter.py", args.expected_deleter_sha256)
    usage = shutil.disk_usage(data)
    result = allocation_budget(usage.total, usage.free, args.additional_bytes, policy["min_bytes"], policy["min_percent"])
    result.update(status="passed", native_policy=policy, no_writes=True,
                  adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope="Pre-allocation check only; all allocations must be counted and rechecked before subsequent phases.")
  except (OSError, StorageError) as exc:
    print(json.dumps({"status": "rejected", "error": str(exc), "no_writes": True}))
    return 2
  print(json.dumps(result))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
