import hashlib

import pytest

from device_storage_guard import GIB, StorageError, allocation_budget, read_policy


def source(tmp_path, text="MIN_BYTES = 5 * 1024 * 1024 * 1024\nMIN_PERCENT = 10\n"):
  path = tmp_path / "deleter.py"
  path.write_text(text)
  return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixed_two_gib_reserve_does_not_protect_native_deleter_threshold():
  # Actual incident disk statistics: 10.0494% free, yet almost no room above
  # the native 10% floor. A 2GiB-only guard would incorrectly admit expansion.
  with pytest.raises(StorageError, match="deleter"):
    allocation_budget(94_524_579_840, 9_499_152_384, 1_340_000_000, 5 * GIB, 10)


def test_budget_includes_allocation_and_margin_above_larger_native_floor():
  value = allocation_budget(100 * GIB, 16 * GIB, 3 * GIB, 5 * GIB, 10)
  assert value["deleter_floor_bytes"] == 10 * GIB
  assert value["required_free_before_bytes"] == 15 * GIB
  assert value["predicted_free_after_bytes"] == 13 * GIB
  assert value["margin_above_deleter_bytes"] == 2 * GIB
  assert allocation_budget(20 * GIB, 8 * GIB, GIB, 5 * GIB, 10)["deleter_floor_bytes"] == 5 * GIB


def test_boundary_uses_ceiling_for_percentage_and_preserves_two_gib_margin():
  total = 100 * GIB + 1
  floor = 10 * GIB + 1
  assert allocation_budget(total, floor + 2 * GIB + 7, 7, 5 * GIB, 10)["deleter_floor_bytes"] == floor
  with pytest.raises(StorageError, match="deleter"):
    allocation_budget(total, floor + 2 * GIB + 6, 7, 5 * GIB, 10)


@pytest.mark.parametrize("values", [(0, 0, 0, 5 * GIB, 10), (10, 11, 0, 1, 10),
                                   (10, 5, -1, 1, 10), (10, 5, 0, 0, 10), (10, 5, 0, 1, 0),
                                   (10, 5, 0, 1, 101), (True, 1, 0, 1, 10), (10, 5, 0., 1, 10)])
def test_invalid_disk_or_allocation_contract_fails(values):
  with pytest.raises(StorageError):
    allocation_budget(*values)


def test_reads_literal_policy_without_importing_device_service(tmp_path):
  path, digest = source(tmp_path)
  value = read_policy(path, digest)
  assert value["min_bytes"] == 5 * GIB and value["min_percent"] == 10
  assert value["source_sha256"] == digest
  with pytest.raises(StorageError, match="hash"):
    read_policy(path, "0" * 64)


@pytest.mark.parametrize("text", ["MIN_BYTES=5<<30\nMIN_PERCENT=10\n", "MIN_BYTES=5\nMIN_PERCENT=10\nMIN_PERCENT=2\n",
                                 "MIN_BYTES=5\n", "MIN_BYTES=True\nMIN_PERCENT=10\n", "MIN_BYTES=5\nMIN_PERCENT=0\n",
                                 "MIN_BYTES=5\nMIN_PERCENT=10\nMIN_BYTES+=100\n"])
def test_unknown_or_ambiguous_policy_has_no_fallback(tmp_path, text):
  path, digest = source(tmp_path, text)
  with pytest.raises(StorageError):
    read_policy(path, digest)


def test_policy_expressions_are_never_executed(tmp_path):
  marker = tmp_path / "unexpected-write"
  path, digest = source(tmp_path, f"MIN_BYTES = __import__('pathlib').Path({str(marker)!r}).write_text('bad')\nMIN_PERCENT=10\n")
  with pytest.raises(StorageError, match="literal"):
    read_policy(path, digest)
  assert not marker.exists()
