#!/usr/bin/env python3
"""Print the PYTHONPATH entry modeld needs for the active model's tinygrad ABI.

Called by the `modeld` launcher before modeld.py imports tinygrad. Prints nothing
when the checkout already on the path is the right one, so the caller can treat
empty output as "use the repo default".

Detection cannot happen inside modeld.py: by the time it runs, `import tinygrad`
has already bound a module, and the wrong one cannot be swapped out afterwards.
"""

import sys

from openpilot.sunnypilot.models.helpers import get_active_bundle
from openpilot.sunnypilot.models.runtime_abi import runtime_pythonpath_entry


def main() -> int:
  try:
    entry = runtime_pythonpath_entry(get_active_bundle())
  except Exception as e:
    # Never block modeld from starting; the repo default is the safe fallback.
    print(f"select_tinygrad_runtime: falling back to default runtime ({e})", file=sys.stderr)
    return 0
  if entry:
    print(entry)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
