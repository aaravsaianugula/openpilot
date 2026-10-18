#!/usr/bin/env bash
# Run by hand (there is no boot hook: rerunning the Vision Long installer after an AGNOS update is
# the supported path). It puts back the Vision Long units (the monitor's and the USB link's) an AGNOS
# update removed, from the shipped copies, and changes nothing when they are in place. Detached and
# time-limited. The outcome goes to /data/vision_long/reestablish.log.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
# timeout sends TERM at 120 s; persist.py turns it into an exception so / is remounted read-only
# before it exits. KILL follows only 30 s later, far longer than that remount takes.
setsid nohup timeout -k 30 120 /usr/local/venv/bin/python3 "$DIR/persist.py" reestablish >/dev/null 2>&1 </dev/null &
exit 0
