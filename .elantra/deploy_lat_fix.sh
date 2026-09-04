#!/usr/bin/env bash
# Stage the CN7 low-speed lateral fix onto the comma and prove it is live.
#
# Run from the superproject root, on the laptop:
#
#     .elantra/deploy_lat_fix.sh <host>            # copy + verify, do NOT restart
#     .elantra/deploy_lat_fix.sh <host> --restart  # copy + verify + restart comma
#
# The copy is harmless on its own: openpilot reads these files at process start, so nothing
# changes until comma restarts. That is why --restart is a separate, deliberate flag.
#
# /data/openpilot is a git checkout, so the rollback is git checkout -- <file>; this script prints
# what it is about to overwrite rather than leaving copies behind. Rollback is printed at the end.
set -euo pipefail

HOST="${1:-}"
RESTART="${2:-}"
if [ -z "$HOST" ]; then
  echo "usage: $0 <user@host|host> [--restart]" >&2
  exit 2
fi
case "$HOST" in *@*) ;; *) HOST="comma@$HOST" ;; esac

KEY="${COMMA_SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH=(ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -i "$KEY" "$HOST")
SCP=(scp -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -i "$KEY")
REMOTE=/data/openpilot
PY="PYTHONPATH=$REMOTE /usr/local/venv/bin/python3"

# The three files the CAR runs. Everything else below is tooling and documentation.
PROD=(
  openpilot/sunnypilot/selfdrive/controls/lib/lat_accel_factor_schedule.py
  openpilot/sunnypilot/selfdrive/controls/lib/latcontrol_torque_jerk_aware.py
  openpilot/sunnypilot/selfdrive/controls/lib/latcontrol_torque_v0.py
)
SUPPORT=(
  openpilot/sunnypilot/selfdrive/controls/lib/tests/test_lat_accel_factor_schedule.py
  .elantra/verify_lat_fix.py
  .elantra/ff_schedule_replay.py
  .elantra/test_guard_ff_schedule.py
  .elantra/guards.py
  .elantra/sync.py
  .elantra/ROAD-TEST-cn7-lateral.md
)

for f in "${PROD[@]}" "${SUPPORT[@]}"; do
  [ -f "$f" ] || { echo "missing locally: $f" >&2; exit 1; }
done

echo "=== who is this? ==="
# Identity before anything else: the whole point of a fingerprint check is to run it BEFORE writing.
"${SSH[@]}" "hostname; cat /data/params/d/DongleId 2>/dev/null; echo; uptime; echo; df -h /data | tail -1"
echo ""
echo "=== build currently on the device ==="
"${SSH[@]}" "cd $REMOTE && git log --oneline -1 && echo -n 'opendbc gitlink: ' && git ls-tree HEAD opendbc_repo | awk '{print substr(\$3,1,12)}'"

echo ""
echo "=== what we are about to overwrite ==="
# No file-copy backups. /data/openpilot is a git checkout, so `git checkout -- <file>` restores the
# pristine committed version, which is a better rollback than a copy. An earlier version of this
# script stamped each backup with the time, which defeated its own `cp -n`: a second run happily
# "backed up" the code the first run had just deployed, producing a file that looked like the
# original and was not. Showing the diff is more honest and leaves nothing behind in .elantra/,
# which the overlay restores wholesale.
"${SSH[@]}" "cd $REMOTE && git status --short -- ${PROD[*]} ${SUPPORT[*]} 2>/dev/null | sed 's/^/  /' || true"
echo "  (rollback: git checkout -- <file> on the device, or the blend constants -- see the end)"

echo ""
echo "=== copying ==="
for f in "${PROD[@]}" "${SUPPORT[@]}"; do
  "${SSH[@]}" "mkdir -p $REMOTE/$(dirname "$f")"
  "${SCP[@]}" -q "$f" "$HOST:$REMOTE/$f"
  echo "  -> $f"
done

echo ""
echo "=== on-device pre-flight (this is the check that has never been run before) ==="
set +e
"${SSH[@]}" "cd $REMOTE && $PY .elantra/verify_lat_fix.py"
VERIFY=$?
set -e
if [ "$VERIFY" -ne 0 ]; then
  echo ""
  echo "PRE-FLIGHT FAILED (exit $VERIFY). Nothing has been restarted, so the car is still running"
  echo "the previous build. Restore with:"
  echo "  ssh $HOST 'cd $REMOTE && git checkout -- <file>'"
  exit "$VERIFY"
fi

if [ "$RESTART" != "--restart" ]; then
  echo ""
  echo "Copied and verified. NOT restarted -- openpilot reads these at process start, so the car is"
  echo "still running the previous build until you either reboot it or re-run with --restart."
  exit 0
fi

echo ""
echo "=== restarting comma ==="
# Detached on purpose: restarting the service kills this ssh session, and a dropped connection then
# looks exactly like the device having crashed.
"${SSH[@]}" "setsid nohup sh -c 'sleep 2; sudo systemctl restart comma' >/dev/null 2>&1 </dev/null &" || true
echo "  restart issued (detached). Give it ~30 s, then:"
echo "    ssh $HOST 'uptime; systemctl is-active comma'"
echo ""
echo "Rollback, in this order:"
echo "  1. LOW_SPEED_KP_BLEND = 0.0   in $REMOTE/${PROD[0]}"
echo "  2. LOW_SPEED_FF_BLEND = 0.0   in the same file, only if 1 was not enough"
echo "  3. sudo systemctl restart comma"
echo "Or restore the committed originals: ssh $HOST 'cd $REMOTE && git checkout -- <file>'"
