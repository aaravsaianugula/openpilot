#!/bin/sh
# Zero-touch lateral data collection for the CN7 Elantra.
#
# Runs from a systemd timer. Every drive is analysed on its own, with nothing to
# switch on and nothing to remember. It never touches the driving path: it refuses to start
# unless the car is offroad, and it holds a lock so a long scan can never overlap the next
# tick.
#
# What it is actually watching for, in order of how much it matters:
#
#   1. CF_Mdps_ToiFlt -- EPS faults, per speed band. This is THE signal for a raised torque
#      ceiling. It costs nothing and it is available on every drive.
#   2. A non-zero CR_Lkas_StrToqReq from the FACTORY camera on bus 2. Nobody has ever measured
#      what the stock CN7 LKAS asks for, and while openpilot is engaged the camera never
#      actuates, so this normally reads zero. If a genuinely passive window ever happens, this
#      catches it without anyone planning for it. It may never fire; it is free if it does not.
#   3. The pinned-percentage and demand-vs-delivered table, tagged with the git commit,
#      safetyParam and lateral params the drive was recorded under, so a before/after can
#      never accidentally mix two configurations.
#
# Install (from a laptop). A crontab is NOT usable here: /var is a tmpfs on this device, so
# the crontab is wiped on every boot and the watcher silently stops. Use the systemd timer,
# which is what is actually deployed (elantra-lateral-watch.timer, every 30 min, Persistent).
#
#   scp .elantra/lateral_report.py .elantra/lateral_watch.sh comma@<device>:/data/elantra-lateral/
#   ssh comma@<device> 'chmod +x /data/elantra-lateral/lateral_watch.sh'
#   # then, as root on the device, write the two units and enable the timer:
#   #   /etc/systemd/system/elantra-lateral-watch.service  (Type=oneshot, User=comma,
#   #     ExecStart=/data/elantra-lateral/lateral_watch.sh, TimeoutStartSec=2700)
#   #   /etc/systemd/system/elantra-lateral-watch.timer    (OnBootSec=8min,
#   #     OnUnitActiveSec=30min, Persistent=true, WantedBy=timers.target)
#   systemctl daemon-reload && systemctl enable --now elantra-lateral-watch.timer
#
# Check:
#   ssh comma@<device> 'systemctl list-timers elantra-lateral-watch.timer --no-pager'
#
# Remove:
#   ssh comma@<device> 'sudo systemctl disable --now elantra-lateral-watch.timer'

set -eu

ROOT=/data/elantra-lateral
REPORTS=$ROOT/reports
LOG=$ROOT/scan.log
LOCK=$ROOT/.lock
PY=/usr/local/venv/bin/python3
OPENPILOT=/data/openpilot

# Onroad means the car is being driven. Reading 500 rlogs would steal CPU from the thing
# actually steering, so this simply does not run then -- the data is not going anywhere.
[ "$(cat /data/params/d/IsOffroad 2>/dev/null || echo 0)" = "1" ] || exit 0

# Nothing to do if the device is short on space; the reports are small but the scan is not
# worth risking a full /data over.
avail=$(df -P /data | awk 'NR==2 {print $4}')
[ "$avail" -gt 262144 ] || exit 0   # 256 MB

[ -x "$PY" ] || exit 0
[ -f "$ROOT/lateral_report.py" ] || exit 0

mkdir -p "$REPORTS"

# Non-blocking: if the previous tick is still scanning, this one is simply skipped. Without
# this two scans would race on the same report files after a long drive.
exec 9>"$LOCK"
flock -n 9 || exit 0

{
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  PYTHONPATH="$OPENPILOT" "$PY" "$ROOT/lateral_report.py" scan --routes /data/media/0/realdata --out "$REPORTS"
} >> "$LOG" 2>&1

# Keep the log from growing without bound -- /data is chronically near full on this device.
if [ "$(wc -c < "$LOG")" -gt 1048576 ]; then
  tail -c 524288 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi

# Surface the two things worth knowing without having to go looking for them.
grep -h "FACTORY-TORQUE" "$LOG" | tail -5 > "$ROOT/FACTORY_ENVELOPE_SEEN.txt" 2>/dev/null || true
[ -s "$ROOT/FACTORY_ENVELOPE_SEEN.txt" ] || rm -f "$ROOT/FACTORY_ENVELOPE_SEEN.txt"
