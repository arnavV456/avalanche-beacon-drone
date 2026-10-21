#!/bin/bash
# Run ONCE on the Pi, from the folder that holds autolog.py:
#     bash install_service.sh            install + start now + start on every boot
#     bash install_service.sh --print    just show the service file (no changes)
# Afterwards the logger is always running; nothing to start by hand.
#     cat data/STATUS.txt                what it is doing right now
#     journalctl -u capstone-logger -f   its log
#     bash install_service.sh --remove   undo
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$DIR/venv/bin/python3"; [ -x "$PY" ] || PY="$(command -v python3)"
WHO="${SUDO_USER:-$USER}"
UNIT=/etc/systemd/system/capstone-logger.service

unit_text() {
cat <<UNIT
[Unit]
Description=Beacon RSSI auto-logger (passive: logs only, never commands the drone)
After=multi-user.target

[Service]
Type=simple
User=$WHO
SupplementaryGroups=dialout
WorkingDirectory=$DIR
ExecStart=$PY -u $DIR/autolog.py --conn auto --data-dir $DIR/data
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT
}

case "${1:-}" in
  --print)  unit_text ;;
  --remove) sudo systemctl disable --now capstone-logger || true
            sudo rm -f "$UNIT"; sudo systemctl daemon-reload; echo "removed" ;;
  *)        [ -f "$DIR/autolog.py" ] || { echo "autolog.py not found in $DIR"; exit 1; }
            mkdir -p "$DIR/data"
            unit_text | sudo tee "$UNIT" >/dev/null
            sudo systemctl daemon-reload
            sudo systemctl enable --now capstone-logger
            sleep 3
            systemctl --no-pager --lines=8 status capstone-logger || true
            echo; echo "Installed. Check:  cat $DIR/data/STATUS.txt" ;;
esac
