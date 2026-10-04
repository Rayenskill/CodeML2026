#!/usr/bin/env bash
# Start the DayOne edge box.  ./run_box.sh [--https] [port]   (arguments in any order)
#   default : http://localhost:<port> on this machine (secure origin for the browser of the box itself)
#   --https : https://<box-ip>:<port> for the phones on the facility Wi-Fi (TLS certificate, see
#             work/strat20/make_cert.py; WebCrypto on the phone requires HTTPS)
set -e
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
PORT=8765; HTTPS=0
for arg in "$@"; do
  case "$arg" in
    --https) HTTPS=1 ;;
    [0-9]*) PORT="$arg" ;;
    *) echo "usage: $0 [--https] [port]" >&2; exit 2 ;;
  esac
done
STATE="${DAYONE_STATE:-$HOME/dayone_local/edge_state}"
[ -z "$DAYONE_BOX_KEY" ] && echo "note: DAYONE_BOX_KEY not set, a random state key is generated in $STATE/box.key" >&2
if [ "$HTTPS" = 1 ]; then
  python work/strat20/make_cert.py "$STATE"
  exec python -m uvicorn edge_server:app --app-dir work/strat20 --host 0.0.0.0 --port "$PORT" \
       --ssl-certfile "$STATE/box_tls.crt" --ssl-keyfile "$STATE/box_tls.key"
fi
exec python -m uvicorn edge_server:app --app-dir work/strat20 --host 0.0.0.0 --port "$PORT"
