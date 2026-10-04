# Start the DayOne edge box — PowerShell.  .\run_box.ps1 [-Port 8765] [-Https]
#   default : http://localhost:<port> on this machine (secure origin for the browser of the box itself)
#   -Https  : https://<box-ip>:<port> for the phones on the facility Wi-Fi (self-signed certificate, see
#             work\strat20\make_cert.py; WebCrypto on the phone requires HTTPS)
param([int]$Port = 8765, [switch]$Https)
$env:PYTHONIOENCODING = "utf-8"
$state = if ($env:DAYONE_STATE) { $env:DAYONE_STATE } else { Join-Path $HOME "dayone_local\edge_state" }
if (-not $env:DAYONE_BOX_KEY) { Write-Warning "DAYONE_BOX_KEY not set: a random state key is generated in $state\box.key" }
if ($Https) {
  python "$PSScriptRoot\work\strat20\make_cert.py" $state
  python -m uvicorn edge_server:app --app-dir "$PSScriptRoot\work\strat20" --host 0.0.0.0 --port $Port `
    --ssl-certfile "$state\box_tls.crt" --ssl-keyfile "$state\box_tls.key"
} else {
  python -m uvicorn edge_server:app --app-dir "$PSScriptRoot\work\strat20" --host 0.0.0.0 --port $Port
}
