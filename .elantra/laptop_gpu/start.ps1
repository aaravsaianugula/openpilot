# Run at logon by the LaptopGpu task (setup.ps1): refresh the code, address the comma link, keep the server up.
param([string]$Branch = "master", [switch]$Prepare)
$root = "$env:ProgramData\laptop_gpu"
$python = "$root\venv\Scripts\python.exe"
$server = "$root\repo\.elantra\laptop_gpu\server.py"

# the branch's latest server and protocol, all or nothing; offline keeps what is here
$files = ".elantra/laptop_gpu/server.py", ".elantra/fetch_lfs_object.py", "openpilot/selfdrive/modeld/laptop.py"
try {
  foreach ($f in $files) {
    New-Item -ItemType Directory -Force (Split-Path "$root\new\$f") | Out-Null
    Invoke-WebRequest "https://raw.githubusercontent.com/aaravsaianugula/openpilot/$Branch/$f" -OutFile "$root\new\$f" -TimeoutSec 20
  }
  Copy-Item "$root\new\*" "$root\repo" -Recurse -Force
} catch { "code update skipped: $_" }

if ($Prepare) { & $python $server --prepare; exit $LASTEXITCODE }

$proc = $null
while ($true) {
  # the comma's USB gadget shows up as a "UsbNcm Host Device" adapter: give it the laptop's end of the link
  Get-NetAdapter -ErrorAction SilentlyContinue | Where-Object { $_.InterfaceDescription -match 'NCM' -and $_.Status -eq 'Up' } | ForEach-Object {
    if (-not (Get-NetIPAddress -InterfaceIndex $_.ifIndex -IPAddress 192.168.250.2 -ErrorAction SilentlyContinue)) {
      New-NetIPAddress -InterfaceIndex $_.ifIndex -IPAddress 192.168.250.2 -PrefixLength 24 | Out-Null
    }
  }
  if ($null -eq $proc -or $proc.HasExited) {
    $proc = Start-Process $python -ArgumentList "`"$server`"" -NoNewWindow -PassThru -RedirectStandardOutput "$root\server.log" -RedirectStandardError "$root\server.err"
  }
  Start-Sleep 5
}
