# One-time laptop GPU setup on Windows 11, in an Administrator PowerShell (or over SSH as an admin).
# Needs WSL's Ubuntu-24.04 installed (wsl --install -d Ubuntu-24.04 --no-launch, then a reboot). Safe to re-run.
param([string]$Branch = "master")
$ErrorActionPreference = "Stop"
$raw = "https://raw.githubusercontent.com/aaravsaianugula/openpilot/$Branch/.elantra/laptop_gpu"

# mirrored networking: the server in WSL listens on the comma's USB network adapter itself
$cfg = "$env:USERPROFILE\.wslconfig"
if (-not (Select-String -Quiet -Pattern networkingMode -Path $cfg -ErrorAction SilentlyContinue)) {
  Add-Content $cfg "[wsl2]`nnetworkingMode=mirrored"
  wsl --shutdown
}

# only the comma's end of the link may reach the server, through both firewalls WSL traffic crosses
$wslVm = "{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}"
if (-not (Get-NetFirewallHyperVRule -Name LaptopGpu -ErrorAction SilentlyContinue)) {
  New-NetFirewallHyperVRule -Name LaptopGpu -DisplayName "Laptop GPU (comma)" -Direction Inbound -VMCreatorId $wslVm -Protocol TCP -LocalPorts 8390 -RemoteAddresses 192.168.250.1 | Out-Null
}
if (-not (Get-NetFirewallRule -Name LaptopGpu -ErrorAction SilentlyContinue)) {
  New-NetFirewallRule -Name LaptopGpu -DisplayName "Laptop GPU (comma)" -Direction Inbound -Protocol TCP -LocalPort 8390 -RemoteAddress 192.168.250.1 -Action Allow | Out-Null
}

# code, GPU runtime, and the default model compiled and checked for this GPU (the compile takes a while)
wsl -d Ubuntu-24.04 -u root -- sh -c "curl -fsSL $raw/setup_wsl.sh | sh -s $Branch"
if ($LASTEXITCODE -ne 0) { throw "WSL setup failed" }

# start at logon: keeps the link addressed and the server running
$dir = "$env:ProgramData\laptop_gpu"
New-Item -ItemType Directory -Force $dir | Out-Null
Invoke-WebRequest "$raw/start.ps1" -OutFile "$dir\start.ps1"
$action = New-ScheduledTaskAction -Execute powershell.exe -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$dir\start.ps1`""
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit 0
Register-ScheduledTask LaptopGpu -Action $action -Trigger (New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME) -Settings $settings -RunLevel Highest -User $env:USERNAME -Force | Out-Null
Start-ScheduledTask LaptopGpu
"laptop GPU ready: plug the comma's aux USB-C port into this laptop"
