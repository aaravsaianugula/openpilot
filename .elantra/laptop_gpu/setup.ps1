# One-time laptop GPU setup on Windows 11 with an NVIDIA GPU, in an Administrator PowerShell. Safe to re-run.
#   irm https://raw.githubusercontent.com/aaravsaianugula/openpilot/master/.elantra/laptop_gpu/setup.ps1 -OutFile $env:TEMP\lg.ps1; & $env:TEMP\lg.ps1
param([string]$Branch = "master")
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # Windows PowerShell downloads crawl with the progress bar on
$root = "$env:ProgramData\laptop_gpu"
New-Item -ItemType Directory -Force $root | Out-Null

# Python 3.12 (python.org's last 3.12 installer) and onnxruntime with CUDA and cuDNN from pip
$py = "$env:ProgramFiles\Python312\python.exe"
if (-not (Test-Path $py)) {
  Invoke-WebRequest https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe -OutFile "$env:TEMP\python312.exe"
  Start-Process "$env:TEMP\python312.exe" "/quiet InstallAllUsers=1 Include_launcher=0 Include_test=0" -Wait
}
if (-not (Test-Path "$env:SystemRoot\System32\vcruntime140_1.dll")) {  # onnxruntime's DLLs need the VC++ runtime
  Invoke-WebRequest https://aka.ms/vs/17/release/vc_redist.x64.exe -OutFile "$env:TEMP\vc_redist.x64.exe"
  Start-Process "$env:TEMP\vc_redist.x64.exe" "/install /quiet /norestart" -Wait
}
if (-not (Test-Path "$root\venv")) { & $py -m venv "$root\venv" }
& "$root\venv\Scripts\python.exe" -m pip install -q --upgrade numpy onnx "onnxruntime-gpu[cuda,cudnn]"  # onnx rebuilds pre-Sep-14 exports
if ($LASTEXITCODE) { throw "pip install failed" }

# only the comma's end of the USB link may reach the server
if (-not (Get-NetFirewallRule -Name LaptopGpu -ErrorAction SilentlyContinue)) {
  New-NetFirewallRule -Name LaptopGpu -DisplayName "Laptop GPU (comma)" -Direction Inbound -Program $py -Protocol TCP -LocalPort 8390 -RemoteAddress 192.168.250.1 -Action Allow | Out-Null
}

# stay awake on the charger and with the lid shut: the laptop rides in the car
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0
powercfg /setdcvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0
powercfg /change standby-timeout-ac 0
powercfg /setactive SCHEME_CURRENT

# code, then every chestnut model comma published weights for, fetched and checked against the CPU (a while the first time)
Invoke-WebRequest "https://raw.githubusercontent.com/aaravsaianugula/openpilot/$Branch/.elantra/laptop_gpu/start.ps1" -OutFile "$root\start.ps1"
& "$root\start.ps1" -Branch $Branch -Prepare
if ($LASTEXITCODE) { throw "model check failed, see above" }

# start at logon: keeps the code current, the link addressed and the server running
$action = New-ScheduledTaskAction -Execute powershell.exe -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$root\start.ps1`" -Branch $Branch"
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit 0
Register-ScheduledTask LaptopGpu -Action $action -Trigger (New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME) -Settings $settings -RunLevel Highest -User $env:USERNAME -Force | Out-Null
Start-ScheduledTask LaptopGpu
"laptop GPU ready: plug the comma's aux USB-C port into this laptop. Log: $root\server.log"
