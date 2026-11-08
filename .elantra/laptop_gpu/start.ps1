# Run at logon / network-connect by the LaptopGpu task (setup.ps1), elevated: refresh the code, address the comma link, keep the server up.
# -Local <repo root> copies the files from there instead of GitHub (deploy before pushing).
param([string]$Branch = "master", [switch]$Prepare, [string]$Local)
$root = "$env:ProgramData\laptop_gpu"
$python = "$root\venv\Scripts\python.exe"
$server = "$root\repo\.elantra\laptop_gpu\server.py"
$nvdrs = "$root\repo\.elantra\laptop_gpu\nvdrs.py"
$saved = "$root\power_saved.json"

# the branch's latest server and protocol, all or nothing; offline keeps what is here
$files = ".elantra/laptop_gpu/server.py", ".elantra/laptop_gpu/nvdrs.py", ".elantra/fetch_lfs_object.py", "openpilot/selfdrive/modeld/laptop.py"
try {
  foreach ($f in $files) {
    New-Item -ItemType Directory -Force (Split-Path "$root\new\$f") | Out-Null
    if ($Local) { Copy-Item "$Local\$f" "$root\new\$f" -Force }
    else { Invoke-WebRequest "https://raw.githubusercontent.com/aaravsaianugula/openpilot/$Branch/$f" -OutFile "$root\new\$f" -TimeoutSec 20 }
  }
  Copy-Item "$root\new\*" "$root\repo" -Recurse -Force
} catch { "code update skipped: $_" }

if ($Prepare) { & $python $server --prepare; exit $LASTEXITCODE }

# the Windows power overlay (Best performance / Balanced slider) has no powercfg switch: powrprof.dll; keep-awake is kernel32
Add-Type -Name P -Namespace LG -MemberDefinition @'
[DllImport("powrprof.dll")] public static extern uint PowerGetActualOverlayScheme(out Guid g);
[DllImport("powrprof.dll")] public static extern uint PowerSetActiveOverlayScheme(ref Guid g);
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);
'@
$bestPerf = [guid]"ded574b5-45a0-4f42-8737-46345c09c238"  # Balanced is 00000000-0000-0000-0000-000000000000

function Get-Ac($sub, $name) {  # AC index of a power setting in the active scheme (/qh: lid action is hidden), or $null when this machine lacks it
  $m = powercfg /qh SCHEME_CURRENT $sub $name | Select-String "Current AC Power Setting Index: 0x([0-9a-f]+)"
  if ($m) { [Convert]::ToInt32($m.Matches[0].Groups[1].Value, 16) }
}
function Set-Ac($sub, $name, $v) { if ($v -is [int] -or $v -is [long]) { powercfg /setacvalueindex SCHEME_CURRENT $sub $name $v } }

function Enter-Linked {
  if (-not (Test-Path $saved)) {  # a crash mid-link must not overwrite the originals with the linked values
    $g = [guid]::Empty; [void][LG.P]::PowerGetActualOverlayScheme([ref]$g)
    @{ lid = Get-Ac SUB_BUTTONS LIDACTION; standby = Get-Ac SUB_SLEEP STANDBYIDLE; hibernate = Get-Ac SUB_SLEEP HIBERNATEIDLE; overlay = "$g" } | ConvertTo-Json | Set-Content $saved
  }
  Set-Ac SUB_BUTTONS LIDACTION 0; Set-Ac SUB_SLEEP STANDBYIDLE 0; Set-Ac SUB_SLEEP HIBERNATEIDLE 0
  powercfg /setactive SCHEME_CURRENT
  [void][LG.P]::PowerSetActiveOverlayScheme([ref]$bestPerf)
  & $python $nvdrs apply
}
function Exit-Linked {
  [void][LG.P]::SetThreadExecutionState([uint32]2147483648)  # ES_CONTINUOUS alone: drop the keep-awake
  if (Test-Path $saved) {
    $s = Get-Content $saved -Raw | ConvertFrom-Json
    Set-Ac SUB_BUTTONS LIDACTION $s.lid; Set-Ac SUB_SLEEP STANDBYIDLE $s.standby; Set-Ac SUB_SLEEP HIBERNATEIDLE $s.hibernate
    powercfg /setactive SCHEME_CURRENT
    $g = [guid]$s.overlay; [void][LG.P]::PowerSetActiveOverlayScheme([ref]$g)
    Remove-Item $saved
  }
  & $python $nvdrs restore
}
function Get-Link { Get-NetAdapter -ErrorAction SilentlyContinue | Where-Object { $_.InterfaceDescription -match 'NCM' -and $_.Status -eq 'Up' } | Select-Object -First 1 }

if ((Test-Path $saved) -and -not (Get-Link)) { Exit-Linked }  # recover from a crash or reboot while linked

$proc = $null
$linked = $false
while ($true) {
  # the comma's USB gadget shows up as a "UsbNcm Host Device" adapter: give it the laptop's end of the link
  $a = Get-Link
  if ($a) {
    $i = $a.ifIndex
    if (-not (Get-NetIPAddress -InterfaceIndex $i -IPAddress 192.168.250.2 -PrefixLength 24 -ErrorAction SilentlyContinue)) {
      Set-NetIPInterface -InterfaceIndex $i -AddressFamily IPv4 -Dhcp Disabled -ErrorAction SilentlyContinue
      Get-NetIPAddress -InterfaceIndex $i -AddressFamily IPv4 -ErrorAction SilentlyContinue | Remove-NetIPAddress -Confirm:$false -ErrorAction SilentlyContinue
      New-NetIPAddress -InterfaceIndex $i -IPAddress 192.168.250.2 -PrefixLength 24 | Out-Null
      Set-NetConnectionProfile -InterfaceIndex $i -NetworkCategory Private -ErrorAction SilentlyContinue
    }
    Set-NetFirewallRule -Name LaptopGpu -InterfaceAlias $a.Name -ErrorAction SilentlyContinue
    if (-not $linked) { Enter-Linked; $linked = $true }
    [void][LG.P]::SetThreadExecutionState([uint32]2147483649)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED, held by this thread while it lives
  } elseif ($linked) { Exit-Linked; $linked = $false }
  if ($null -eq $proc -or $proc.HasExited) {
    $proc = Start-Process $python -ArgumentList "`"$server`"" -NoNewWindow -PassThru -RedirectStandardOutput "$root\server.log" -RedirectStandardError "$root\server.err"
  }
  Start-Sleep 5
}
