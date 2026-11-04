# Run at logon by the LaptopGpu scheduled task (setup.ps1): address the comma link, keep the server up.
$server = $null
while ($true) {
  # the comma's USB gadget appears as a "UsbNcm Host Device" adapter; give it the laptop's end of the link
  Get-NetAdapter -ErrorAction SilentlyContinue | Where-Object { $_.InterfaceDescription -match 'NCM' -and $_.Status -eq 'Up' } | ForEach-Object {
    if (-not (Get-NetIPAddress -InterfaceIndex $_.ifIndex -IPAddress 192.168.250.2 -ErrorAction SilentlyContinue)) {
      New-NetIPAddress -InterfaceIndex $_.ifIndex -IPAddress 192.168.250.2 -PrefixLength 24 | Out-Null
    }
  }
  if ($null -eq $server -or $server.HasExited) {
    $server = Start-Process wsl.exe -ArgumentList "-d Ubuntu-24.04 -u root -- sh /root/openpilot/.elantra/laptop_gpu/run.sh" -WindowStyle Hidden -PassThru
  }
  Start-Sleep 5
}
