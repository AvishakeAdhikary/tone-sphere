# ToneSphere hosting ASIO4ALL as the interface driver, in the test VM. Run right after
# Test-VbCable.ps1 (it needs that VM: ToneSphere installed, VB-CABLE installed): ASIO4ALL 2.22 is
# downloaded from asio4all.org and installed, the VM restarts (VB-CABLE asks for it), and
# tests/integration/asio-host.mjs runs with "ASIO4ALL v2": engine tone -> ASIO4ALL out ->
# VB-CABLE -> ASIO4ALL in, and two DAWs sharing the interface through ToneSphere ASIO.
#   .\vm\Test-VbCable.ps1 ; .\vm\Test-Asio4All.ps1
param([string]$VmName = 'ToneSphere-DriverTest', [int]$Seconds = 8)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'vmcommon.ps1')
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
$cred = Get-TestVMCredential
$s = New-PSSession -VMName $VmName -Credential $cred
Copy-Item -ToSession $s -Path "$root\build\x64-release\bin\asio-test-host.exe" -Destination 'C:\ToneSphereTest\build\x64-release\bin\' -Force
Copy-Item -ToSession $s -Path "$root\tests\integration\asio-host.mjs" -Destination 'C:\ToneSphereTest\tests\integration\' -Force
Invoke-Command -Session $s {
  $ErrorActionPreference = 'Continue'
  Invoke-WebRequest 'https://asio4all.org/downloads/ASIO4ALL_2_22.exe' -OutFile C:\ToneSphereTest\asio4all.exe -UseBasicParsing
  "ASIO4ALL signature: $((Get-AuthenticodeSignature C:\ToneSphereTest\asio4all.exe).Status)"
  $p = Start-Process C:\ToneSphereTest\asio4all.exe -ArgumentList '/S' -Wait -PassThru
  "ASIO4ALL installer exit code: $($p.ExitCode); registered: $(Test-Path 'HKLM:\SOFTWARE\ASIO\ASIO4ALL v2')"
}
Remove-PSSession $s
Restart-VM -Name $VmName -Force
Wait-TestVMSettled -VmName $VmName -Minutes 2
Invoke-Command -VMName $VmName -Credential $cred -ArgumentList $Seconds {
  param($secs)
  # Seen here: after the restart ASIO4ALL's registration was gone (files still installed).
  # Register it the way its own installer does.
  if (-not (Test-Path 'HKLM:\SOFTWARE\ASIO\ASIO4ALL v2')) {
    $dir = 'C:\Program Files (x86)\ASIO4ALL v2'
    Start-Process "$env:SystemRoot\System32\regsvr32.exe" -ArgumentList '/s', "`"$dir\asio4all64.dll`"" -Wait
    Start-Process "$env:SystemRoot\SysWOW64\regsvr32.exe" -ArgumentList '/s', "`"$dir\asio4all.dll`"" -Wait
    "ASIO4ALL registration was missing after the restart; re-registered: $(Test-Path 'HKLM:\SOFTWARE\ASIO\ASIO4ALL v2')"
  }
  schtasks /run /tn ToneSphereApp 2>$null | Out-Null
  $disc = 'C:\Users\tonesphere\AppData\Local\ToneSphere\engine.json'
  $deadline = (Get-Date).AddSeconds(90)
  do { Start-Sleep -Seconds 3 } while (-not (Test-Path $disc) -and (Get-Date) -lt $deadline)
  Start-Sleep -Seconds 5
  $cmd = "cmd /c C:\ToneSphereTest\node.exe C:\ToneSphereTest\tests\integration\asio-host.mjs `"ASIO4ALL v2`" $secs > C:\ToneSphereTest\asio4all.txt 2>&1"
  Set-Content C:\ToneSphereTest\asio4all.cmd "@$cmd"
  schtasks /create /f /tn ToneSphereAsio4All /sc once /st 00:00 /ru tonesphere /it /rl highest /tr C:\ToneSphereTest\asio4all.cmd 2>$null | Out-Null
  schtasks /run /tn ToneSphereAsio4All 2>$null | Out-Null
  $deadline = (Get-Date).AddMinutes(5)
  do { Start-Sleep -Seconds 3 } while (-not ((Get-Content C:\ToneSphereTest\asio4all.txt -Raw -ErrorAction SilentlyContinue) -match '\d+/\d+ passed') -and (Get-Date) -lt $deadline)
  '--- asio-host with ASIO4ALL'; Get-Content C:\ToneSphereTest\asio4all.txt
}
