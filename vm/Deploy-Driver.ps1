# Installs the freshly built NexTone VAD driver package inside the Hyper-V test VM and
# reports the audio endpoints it created. Never touches the host.
#   .\vm\Deploy-Driver.ps1            # install / update
#   .\vm\Deploy-Driver.ps1 -Revert    # first revert the VM to the 'clean' checkpoint
#   .\vm\Deploy-Driver.ps1 -Verifier  # also enable Driver Verifier for nextonevad.sys (reboots the VM)
param([string]$VmName = 'NexTone-DriverTest', [switch]$Revert, [switch]$Verifier)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'vmcommon.ps1')
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
$pkg = Join-Path $root 'drivers\win\vad\build\package'
$devcon = "${env:ProgramFiles(x86)}\Windows Kits\10\Tools\10.0.26100.0\x64\devcon.exe"
if (-not (Test-Path (Join-Path $pkg 'nextonevad.sys'))) { throw 'Build the driver first: drivers\win\vad\build.ps1' }

if ($Revert) { Restore-TestVM -VmName $VmName }
if ((Get-VM -Name $VmName).State -ne 'Running') { Start-VM -Name $VmName; Wait-TestVM -VmName $VmName }

$s = New-PSSession -VMName $VmName -Credential (Get-TestVMCredential)
try {
  Invoke-Command -Session $s { New-Item -ItemType Directory -Force C:\NexTone\driver | Out-Null }
  Copy-Item -ToSession $s -Path (Join-Path $pkg '*') -Destination 'C:\NexTone\driver' -Force
  Copy-Item -ToSession $s -Path $devcon -Destination 'C:\NexTone\devcon.exe' -Force
  Invoke-Command -Session $s -ArgumentList $Verifier.IsPresent {
    param($verifier)
    $ErrorActionPreference = 'Continue'
    $cer = 'C:\NexTone\driver\NexToneTest.cer'
    Import-Certificate -FilePath $cer -CertStoreLocation Cert:\LocalMachine\Root | Out-Null
    Import-Certificate -FilePath $cer -CertStoreLocation Cert:\LocalMachine\TrustedPublisher | Out-Null
    $existing = Get-PnpDevice -ErrorAction SilentlyContinue | Where-Object { $_.HardwareID -contains 'ROOT\NexToneVad' }
    if ($existing) {
      & C:\NexTone\devcon.exe update C:\NexTone\driver\nextonevad.inf ROOT\NexToneVad
    } else {
      & C:\NexTone\devcon.exe install C:\NexTone\driver\nextonevad.inf ROOT\NexToneVad
    }
    "devcon exit: $LASTEXITCODE"
    if ($verifier) { verifier /standard /driver nextonevad.sys | Out-Null; 'Driver Verifier enabled (takes effect after reboot)' }
    Start-Sleep -Seconds 4
    '--- device'
    Get-PnpDevice -ErrorAction SilentlyContinue | Where-Object { $_.HardwareID -contains 'ROOT\NexToneVad' } |
      Select-Object Status, Class, FriendlyName, Problem | Format-Table -AutoSize | Out-String -Width 200
    '--- audio endpoints'
    Get-PnpDevice -Class AudioEndpoint -ErrorAction SilentlyContinue | Select-Object Status, FriendlyName |
      Format-Table -AutoSize | Out-String -Width 200
    '--- driver service'
    sc.exe query NexToneVad | Select-String STATE
  }
} finally {
  Remove-PSSession $s
}
