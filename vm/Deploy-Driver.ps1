# Installs the freshly built ToneSphere VAD driver package inside the Hyper-V test VM and
# reports the audio endpoints it created. Never touches the host.
#   .\vm\Deploy-Driver.ps1            # install / update
#   .\vm\Deploy-Driver.ps1 -Revert    # first revert the VM to the 'clean' checkpoint
#   .\vm\Deploy-Driver.ps1 -Verifier  # also enable Driver Verifier for tonespherevad.sys (reboots the VM)
param([string]$VmName = 'ToneSphere-DriverTest', [switch]$Revert, [switch]$Verifier)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'vmcommon.ps1')
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
$pkg = Join-Path $root 'drivers\win\vad\build\package'
$devcon = "${env:ProgramFiles(x86)}\Windows Kits\10\Tools\10.0.26100.0\x64\devcon.exe"
if (-not (Test-Path (Join-Path $pkg 'tonespherevad.sys'))) { throw 'Build the driver first: drivers\win\vad\build.ps1' }

if ($Revert) { Restore-TestVM -VmName $VmName }
if ((Get-VM -Name $VmName).State -ne 'Running') { Start-VM -Name $VmName; Wait-TestVM -VmName $VmName }

$s = New-PSSession -VMName $VmName -Credential (Get-TestVMCredential)
try {
  Invoke-Command -Session $s { New-Item -ItemType Directory -Force C:\ToneSphere\driver | Out-Null }
  Copy-Item -ToSession $s -Path (Join-Path $pkg '*') -Destination 'C:\ToneSphere\driver' -Force
  Copy-Item -ToSession $s -Path $devcon -Destination 'C:\ToneSphere\devcon.exe' -Force
  Invoke-Command -Session $s -ArgumentList $Verifier.IsPresent {
    param($verifier)
    $ErrorActionPreference = 'Continue'
    $cer = 'C:\ToneSphere\driver\ToneSphereTest.cer'
    Import-Certificate -FilePath $cer -CertStoreLocation Cert:\LocalMachine\Root | Out-Null
    Import-Certificate -FilePath $cer -CertStoreLocation Cert:\LocalMachine\TrustedPublisher | Out-Null
    $existing = Get-PnpDevice -ErrorAction SilentlyContinue | Where-Object { $_.HardwareID -contains 'ROOT\ToneSphereVad' }
    if ($existing) {
      & C:\ToneSphere\devcon.exe update C:\ToneSphere\driver\tonespherevad.inf ROOT\ToneSphereVad
    } else {
      & C:\ToneSphere\devcon.exe install C:\ToneSphere\driver\tonespherevad.inf ROOT\ToneSphereVad
    }
    "devcon exit: $LASTEXITCODE"
    if ($verifier) { verifier /standard /driver tonespherevad.sys | Out-Null; 'Driver Verifier enabled (takes effect after reboot)' }
    Start-Sleep -Seconds 4
    '--- device'
    Get-PnpDevice -ErrorAction SilentlyContinue | Where-Object { $_.HardwareID -contains 'ROOT\ToneSphereVad' } |
      Select-Object Status, Class, FriendlyName, Problem | Format-Table -AutoSize | Out-String -Width 200
    '--- audio endpoints'
    Get-PnpDevice -Class AudioEndpoint -ErrorAction SilentlyContinue | Select-Object Status, FriendlyName |
      Format-Table -AutoSize | Out-String -Width 200
    '--- driver service'
    sc.exe query ToneSphereVad | Select-String STATE
  }
} finally {
  Remove-PSSession $s
}
