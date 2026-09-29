# Install the test-signed ToneSphere virtual audio driver. FOR A TEST VM ONLY.
#
# Run from the vm_kit folder, in an elevated PowerShell, inside a VM whose boot configuration
# has test-signing on (`bcdedit /set testsigning on`, then reboot; Secure Boot off in the
# VM's settings). It refuses to run anywhere else: a test-signed kernel driver does not
# belong on a machine anyone depends on.

$ErrorActionPreference = 'Stop'
$kit = $PSScriptRoot
$hardwareId = 'ROOT\ToneSphereVirtualAudio'

$principal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this from an elevated PowerShell.'
}

$boot = (bcdedit /enum '{current}') -join "`n"
if ($boot -notmatch 'testsigning\s+Yes') {
    throw 'Test-signing is off. This driver is test-signed and is installed only in a test VM with test-signing on.'
}

# certutil, not Import-Certificate: over PowerShell Direct or remoting, Import-Certificate is
# refused ("access denied") for LocalMachine\Root even with a full administrator token.
$cer = Join-Path $kit 'ToneSphereTestSigning.cer'
foreach ($store in 'Root', 'TrustedPublisher') {
    certutil -addstore -f $store $cer | Out-Null
    if ($LASTEXITCODE) { throw "could not add the test certificate to LocalMachine\$store ($LASTEXITCODE)" }
}
Write-Host "Trusted the test certificate (Root and TrustedPublisher)."

& (Join-Path $kit 'devcon.exe') install (Join-Path $kit 'SimpleAudioSample.inf') $hardwareId
if ($LASTEXITCODE -gt 1) { throw "devcon install failed with exit code $LASTEXITCODE" }

Start-Sleep -Seconds 3
Write-Host "`nDevice:"
Get-PnpDevice -Class MEDIA | Where-Object { $_.HardwareID -contains $hardwareId } |
    Format-Table -AutoSize Status, FriendlyName, InstanceId
Write-Host "Endpoints:"
Get-PnpDevice -Class AudioEndpoint | Where-Object { $_.FriendlyName -like '*ToneSphere*' } |
    Format-Table -AutoSize Status, FriendlyName
