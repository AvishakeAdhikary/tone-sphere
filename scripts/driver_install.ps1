# Install the test-signed ToneSphere virtual audio driver. FOR A TEST VM ONLY.
#
# This trusts the test certificate and adds the driver package to the driver store; it does
# not create cables. Cables are created by ToneSphere itself, the way a user creates them:
#   uv run python main.py cable-admin install-cables      (the two default cables)
#   uv run python main.py cable-admin add "Name"           (one more)
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

pnputil /add-driver (Join-Path $kit 'SimpleAudioSample.inf') | Out-Host
if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 3010) { throw "pnputil /add-driver failed with exit code $LASTEXITCODE" }
Write-Host "The driver package is in the driver store ($hardwareId); create cables with ToneSphere."
