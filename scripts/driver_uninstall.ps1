# Remove the ToneSphere virtual audio driver completely: the device, the driver package in
# the driver store, and the trust in its test certificate. Elevated PowerShell, from vm_kit.
#
# A virtual audio driver that cannot be removed cleanly is worse than none, so this checks
# afterwards that nothing is left and says so if something is.

$ErrorActionPreference = 'Stop'
$kit = $PSScriptRoot
$hardwareId = 'ROOT\ToneSphereVirtualAudio'

# Each cable is stopped first, refusing rather than deferring while the audio engine has it
# open: removing one cable makes another the default device, the audio engine opens that for
# a moment, and a removal that meets it is otherwise put off until a reboot.
Add-Type -Namespace ToneSphere -Name CfgMgr -MemberDefinition @'
[DllImport("cfgmgr32.dll", CharSet = CharSet.Unicode)]
public static extern int CM_Locate_DevNodeW(out uint devInst, string instanceId, uint flags);
[DllImport("cfgmgr32.dll")]
public static extern int CM_Disable_DevNode(uint devInst, uint flags);
'@
$cables = @(Get-PnpDevice -Class MEDIA -ErrorAction SilentlyContinue | Where-Object { $_.HardwareID -contains $hardwareId })
foreach ($cable in $cables) {
    $devInst = 0
    if ([ToneSphere.CfgMgr]::CM_Locate_DevNodeW([ref]$devInst, $cable.InstanceId, 0) -ne 0) { continue }
    $deadline = (Get-Date).AddSeconds(15)
    do {
        $cr = [ToneSphere.CfgMgr]::CM_Disable_DevNode($devInst, 0x4)   # CM_DISABLE_UI_NOT_OK
        if ($cr -ne 0x17) { break }                                     # CR_REMOVE_VETOED
        Start-Sleep -Milliseconds 500
    } while ((Get-Date) -lt $deadline)
    if ($cr -ne 0) { Write-Warning "$($cable.InstanceId) could not be stopped first (CONFIGRET $cr); it may need a reboot to go" }
}

& (Join-Path $kit 'devcon.exe') remove $hardwareId

# The published name (oemNN.inf) of every ToneSphere package in the driver store.
$published = @()
$current = @{}
foreach ($line in (pnputil /enum-drivers)) {
    if ($line -match '^\s*Published Name\s*:\s*(\S+)') { $current = @{ Published = $Matches[1] } }
    elseif ($line -match '^\s*Original Name\s*:\s*(\S+)') { $current.Original = $Matches[1] }
    elseif ($line -match '^\s*Provider Name\s*:\s*(.+)$') {
        $current.Provider = $Matches[1].Trim()
        if ($current.Original -eq 'simpleaudiosample.inf' -and $current.Provider -eq 'Neural Nexus Studios') {
            $published += $current.Published
        }
    }
}
foreach ($inf in $published) {
    pnputil /delete-driver $inf /uninstall /force | Out-Host
}

$cer = New-Object System.Security.Cryptography.X509Certificates.X509Certificate2 (Join-Path $kit 'ToneSphereTestSigning.cer')
foreach ($store in 'Root', 'TrustedPublisher') {
    if (Get-ChildItem "Cert:\LocalMachine\$store" | Where-Object { $_.Thumbprint -eq $cer.Thumbprint }) {
        certutil -delstore $store $cer.Thumbprint | Out-Null
        if ($LASTEXITCODE) { throw "could not remove the test certificate from LocalMachine\$store ($LASTEXITCODE)" }
    }
}

Start-Sleep -Seconds 3
$left = @(Get-PnpDevice -Class AudioEndpoint -ErrorAction SilentlyContinue |
          Where-Object { $_.FriendlyName -like '*ToneSphere*' -and $_.Status -eq 'OK' })
$left += @(Get-PnpDevice -Class MEDIA -ErrorAction SilentlyContinue |
           Where-Object { $_.HardwareID -contains $hardwareId })
if ($left.Count) {
    $left | Format-Table -AutoSize Status, FriendlyName, InstanceId
    throw 'Something was left behind (listed above).'
}
Write-Host 'Removed: no ToneSphere device or endpoint remains, and the test certificate is no longer trusted.'
