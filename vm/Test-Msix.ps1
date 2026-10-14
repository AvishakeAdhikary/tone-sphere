# Sideload-tests the Store package in a pristine test VM: trusts the dev certificate,
# installs the MSIX, launches NexTone from its tile like a user would, and checks the engine.
#   .\vm\Test-Msix.ps1 [-Package ui\release-store\NexTone-0.1.0.appx]
param([string]$VmName = 'NexTone-DriverTest', [string]$Package)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'vmcommon.ps1')
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
if (-not $Package) { $Package = Get-ChildItem (Join-Path $root 'ui\release-store\NexTone-*.appx') | Sort-Object LastWriteTime | Select-Object -Last 1 }
Restore-TestVM -VmName $VmName
$s = New-PSSession -VMName $VmName -Credential (Get-TestVMCredential)
try {
  Invoke-Command -Session $s { New-Item -ItemType Directory -Force C:\NexToneTest | Out-Null }
  foreach ($f in @($Package, "$root\packaging\msix\nextone-dev.cer", (Get-Command node).Source, "$root\tools\nextone-ctl.mjs")) {
    Copy-Item -ToSession $s -Path $f -Destination 'C:\NexToneTest\' -Force
  }
  Invoke-Command -Session $s -ArgumentList (Split-Path $Package -Leaf) {
    param($pkg)
    $ErrorActionPreference = 'Continue'
    Import-Certificate -FilePath C:\NexToneTest\nextone-dev.cer -CertStoreLocation Cert:\LocalMachine\TrustedPeople | Out-Null
    # Install for the test user (Add-AppxPackage registers per user, so run it in their session).
    $cmd = "powershell -NoProfile -Command `"Add-AppxPackage C:\NexToneTest\$pkg; (Get-AppxPackage NexTone.NexTone).PackageFamilyName | Out-File C:\NexToneTest\pfn.txt; Start-Process ('shell:AppsFolder\' + (Get-AppxPackage NexTone.NexTone).PackageFamilyName + '!NexTone')`" > C:\NexToneTest\install.txt 2>&1"
    Set-Content C:\NexToneTest\install.cmd $cmd
    schtasks /create /f /tn NexToneMsix /sc once /st 00:00 /ru nextone /it /rl highest /tr C:\NexToneTest\install.cmd 2>$null | Out-Null
    schtasks /run /tn NexToneMsix 2>$null | Out-Null
    $deadline = (Get-Date).AddSeconds(180)
    $found = $null
    do {
      Start-Sleep -Seconds 3
      $found = Get-ChildItem -Recurse -ErrorAction SilentlyContinue -Filter engine.json C:\Users\nextone\AppData\Local | Select-Object -First 1
    } while (-not $found -and (Get-Date) -lt $deadline)
    Get-Content C:\NexToneTest\install.txt -ErrorAction SilentlyContinue
    "package: $(Get-Content C:\NexToneTest\pfn.txt -ErrorAction SilentlyContinue)"
    "discovery: $($found.FullName)"
    Get-Process -ErrorAction SilentlyContinue | Where-Object Name -like '*exTone*' | Select-Object Name, SessionId, Path -Unique | Format-Table -AutoSize | Out-String -Width 300
    if ($found) {
      Start-Sleep -Seconds 3
      '--- nextone-ctl (finds the packaged engine by itself) ---'
      Copy-Item C:\NexToneTest\nextone-ctl.mjs C:\NexToneTest\ctl.mjs
      Set-Content C:\NexToneTest\ctl.cmd 'C:\NexToneTest\node.exe C:\NexToneTest\ctl.mjs state > C:\NexToneTest\ctl.txt 2>&1'
      schtasks /create /f /tn NexToneCtl /sc once /st 00:00 /ru nextone /it /rl highest /tr C:\NexToneTest\ctl.cmd 2>$null | Out-Null
      schtasks /run /tn NexToneCtl 2>$null | Out-Null
      Start-Sleep -Seconds 6
      Get-Content C:\NexToneTest\ctl.txt -ErrorAction SilentlyContinue | Select-Object -First 12
    }
    '--- protocol + alias registration ---'
    "nextone: protocol registered: $(Test-Path Registry::HKEY_CLASSES_ROOT\nextone)"
    "alias present: $(Test-Path C:\Users\nextone\AppData\Local\Microsoft\WindowsApps\nextone-engine.exe)"
    '--- simulated sign-in: stop everything, run the startup entry (engine, no arguments) ---'
    Get-Process -ErrorAction SilentlyContinue | Where-Object Name -like '*exTone*' | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    Set-Content C:\NexToneTest\signin.cmd '@start "" C:\Users\nextone\AppData\Local\Microsoft\WindowsApps\nextone-engine.exe'
    schtasks /create /f /tn NexToneSignin /sc once /st 00:00 /ru nextone /it /rl highest /tr C:\NexToneTest\signin.cmd 2>$null | Out-Null
    schtasks /run /tn NexToneSignin 2>$null | Out-Null
    Start-Sleep -Seconds 20
    Get-CimInstance Win32_Process | Where-Object Name -like '*exTone*' | Select-Object Name, CommandLine | Format-Table -AutoSize | Out-String -Width 300
  }
} finally {
  Remove-PSSession $s
}
