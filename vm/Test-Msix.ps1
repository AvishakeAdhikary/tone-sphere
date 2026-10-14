# Sideload-tests the Store package in a pristine test VM: trusts the dev certificate,
# installs the MSIX, launches ToneSphere from its tile like a user would, and checks the engine.
#   .\vm\Test-Msix.ps1 [-Package ui\release-store\ToneSphere-0.1.0.appx]
param([string]$VmName = 'ToneSphere-DriverTest', [string]$Package)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'vmcommon.ps1')
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
if (-not $Package) { $Package = Get-ChildItem (Join-Path $root 'ui\release-store\ToneSphere-*.appx') | Sort-Object LastWriteTime | Select-Object -Last 1 }
Restore-TestVM -VmName $VmName
$s = New-PSSession -VMName $VmName -Credential (Get-TestVMCredential)
try {
  Invoke-Command -Session $s { New-Item -ItemType Directory -Force C:\ToneSphereTest | Out-Null }
  foreach ($f in @($Package, "$root\packaging\msix\tonesphere-dev.cer", (Get-Command node).Source, "$root\tools\tonesphere-ctl.mjs")) {
    Copy-Item -ToSession $s -Path $f -Destination 'C:\ToneSphereTest\' -Force
  }
  Invoke-Command -Session $s -ArgumentList (Split-Path $Package -Leaf) {
    param($pkg)
    $ErrorActionPreference = 'Continue'
    Import-Certificate -FilePath C:\ToneSphereTest\tonesphere-dev.cer -CertStoreLocation Cert:\LocalMachine\TrustedPeople | Out-Null
    # Install for the test user (Add-AppxPackage registers per user, so run it in their session).
    $cmd = "powershell -NoProfile -Command `"Add-AppxPackage C:\ToneSphereTest\$pkg; (Get-AppxPackage ToneSphere.ToneSphere).PackageFamilyName | Out-File C:\ToneSphereTest\pfn.txt; Start-Process ('shell:AppsFolder\' + (Get-AppxPackage ToneSphere.ToneSphere).PackageFamilyName + '!ToneSphere')`" > C:\ToneSphereTest\install.txt 2>&1"
    Set-Content C:\ToneSphereTest\install.cmd $cmd
    schtasks /create /f /tn ToneSphereMsix /sc once /st 00:00 /ru tonesphere /it /rl highest /tr C:\ToneSphereTest\install.cmd 2>$null | Out-Null
    schtasks /run /tn ToneSphereMsix 2>$null | Out-Null
    $deadline = (Get-Date).AddSeconds(180)
    $found = $null
    do {
      Start-Sleep -Seconds 3
      $found = Get-ChildItem -Recurse -ErrorAction SilentlyContinue -Filter engine.json C:\Users\tonesphere\AppData\Local | Select-Object -First 1
    } while (-not $found -and (Get-Date) -lt $deadline)
    Get-Content C:\ToneSphereTest\install.txt -ErrorAction SilentlyContinue
    "package: $(Get-Content C:\ToneSphereTest\pfn.txt -ErrorAction SilentlyContinue)"
    "discovery: $($found.FullName)"
    Get-Process -ErrorAction SilentlyContinue | Where-Object Name -like '*exTone*' | Select-Object Name, SessionId, Path -Unique | Format-Table -AutoSize | Out-String -Width 300
    if ($found) {
      Start-Sleep -Seconds 3
      '--- tonesphere-ctl (finds the packaged engine by itself) ---'
      Copy-Item C:\ToneSphereTest\tonesphere-ctl.mjs C:\ToneSphereTest\ctl.mjs
      Set-Content C:\ToneSphereTest\ctl.cmd 'C:\ToneSphereTest\node.exe C:\ToneSphereTest\ctl.mjs state > C:\ToneSphereTest\ctl.txt 2>&1'
      schtasks /create /f /tn ToneSphereCtl /sc once /st 00:00 /ru tonesphere /it /rl highest /tr C:\ToneSphereTest\ctl.cmd 2>$null | Out-Null
      schtasks /run /tn ToneSphereCtl 2>$null | Out-Null
      Start-Sleep -Seconds 6
      Get-Content C:\ToneSphereTest\ctl.txt -ErrorAction SilentlyContinue | Select-Object -First 12
    }
    '--- protocol + alias registration ---'
    "tonesphere: protocol registered: $(Test-Path Registry::HKEY_CLASSES_ROOT\tonesphere)"
    "alias present: $(Test-Path C:\Users\tonesphere\AppData\Local\Microsoft\WindowsApps\tonesphere-engine.exe)"
    '--- simulated sign-in: stop everything, run the startup entry (engine, no arguments) ---'
    Get-Process -ErrorAction SilentlyContinue | Where-Object Name -like '*exTone*' | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    Set-Content C:\ToneSphereTest\signin.cmd '@start "" C:\Users\tonesphere\AppData\Local\Microsoft\WindowsApps\tonesphere-engine.exe'
    schtasks /create /f /tn ToneSphereSignin /sc once /st 00:00 /ru tonesphere /it /rl highest /tr C:\ToneSphereTest\signin.cmd 2>$null | Out-Null
    schtasks /run /tn ToneSphereSignin 2>$null | Out-Null
    Start-Sleep -Seconds 20
    Get-CimInstance Win32_Process | Where-Object Name -like '*exTone*' | Select-Object Name, CommandLine | Format-Table -AutoSize | Out-String -Width 300
  }
} finally {
  Remove-PSSession $s
}
