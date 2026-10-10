# Runs the Windows App Certification Kit against the Store package inside a pristine test VM
# (WACK installs the package it tests, so this keeps the host clean).
#   .\vm\Test-Wack.ps1 [-Package ui\release-store\NexTone-0.1.0.appx]
param([string]$VmName = 'NexTone-DriverTest', [string]$Package)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'vmcommon.ps1')
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
if (-not $Package) { $Package = Get-ChildItem (Join-Path $root 'ui\release-store\NexTone-*.appx') | Sort-Object LastWriteTime | Select-Object -Last 1 }
$kit = 'C:\Program Files (x86)\Windows Kits\10\App Certification Kit'
Restore-TestVM -VmName $VmName
$s = New-PSSession -VMName $VmName -Credential (Get-TestVMCredential)
try {
  $taef = 'C:\Program Files (x86)\Windows Kits\10\Testing\Runtimes\TAEF'
  $meta = 'C:\Program Files (x86)\Windows Kits\10\UnionMetadata'  # platform metadata for the metadata tests
  Invoke-Command -Session $s -ArgumentList $kit, $taef, $meta { param($k, $t, $m) New-Item -ItemType Directory -Force C:\NexToneTest, $k, $t, $m | Out-Null }
  $regs = @()
  foreach ($k in 'HKLM\SOFTWARE\WOW6432Node\Microsoft\Windows Kits', 'HKLM\SOFTWARE\Microsoft\Windows Kits', 'HKLM\SOFTWARE\WOW6432Node\Microsoft\Microsoft SDKs\Windows', 'HKLM\SOFTWARE\Microsoft\Windows App Certification Kit') {
    $out = Join-Path $env:TEMP ("nextone-wack-" + $regs.Count + '.reg')
    [IO.File]::Delete($out)  # no stale export if the key is missing
    reg export $k $out /y 2>$null | Out-Null
    if (Test-Path $out) { $regs += $out }
  }
  Copy-Item -ToSession $s -Path (@($Package, "$root\packaging\msix\nextone-dev.cer") + $regs) -Destination 'C:\NexToneTest\' -Force
  # WACK finds its test runner (TAEF) by absolute path, so mirror both folders.
  Copy-Item -ToSession $s -Path "$kit\*" -Destination $kit -Recurse -Force
  Copy-Item -ToSession $s -Path "$taef\*" -Destination $taef -Recurse -Force
  Copy-Item -ToSession $s -Path "$meta\*" -Destination $meta -Recurse -Force
  Invoke-Command -Session $s -ArgumentList (Split-Path $Package -Leaf) {
    param($pkg)
    $ErrorActionPreference = 'Continue'
    Import-Certificate -FilePath C:\NexToneTest\nextone-dev.cer -CertStoreLocation Cert:\LocalMachine\TrustedPeople | Out-Null
    # WACK locates the SDK (and its platform metadata) through the Windows Kits / SDK registry.
    Get-ChildItem C:\NexToneTest\*.reg | ForEach-Object { reg import $_.FullName 2>$null }
    $cmd = "`"C:\Program Files (x86)\Windows Kits\10\App Certification Kit\appcert.exe`" test -appxpackagepath C:\NexToneTest\$pkg -reportoutputpath C:\NexToneTest\wack.xml > C:\NexToneTest\wack.txt 2>&1"
    Set-Content C:\NexToneTest\wack.cmd $cmd
    schtasks /create /f /tn NexToneWack /sc once /st 00:00 /ru nextone /it /rl highest /tr C:\NexToneTest\wack.cmd 2>$null | Out-Null
    schtasks /run /tn NexToneWack 2>$null | Out-Null
    $deadline = (Get-Date).AddMinutes(30)
    do { Start-Sleep -Seconds 10 } while (-not (Test-Path C:\NexToneTest\wack.xml) -and -not (Select-String -Quiet -Path C:\NexToneTest\wack.txt -Pattern 'Unhandled Exception' -ErrorAction SilentlyContinue) -and (Get-Date) -lt $deadline)
    Start-Sleep -Seconds 3
    Get-Content C:\NexToneTest\wack.txt -ErrorAction SilentlyContinue | Select-Object -Last 15
    if (Test-Path C:\NexToneTest\wack.xml) {
      [xml]$r = Get-Content C:\NexToneTest\wack.xml
      "OVERALL: $($r.REPORT.OVERALL_RESULT)"
      foreach ($t in $r.SelectNodes('//TEST')) {
        if ($t.RESULT.'#cdata-section' -and $t.RESULT.'#cdata-section' -ne 'PASS') {
          "[$($t.RESULT.'#cdata-section')] $($t.NAME)"
          foreach ($m in $t.SelectNodes('.//MESSAGE')) { "    $($m.TEXT)" }
        }
      }
    }
  }
  Copy-Item -FromSession $s -Path C:\NexToneTest\wack.xml -Destination (Join-Path $root 'ui\release-store\wack-report.xml') -ErrorAction SilentlyContinue
} finally {
  Remove-PSSession $s
}
