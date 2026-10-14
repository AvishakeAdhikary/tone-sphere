# Soak test for the NexTone virtual audio driver under Driver Verifier, in the test VM only.
# Installs the driver in a pristine VM with Driver Verifier (standard checks) on, reboots the VM,
# then for -Minutes repeats: the full virtual-device audio test, followed by a device
# disable/enable cycle (hot unplug while the engine is streaming). Finally reports bugchecks,
# crash dumps and Driver Verifier statistics.
#   .\vm\Test-Soak.ps1 [-Minutes 60] [-Seconds 20]
param([string]$VmName = 'NexTone-DriverTest', [int]$Minutes = 60, [int]$Seconds = 20)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'vmcommon.ps1')
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
$bin = Join-Path $root 'build\x64-release\bin'

& (Join-Path $PSScriptRoot 'Deploy-Driver.ps1') -VmName $VmName -Revert -Verifier | Select-String 'devcon exit|Verifier'
Restart-VM -Name $VmName -Force -Wait -For Heartbeat
Wait-TestVM -VmName $VmName
Start-Sleep -Seconds 20  # let the user session come up

$s = New-PSSession -VMName $VmName -Credential (Get-TestVMCredential)
try {
  Invoke-Command -Session $s { New-Item -ItemType Directory -Force C:\NexTone\bin | Out-Null }
  foreach ($f in @("$bin\nextone-engine.exe", "$bin\audio-probe.exe", (Get-Command node).Source, "$root\tests\integration\vad-audio.mjs")) {
    Copy-Item -ToSession $s -Path $f -Destination 'C:\NexTone\bin\' -Force
  }
  Invoke-Command -Session $s {
    '--- verifier settings'
    verifier /querysettings | Select-String 'nextonevad|Standard|Special pool|Force IRQL|Pool Tracking' | Select-Object -First 6
    schtasks /create /f /tn NexToneEngine /sc once /st 00:00 /ru nextone /it /rl highest /tr 'C:\NexTone\bin\nextone-engine.exe' 2>$null | Out-Null
    schtasks /run /tn NexToneEngine 2>$null | Out-Null
    Start-Sleep -Seconds 5
  }
  $end = (Get-Date).AddMinutes($Minutes)
  $iter = 0; $passed = 0; $failures = @()
  while ((Get-Date) -lt $end) {
    $iter++
    $r = Invoke-Command -Session $s -ArgumentList $Seconds, $iter {
      param($secs, $n)
      $ErrorActionPreference = 'Continue'
      [IO.File]::Delete('C:\NexTone\bin\vad-audio-report.json')
      $cmd = "cmd /c C:\NexTone\bin\node.exe C:\NexTone\bin\vad-audio.mjs C:\NexTone\bin\audio-probe.exe $secs > C:\NexTone\bin\soak-$n.txt 2>&1"
      schtasks /create /f /tn NexToneSoak /sc once /st 00:00 /ru nextone /it /rl highest /tr $cmd 2>$null | Out-Null
      schtasks /run /tn NexToneSoak 2>$null | Out-Null
      $deadline = (Get-Date).AddMinutes(4)
      do { Start-Sleep -Seconds 2 } while (-not (Test-Path C:\NexTone\bin\vad-audio-report.json) -and (Get-Date) -lt $deadline)
      Start-Sleep -Seconds 1
      $summary = (Get-Content "C:\NexTone\bin\soak-$n.txt" -ErrorAction SilentlyContinue | Select-String 'passed$' | Select-Object -Last 1).Line
      $fails = Get-Content "C:\NexTone\bin\soak-$n.txt" -ErrorAction SilentlyContinue | Select-String '^FAIL' | ForEach-Object Line
      # Hot unplug while the engine streams, then plug back in.
      $dev = Get-PnpDevice -ErrorAction SilentlyContinue | Where-Object { $_.HardwareID -contains 'ROOT\NexToneVad' }
      Disable-PnpDevice -InstanceId $dev.InstanceId -Confirm:$false -ErrorAction SilentlyContinue
      Start-Sleep -Seconds 3
      Enable-PnpDevice -InstanceId $dev.InstanceId -Confirm:$false -ErrorAction SilentlyContinue
      Start-Sleep -Seconds 6
      $engine = [bool](Get-Process nextone-engine -ErrorAction SilentlyContinue)
      [pscustomobject]@{ Summary = $summary; Fails = $fails; DeviceOk = ((Get-PnpDevice -InstanceId $dev.InstanceId).Status -eq 'OK'); Engine = $engine }
    }
    $ok = $r.Summary -match '^(\d+)/\1 passed' -and $r.DeviceOk -and $r.Engine
    if ($ok) { $passed++ } else { $failures += "iteration ${iter}: $($r.Summary) device=$($r.DeviceOk) engine=$($r.Engine) $($r.Fails -join '; ')" }
    Write-Host ("[{0:HH:mm}] iteration {1}: {2} (device {3}, engine {4})" -f (Get-Date), $iter, $r.Summary, $(if ($r.DeviceOk) { 'ok' } else { 'FAILED' }), $(if ($r.Engine) { 'running' } else { 'GONE' }))
  }
  '--- result'
  "iterations: $iter, fully passed: $passed"
  $failures
  Invoke-Command -Session $s {
    '--- bugchecks / crash dumps'
    $bc = Get-WinEvent -FilterHashtable @{ LogName = 'System'; Id = 1001 } -ErrorAction SilentlyContinue | Where-Object ProviderName -match 'BugCheck|WER-SystemErrorReporting'
    "bugcheck events: $(@($bc).Count)"
    "MEMORY.DMP present: $(Test-Path C:\Windows\MEMORY.DMP)"
    "minidumps: $(@(Get-ChildItem C:\Windows\Minidump -ErrorAction SilentlyContinue).Count)"
    '--- verifier statistics'
    verifier /query | Select-String 'nextonevad|Pool|IRQL|Faults|Violations|CurrentPagedPool|Peak' | Select-Object -First 15
  }
} finally {
  Remove-PSSession $s -ErrorAction SilentlyContinue
}
