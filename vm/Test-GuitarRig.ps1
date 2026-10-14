# Guitar Rig live scenario in a pristine test VM:
#   NexTone desktop installer + VC++ runtime + this PC's Guitar Rig 7 (set to the NexTone ASIO driver),
#   then tests/integration/guitar-rig-scenario.mjs in the VM's desktop session.
#   .\vm\Test-GuitarRig.ps1 [-Installer ui\release\NexTone-Setup-0.1.0.exe] [-Screenshot path.png]
param([string]$VmName = 'NexTone-DriverTest', [string]$Installer, [string]$Screenshot)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'vmcommon.ps1')
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
if (-not $Installer) { $Installer = Get-ChildItem (Join-Path $root 'ui\release\NexTone-Setup-*.exe') | Sort-Object LastWriteTime | Select-Object -Last 1 }
$grApp = 'C:\Program Files\Native Instruments\Guitar Rig 7'
$grContent = 'C:\Program Files\Common Files\Native Instruments\Guitar Rig 7'
if (-not (Test-Path "$grApp\Guitar Rig 7.exe")) { throw 'Guitar Rig 7 is not installed on this PC' }
$redist = Get-ChildItem 'C:\Program Files (x86)\Microsoft Visual Studio\*\BuildTools\VC\Redist\MSVC\*\vc_redist.x64.exe' |
  Sort-Object { [version]$_.VersionInfo.ProductVersion } | Select-Object -Last 1
$tmp = Join-Path $env:TEMP 'nextone-gr7'
New-Item -ItemType Directory -Force $tmp | Out-Null
reg export 'HKLM\SOFTWARE\Native Instruments\Guitar Rig 7' "$tmp\gr7-machine.reg" /y | Out-Null
reg export 'HKCU\SOFTWARE\Native Instruments\Guitar Rig 7' "$tmp\gr7-user.reg" /y | Out-Null

Restore-TestVM -VmName $VmName
$s = New-PSSession -VMName $VmName -Credential (Get-TestVMCredential)
try {
  Invoke-Command -Session $s {
    New-Item -ItemType Directory -Force C:\NexToneTest, 'C:\Program Files\Native Instruments', 'C:\Program Files\Common Files\Native Instruments' | Out-Null
  }
  $files = @($Installer, $redist.FullName, (Get-Command node).Source, "$root\build\x64-release\bin\audio-probe.exe",
    "$root\tests\integration\guitar-rig-scenario.mjs", "$root\tests\presets\vm-guitar-rig.nextone.json", "$tmp\gr7-machine.reg", "$tmp\gr7-user.reg")
  Copy-Item -ToSession $s -Path $files -Destination 'C:\NexToneTest\' -Force
  Write-Host 'Copying Guitar Rig 7 into the VM…'
  Copy-Item -ToSession $s -Path $grApp -Destination 'C:\Program Files\Native Instruments\' -Recurse -Force
  Copy-Item -ToSession $s -Path $grContent -Destination 'C:\Program Files\Common Files\Native Instruments\' -Recurse -Force

  Invoke-Command -Session $s -ArgumentList (Split-Path $Installer -Leaf), $redist.Name {
    param($setup, $redistName)
    $ErrorActionPreference = 'Continue'
    # Throwaway test VM: keep Defender from scanning the copied files during the audio checks.
    Add-MpPreference -ExclusionPath 'C:\Program Files\Native Instruments', 'C:\Program Files\Common Files\Native Instruments', 'C:\Program Files\NexTone', 'C:\NexToneTest' -ErrorAction SilentlyContinue
    $p = Start-Process "C:\NexToneTest\$redistName" -ArgumentList '/install', '/quiet', '/norestart' -Wait -PassThru
    "VC++ runtime: exit $($p.ExitCode)"
    reg import C:\NexToneTest\gr7-machine.reg 2>$null
    $p = Start-Process "C:\NexToneTest\$setup" -ArgumentList '/S' -Wait -PassThru
    "NexTone installer: exit $($p.ExitCode)"
    & 'C:\Program Files\NexTone\resources\drivers\nextone-setup.exe' status

    # Guitar Rig settings for the test user: this PC's settings, with the NexTone ASIO driver.
    $key = 'HKCU\SOFTWARE\Native Instruments\Guitar Rig 7'
    Set-Content C:\NexToneTest\gr-user.cmd @"
reg import C:\NexToneTest\gr7-user.reg
reg delete "$key" /v "Guitar Rig 7 PresetManagerDirectory" /f
reg add "$key" /v "AB2 AudioDevice" /t REG_SZ /d "ASIO NexTone ASIO" /f
reg add "$key" /v "AB2 AudioIO [ASIO NexTone ASIO]" /t REG_SZ /d "I0 0,I1 1,O0 0,O1 1," /f
reg add "$key" /v "AB2 AudioBlockSize" /t REG_DWORD /d 256 /f
reg add "$key" /v "AB2 AudioSampleRate" /t REG_DWORD /d 48000 /f
echo done> C:\NexToneTest\gr-user.done
"@
    schtasks /create /f /tn NexToneGrUser /sc once /st 00:00 /ru nextone /it /rl highest /tr C:\NexToneTest\gr-user.cmd 2>$null | Out-Null
    schtasks /run /tn NexToneGrUser 2>$null | Out-Null
    $deadline = (Get-Date).AddSeconds(60)
    do { Start-Sleep -Seconds 1 } while (-not (Test-Path C:\NexToneTest\gr-user.done) -and (Get-Date) -lt $deadline)

    # Start NexTone like a user (the app starts the engine).
    Set-Content C:\NexToneTest\launch.cmd '@start "" "C:\Program Files\NexTone\NexTone.exe"'
    schtasks /create /f /tn NexToneApp /sc once /st 00:00 /ru nextone /it /rl highest /tr C:\NexToneTest\launch.cmd 2>$null | Out-Null
    schtasks /run /tn NexToneApp 2>$null | Out-Null
    $deadline = (Get-Date).AddSeconds(120)
    do { Start-Sleep -Seconds 2 } while (-not (Test-Path C:\Users\nextone\AppData\Local\NexTone\engine.json) -and (Get-Date) -lt $deadline)

    # Wait (up to 5 min) for first-boot background work to settle: audio checks need a calm machine.
    $deadline = (Get-Date).AddMinutes(5)
    do {
      $cpu = (Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 3 -MaxSamples 1).CounterSamples[0].CookedValue
    } while ($cpu -gt 25 -and (Get-Date) -lt $deadline)
    "VM CPU before the audio checks: {0:N0}%" -f $cpu
    $cmd = 'cmd /c C:\NexToneTest\node.exe C:\NexToneTest\guitar-rig-scenario.mjs C:\NexToneTest\vm-guitar-rig.nextone.json C:\NexToneTest\audio-probe.exe > C:\NexToneTest\scenario.txt 2>&1'
    schtasks /create /f /tn NexToneScenario /sc once /st 00:00 /ru nextone /it /rl highest /tr $cmd 2>$null | Out-Null
    schtasks /run /tn NexToneScenario 2>$null | Out-Null
    $deadline = (Get-Date).AddMinutes(12)
    do { Start-Sleep -Seconds 5 } while (-not (Test-Path C:\NexToneTest\guitar-rig-report.json) -and (Get-Date) -lt $deadline)
    Get-Content C:\NexToneTest\scenario.txt -ErrorAction SilentlyContinue
  }
  if ($Screenshot) {
    # Screenshot of the VM's desktop session (taken inside the VM).
    Invoke-Command -Session $s {
      Set-Content C:\NexToneTest\shot.ps1 @'
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
$b = [System.Windows.Forms.SystemInformation]::VirtualScreen
$bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
[System.Drawing.Graphics]::FromImage($bmp).CopyFromScreen($b.Left, $b.Top, 0, 0, $bmp.Size)
$bmp.Save('C:\NexToneTest\screen.png', [System.Drawing.Imaging.ImageFormat]::Png)
'@
      schtasks /create /f /tn NexToneShot /sc once /st 00:00 /ru nextone /it /tr 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\NexToneTest\shot.ps1' 2>$null | Out-Null
      schtasks /run /tn NexToneShot 2>$null | Out-Null
      $d = (Get-Date).AddSeconds(20); do { Start-Sleep 1 } while (-not (Test-Path C:\NexToneTest\screen.png) -and (Get-Date) -lt $d)
    }
    Copy-Item -FromSession $s -Path C:\NexToneTest\screen.png -Destination $Screenshot -ErrorAction SilentlyContinue
  }
} finally {
  Remove-PSSession $s
}
