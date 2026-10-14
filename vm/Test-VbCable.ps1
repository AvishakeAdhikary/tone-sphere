# The public desktop download on a pristine Windows, then "Install drivers" with VB-CABLE:
#  1. silent install of the public installer (no virtual audio driver inside), launched like a user;
#  2. the official VB-CABLE package downloaded from vb-audio.com, as the app does;
#  3. `tonesphere-setup install-all --vbcable` (what Settings > Devices > Install drivers runs after
#     its UAC prompt): VB-Audio's signature is checked, VB-Audio's installer opens, and
#     vbcable-click.ps1 plays the user (clicks "Install Driver");
#  4. the running engine picks up VB-CABLE by itself, and audio passes both ways through it
#     (tests/integration/host-audio.mjs, measured with audio-probe).
#   .\vm\Test-VbCable.ps1 [-Installer ui\release\ToneSphere-Setup-<version>.exe]
param([string]$VmName = 'ToneSphere-DriverTest', [string]$Installer, [int]$Seconds = 8)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'vmcommon.ps1')
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
if (-not $Installer) { $Installer = Get-ChildItem (Join-Path $root 'ui\release\ToneSphere-Setup-*.exe') | Sort-Object LastWriteTime | Select-Object -Last 1 }
Restore-TestVM -VmName $VmName
$s = New-PSSession -VMName $VmName -Credential (Get-TestVMCredential)
try {
  Invoke-Command -Session $s { New-Item -ItemType Directory -Force C:\ToneSphereTest\build\x64-release\bin, C:\ToneSphereTest\tests\integration | Out-Null }
  Copy-Item -ToSession $s -Path $Installer, (Get-Command node).Source, "$PSScriptRoot\vbcable-click.ps1" -Destination 'C:\ToneSphereTest\' -Force
  Copy-Item -ToSession $s -Path "$root\build\x64-release\bin\audio-probe.exe" -Destination 'C:\ToneSphereTest\build\x64-release\bin\' -Force
  Copy-Item -ToSession $s -Path "$root\tests\integration\host-audio.mjs" -Destination 'C:\ToneSphereTest\tests\integration\' -Force
  Invoke-Command -Session $s -ArgumentList (Split-Path $Installer -Leaf), $Seconds {
    param($setup, $secs)
    $ErrorActionPreference = 'Continue'
    $p = Start-Process "C:\ToneSphereTest\$setup" -ArgumentList '/S' -Wait -PassThru
    "installer exit code: $($p.ExitCode)"
    $tool = 'C:\Program Files\ToneSphere\resources\drivers\tonesphere-setup.exe'
    "before: $(& $tool status)"
    "virtual audio driver in the public build: $(Test-Path 'C:\Program Files\ToneSphere\resources\drivers\vad')"

    # Start ToneSphere in the signed-in user's session, as a user would.
    Set-Content C:\ToneSphereTest\launch.cmd '@start "" "C:\Program Files\ToneSphere\ToneSphere.exe"'
    schtasks /create /f /tn ToneSphereApp /sc once /st 00:00 /ru tonesphere /it /rl highest /tr C:\ToneSphereTest\launch.cmd 2>$null | Out-Null
    schtasks /run /tn ToneSphereApp 2>$null | Out-Null
    $disc = 'C:\Users\tonesphere\AppData\Local\ToneSphere\engine.json'
    $deadline = (Get-Date).AddSeconds(120)
    while (-not (Test-Path $disc) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 2 }
    "engine started by the app: $(Test-Path $disc)"

    # What the app does before its UAC prompt: the official package from vb-audio.com, unpacked.
    $dl = 'C:\ToneSphereTest\vbcable'
    New-Item -ItemType Directory -Force "$dl\pack" | Out-Null
    Invoke-WebRequest 'https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip' -OutFile "$dl\pack.zip" -UseBasicParsing
    & "$env:SystemRoot\System32\tar.exe" -xf "$dl\pack.zip" -C "$dl\pack"
    "downloaded: $((Get-Item "$dl\pack.zip").Length) bytes, setup present: $(Test-Path "$dl\pack\VBCABLE_Setup_x64.exe")"

    # The elevated step, in the desktop session (VB-Audio's installer has a window), with the
    # user's clicks played by vbcable-click.ps1.
    # schtasks /tr is limited to 261 characters: the command goes in a script.
    Set-Content C:\ToneSphereTest\install.cmd "@`"$tool`" install-all `"C:\Program Files\ToneSphere\resources\drivers`" --vbcable $dl\pack --result C:\ToneSphereTest\setup-result.jsonl > C:\ToneSphereTest\setup-out.txt 2>&1"
    schtasks /create /f /tn ToneSphereInstall /sc once /st 00:00 /ru tonesphere /it /rl highest /tr C:\ToneSphereTest\install.cmd 2>$null | Out-Null
    $click = 'powershell -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File C:\ToneSphereTest\vbcable-click.ps1'
    schtasks /create /f /tn ToneSphereClick /sc once /st 00:00 /ru tonesphere /it /rl highest /tr $click 2>$null | Out-Null
    schtasks /run /tn ToneSphereClick 2>$null | Out-Null
    schtasks /run /tn ToneSphereInstall 2>$null | Out-Null
    $deadline = (Get-Date).AddMinutes(6)
    do { Start-Sleep -Seconds 3 } while (-not (Test-Path C:\ToneSphereTest\setup-out.txt) -or ((Get-Content C:\ToneSphereTest\setup-out.txt -Raw) -notmatch 'VB-CABLE (installed|was not)') -and (Get-Date) -lt $deadline)
    Start-Sleep -Seconds 2
    '--- install-all results'; Get-Content C:\ToneSphereTest\setup-result.jsonl -ErrorAction SilentlyContinue
    '--- clicks'; Get-Content C:\ToneSphereTest\vbcable-click.log -ErrorAction SilentlyContinue
    "after: $(& $tool status)"

    # The running engine notices the new devices by itself (no restart, no rescan).
    $d = Get-Content $disc -Raw | ConvertFrom-Json
    $state = $null
    $deadline = (Get-Date).AddSeconds(60)
    do {
      Start-Sleep -Seconds 3
      $r = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$($d.port)/rpc" -Headers @{ Authorization = "Bearer $($d.token)" } -ContentType 'application/json' -Body '{"id":1,"method":"engine.getState"}'
      $state = $r.result
      $cable = @($state.nodes | Where-Object { $_.kind -eq 'compat' -and $_.state -ne 'unavailable' })
    } while ($cable.Count -lt 2 -and (Get-Date) -lt $deadline)
    "engine sees VB-CABLE by itself: $($cable.Count -ge 2) ($($cable.name -join ', ')); driverPack: $($state.status.driverPack | ConvertTo-Json -Compress)"

    # Audio both ways through the cable, measured (in the desktop session: audio needs it).
    $test = "cmd /c C:\ToneSphereTest\node.exe C:\ToneSphereTest\tests\integration\host-audio.mjs --soak $secs > C:\ToneSphereTest\audio.txt 2>&1"
    schtasks /create /f /tn ToneSphereAudio /sc once /st 00:00 /ru tonesphere /it /rl highest /tr $test 2>$null | Out-Null
    schtasks /run /tn ToneSphereAudio 2>$null | Out-Null
    $deadline = (Get-Date).AddMinutes(5)
    do { Start-Sleep -Seconds 3 } while (-not ((Get-Content C:\ToneSphereTest\audio.txt -Raw -ErrorAction SilentlyContinue) -match '\d+/\d+ passed') -and (Get-Date) -lt $deadline)
    '--- host-audio through VB-CABLE'; Get-Content C:\ToneSphereTest\audio.txt
  }
} finally {
  Remove-PSSession $s
}
