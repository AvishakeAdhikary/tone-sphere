# One full driver test pass inside the VM built by new_driver_vm.ps1. Elevated Windows
# PowerShell on the host, after `uv run python scripts/build_native.py` and
# `uv run python scripts/build_driver.py`:
#
#   powershell -ExecutionPolicy Bypass -File scripts\vm\run_driver_tests.ps1
#
# Each pass starts from the 'clean' checkpoint, so it tests exactly this build of the
# driver on a guest that has never seen another one. In the guest it installs the driver,
# runs tests/hardware/test_virtual_driver.py, uninstalls, and checks nothing is left. Every
# log comes back to -Out, including the guest's setupapi log and any crash dump, because a
# failed driver install or a bugcheck is only diagnosable from those.

param(
    [string]$Name = 'ToneSphereDriverVM',
    [string]$Directory = 'C:\ToneSphereVM',
    [string]$Out = (Join-Path 'C:\ToneSphereVM' ('run-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))),
    [string]$PytestArgs = '-m hardware -s -v -rA'
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'vm_common.ps1')
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$credential = Get-GuestCredential -Directory $Directory
New-Item -ItemType Directory -Path $Out -Force | Out-Null

# The tree the guest runs: what git tracks, plus the built binaries git ignores.
$stage = Join-Path $Directory 'stage'
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage | Out-Null
Push-Location $repo
try {
    $files = @(git ls-files) + @(git ls-files --others --exclude-standard)
    $files += (Get-ChildItem tonesphere\native\_bin -File).FullName | ForEach-Object { Resolve-Path -Relative $_ }
    $files += Get-ChildItem native\build -Recurse -File |
        Where-Object { $_.FullName -match '\\VST3\\Release\\tonesphere_test_gain\.vst3\\' } |
        ForEach-Object { Resolve-Path -Relative $_.FullName }
    $files += Get-ChildItem driver\windows_virtual_audio\x64\Release\vm_kit -File |
        ForEach-Object { Resolve-Path -Relative $_.FullName }
    foreach ($f in $files) {
        $relative = $f -replace '^\.\\', '' -replace '/', '\'
        if (-not (Test-Path -LiteralPath $relative -PathType Leaf)) { continue }
        $target = Join-Path $stage $relative
        New-Item -ItemType Directory -Path (Split-Path $target) -Force | Out-Null
        Copy-Item -LiteralPath $relative -Destination $target
    }
} finally {
    Pop-Location
}
Copy-Item (Get-Command uv).Source (Join-Path $stage 'uv.exe')
$zip = Join-Path $Directory 'stage.zip'
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip
Write-Host "Staged $($files.Count) files -> $zip"

# 'deps' is 'clean' plus Python and the locked dependencies, taken by the first pass that
# installs them: the driver has still never been near it, and later passes skip a download
# of several hundred megabytes. The environment lives outside C:\ts, which each pass replaces.
$start = if (Get-VMSnapshot -VMName $Name -Name 'deps' -ErrorAction SilentlyContinue) { 'deps' } else { 'clean' }
Write-Host "Restoring checkpoint '$start'..."
Restore-VMSnapshot -VMName $Name -Name $start -Confirm:$false
if ((Get-VM -Name $Name).State -ne 'Running') { Start-VM -Name $Name }
Wait-Guest -Name $Name -Credential $credential -TimeoutMinutes 20

$session = New-PSSession -VMName $Name -Credential $credential
try {
    Invoke-Command -Session $session -ScriptBlock {
        Remove-Item C:\ts, C:\ts-out -Recurse -Force -ErrorAction SilentlyContinue
        New-Item -ItemType Directory C:\ts, C:\ts-out -Force | Out-Null
    }
    Copy-Item -Path $zip -Destination 'C:\ts-stage.zip' -ToSession $session
    $synced = [int](Invoke-Command -Session $session -ScriptBlock {
        Expand-Archive C:\ts-stage.zip -DestinationPath C:\ts -Force
        Set-Location C:\ts
        [Environment]::SetEnvironmentVariable('UV_PROJECT_ENVIRONMENT', 'C:\ts-venv', 'Machine')
        [Environment]::SetEnvironmentVariable('UV_LINK_MODE', 'copy', 'Machine')
        $env:UV_PROJECT_ENVIRONMENT = 'C:\ts-venv'
        $env:UV_LINK_MODE = 'copy'
        & .\uv.exe sync *>&1 | Out-File -Encoding utf8 'C:\ts-out\uv_sync.log'
        $LASTEXITCODE
    })
    if ($synced -eq 0 -and $start -eq 'clean') {
        Write-Host "Dependencies installed; taking checkpoint 'deps'."
        Checkpoint-VM -Name $Name -SnapshotName 'deps'
        # A checkpoint of a running VM breaks the PowerShell Direct session.
        Remove-PSSession $session
        Wait-Guest -Name $Name -Credential $credential -TimeoutMinutes 5
        $session = New-PSSession -VMName $Name -Credential $credential
    }
    $result = Invoke-Command -Session $session -ArgumentList $PytestArgs, $synced -ScriptBlock {
        param($PytestArgs, $synced)
        $ErrorActionPreference = 'Continue'
        $log = 'C:\ts-out'
        Set-Location C:\ts
        $env:UV_PROJECT_ENVIRONMENT = 'C:\ts-venv'
        $env:UV_LINK_MODE = 'copy'
        $steps = [ordered]@{ uv_sync = $synced }

        $kit = 'C:\ts\driver\windows_virtual_audio\x64\Release\vm_kit'
        & powershell -NoProfile -ExecutionPolicy Bypass -File "$kit\driver_install.ps1" *>&1 | Out-File -Encoding utf8 "$log\install.log"
        $steps.install = $LASTEXITCODE
        Get-PnpDevice -PresentOnly | Where-Object { $_.FriendlyName -like '*ToneSphere*' } |
            Format-Table -AutoSize Status, Class, FriendlyName, InstanceId | Out-String -Width 250 |
            Set-Content "$log\devices_after_install.txt"

        $pytest = "C:\ts\uv.exe run pytest tests/hardware/test_virtual_driver.py $PytestArgs"
        cmd /c "$pytest > $log\pytest.log 2>&1"
        $steps.pytest = $LASTEXITCODE

        & powershell -NoProfile -ExecutionPolicy Bypass -File "$kit\driver_uninstall.ps1" *>&1 | Out-File -Encoding utf8 "$log\uninstall.log"
        $steps.uninstall = $LASTEXITCODE
        Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue | Where-Object { $_.FriendlyName -like '*ToneSphere*' } |
            Format-Table -AutoSize Status, Class, FriendlyName, InstanceId | Out-String -Width 250 |
            Set-Content "$log\devices_after_uninstall.txt"

        Copy-Item C:\Windows\INF\setupapi.dev.log $log -ErrorAction SilentlyContinue
        if (Test-Path C:\Windows\Minidump) { Copy-Item C:\Windows\Minidump\* $log -ErrorAction SilentlyContinue }
        [pscustomobject]@{
            Steps   = $steps
            Os      = (Get-CimInstance Win32_OperatingSystem).Caption + ' ' + [Environment]::OSVersion.Version
            Session = [Diagnostics.Process]::GetCurrentProcess().SessionId
        }
    }
    Copy-Item -Path 'C:\ts-out\*' -Destination $Out -FromSession $session -Recurse
} finally {
    Remove-PSSession $session
}

$result | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $Out 'summary.json')
$result | ConvertTo-Json -Depth 4 | Out-Host
Write-Host "Logs: $Out"
