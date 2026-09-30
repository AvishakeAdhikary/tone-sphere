# Build the Hyper-V VM the virtual audio driver is tested in. Elevated Windows PowerShell,
# on a host with Hyper-V enabled:
#
#   powershell -ExecutionPolicy Bypass -File scripts\vm\new_driver_vm.ps1 -Iso <Windows 11 ISO>
#
# The image is applied straight to a new VHDX instead of booting Windows Setup: Setup on a
# Generation 2 VM stops at "Press any key to boot from CD or DVD" and at Windows 11's TPM
# check, and neither can be answered unattended. Test-signing is switched on in the VHDX's
# own boot store, offline; the host's boot configuration is never touched (AGENTS.md: the
# development host is never put into test-signing mode). Secure Boot is off in the VM's
# firmware, because Windows ignores test-signing while it is on.
#
# The result is a VM with a local administrator the host reaches over PowerShell Direct
# (no network needed for control), and a checkpoint named 'clean' that every test run
# starts from, so no run inherits what a previous driver build left behind.

param(
    [string]$Iso,
    [string]$Name = 'ToneSphereDriverVM',
    [string]$Directory = 'C:\ToneSphereVM',
    [string]$Edition = 'LTSC',
    [long]$DiskBytes = 64GB,
    [long]$MemoryBytes = 4GB,
    [int]$Processors = 4,
    # Reuse a VHDX this script already prepared (image applied, boot files, test-signing,
    # answer file) and only create and boot the VM.
    [switch]$ExistingDisk
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'vm_common.ps1')

$principal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this from an elevated PowerShell.'
}
if (Get-VM -Name $Name -ErrorAction SilentlyContinue) { throw "A VM named $Name already exists." }

New-Item -ItemType Directory -Path $Directory -Force | Out-Null
$vhd = Join-Path $Directory "$Name.vhdx"
if ((Test-Path $vhd) -and -not $ExistingDisk) { throw "$vhd already exists (use -ExistingDisk to reuse it)." }
$credential = Get-GuestCredential -Directory $Directory -Create

if (-not $ExistingDisk) {
    if (-not $Iso) { throw 'Pass -Iso <Windows 11 ISO>.' }
    $isoImage = Mount-DiskImage -ImagePath (Resolve-Path $Iso).Path -PassThru
    try {
        $isoLetter = ($isoImage | Get-Volume).DriveLetter
        $wim = "${isoLetter}:\sources\install.wim"
        if (-not (Test-Path $wim)) { $wim = "${isoLetter}:\sources\install.esd" }
        $image = Get-WindowsImage -ImagePath $wim | Where-Object { $_.ImageName -like "*$Edition*" } | Select-Object -First 1
        if (-not $image) { throw "No image matching '$Edition' in $wim" }
        Write-Host "Applying '$($image.ImageName)' (index $($image.ImageIndex)) from $wim"

        New-VHD -Path $vhd -SizeBytes $DiskBytes -Dynamic | Out-Null
        $disk = Mount-VHD -Path $vhd -Passthru | Get-Disk
        try {
            # From Z down: a letter can be held by something that is not a volume (a card
            # reader, a mapped share), so every source of drive letters is consulted.
            $used = @((Get-Volume).DriveLetter) + @((Get-PSDrive -PSProvider FileSystem).Name) +
                    @([IO.DriveInfo]::GetDrives() | ForEach-Object { $_.Name[0] }) |
                    Where-Object { $_ } | ForEach-Object { [string]$_ }
            $free = [char[]](90..70) | Where-Object { $used -notcontains [string]$_ }
            $system, $windows = $free[0], $free[1]
            $diskpart = @"
select disk $($disk.Number)
convert gpt noerr
create partition efi size=260
format quick fs=fat32 label="System"
assign letter=$system
create partition primary
format quick fs=ntfs label="Windows"
assign letter=$windows
"@
            $diskpart | diskpart | Out-Host
            if (-not (Test-Path "${windows}:\")) { throw "diskpart did not produce the Windows volume" }

            Expand-WindowsImage -ImagePath $wim -Index $image.ImageIndex -ApplyPath "${windows}:\" | Out-Null
            # The host's bcdboot, not the image's: the 26100 image's own copy fails on a 26200
            # host (0xC0EA0002). With /s it writes only the named system partition and leaves the
            # host's firmware boot entries alone (checked with `bcdedit /enum firmware`).
            & "$env:SystemRoot\System32\bcdboot.exe" "${windows}:\Windows" /s "${system}:" /f UEFI | Out-Host
            if ($LASTEXITCODE) { throw "bcdboot failed ($LASTEXITCODE)" }
            # /store: this edits the VM disk's boot store, never the host's.
            $store = "${system}:\EFI\Microsoft\Boot\BCD"
            bcdedit /store $store /set '{default}' testsigning on | Out-Host
            if ($LASTEXITCODE) { throw "could not enable test-signing in $store" }

            $panther = "${windows}:\Windows\Panther"
            New-Item -ItemType Directory -Path $panther -Force | Out-Null
            $password = $credential.GetNetworkCredential().Password
            (Get-Content (Join-Path $PSScriptRoot 'unattend.xml') -Raw).Replace('@USER@', $credential.UserName).Replace('@PASSWORD@', $password) |
                Set-Content -Path (Join-Path $panther 'unattend.xml') -Encoding UTF8
        } finally {
            Dismount-VHD -Path $vhd
        }
    } finally {
        Dismount-DiskImage -ImagePath $isoImage.ImagePath | Out-Null
    }
}

New-VM -Name $Name -Generation 2 -MemoryStartupBytes $MemoryBytes -VHDPath $vhd -SwitchName 'Default Switch' -Path $Directory | Out-Null
Set-VM -Name $Name -ProcessorCount $Processors -StaticMemory -AutomaticCheckpointsEnabled $false -CheckpointType Standard
Set-VMFirmware -VMName $Name -EnableSecureBoot Off
Enable-VMIntegrationService -VMName $Name -Name 'Guest Service Interface'
Start-VM -Name $Name

Write-Host 'Waiting for the guest to finish its first boot and answer PowerShell Direct...'
Wait-Guest -Name $Name -Credential $credential -TimeoutMinutes 45

# A test run must not be interrupted by the guest restarting itself. Windows Update would,
# and so does OOBE: its CloudExperienceHost restarts the guest once, about a quarter of an
# hour after the first logon. A checkpoint taken before that restart replays it on every
# restore (seen here: a test pass cut off by "a system shutdown is in progress"). So updates
# are switched off, and the checkpoint waits until that restart has happened, or until the
# guest has run long enough that it is not coming.
Invoke-Command -VMName $Name -Credential $credential -ScriptBlock {
    New-Item 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU' -Force | Out-Null
    Set-ItemProperty 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU' NoAutoUpdate 1 -Type DWord
    Stop-Service wuauserv -Force -ErrorAction SilentlyContinue
    Set-Service wuauserv -StartupType Disabled
}
Write-Host 'Waiting for the guest to settle (OOBE restart, or 25 minutes without one)...'
$started = Get-Date
$settled = $false
while (-not $settled -and (Get-Date) -lt $started.AddMinutes(45)) {
    Start-Sleep -Seconds 30
    try {
        $state = Invoke-Command -VMName $Name -Credential $credential -ErrorAction Stop -ScriptBlock {
            $boot = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime
            [pscustomobject]@{
                UpMinutes = ((Get-Date) - $boot).TotalMinutes
                Restarted = [bool](Get-WinEvent -FilterHashtable @{ LogName = 'System'; Id = 1074 } -ErrorAction SilentlyContinue |
                                   Where-Object { $_.Message -like '*CloudExperienceHostBroker*' })
            }
        }
    } catch {
        continue
    }
    $settled = ($state.Restarted -and $state.UpMinutes -ge 3) -or $state.UpMinutes -ge 25
}
if (-not $settled) { throw 'The guest did not settle within 45 minutes.' }
$guest = Invoke-Command -VMName $Name -Credential $credential -ScriptBlock {
    $secureBoot = $false
    try { $secureBoot = Confirm-SecureBootUEFI } catch {}
    [pscustomobject]@{
        Os          = (Get-CimInstance Win32_OperatingSystem).Caption
        Build       = [Environment]::OSVersion.Version.ToString()
        TestSigning = ((bcdedit /enum '{current}') -join "`n") -match 'testsigning\s+Yes'
        SecureBoot  = $secureBoot
        Audio       = (Get-Service Audiosrv).Status.ToString()
        Elevated    = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
                          [Security.Principal.WindowsBuiltInRole]::Administrator)
    }
}
$guest | Format-List | Out-Host
if (-not $guest.TestSigning) { throw 'The guest booted without test-signing.' }
if (-not $guest.Elevated) { throw 'PowerShell Direct reaches the guest without administrator rights.' }

Checkpoint-VM -Name $Name -SnapshotName 'clean'
Write-Host "VM $Name is ready; checkpoint 'clean' taken."
