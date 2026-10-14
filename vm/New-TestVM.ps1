# Creates the ToneSphere driver-test VM fully unattended (no Windows Setup UI):
#  1. creates a GPT VHDX, applies install.wim from the Windows 11 evaluation ISO with DISM,
#  2. makes it bootable (bcdboot) and enables TEST SIGNING offline in the VM's BCD,
#  3. injects unattend.xml (local admin, skip OOBE, autologon),
#  4. creates a Gen2 Hyper-V VM with Secure Boot OFF, starts it, waits for PowerShell Direct,
#  5. takes a "clean" checkpoint to revert to after any driver crash.
# The host is never modified beyond creating the VM. Requires an elevated shell.
param(
  [string]$Iso = 'C:\ToneSphereVM\win11eval.iso',
  [string]$VmName = 'ToneSphere-DriverTest',
  [string]$Dir = 'C:\ToneSphereVM',
  [int]$DiskGB = 64,
  [long]$MemoryBytes = 6GB,
  [int]$Cpus = 4
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'vmcommon.ps1')

if (Get-VM -Name $VmName -ErrorAction SilentlyContinue) { Write-Host "VM $VmName already exists"; return }
if (-not (Test-Path $Iso)) {
  # Windows 11 Enterprise evaluation (90 days, en-US x64) from Microsoft's Evaluation Center.
  Write-Host '[0/5] Downloading the Windows 11 Enterprise evaluation ISO (about 7 GB)...'
  New-Item -ItemType Directory -Force (Split-Path $Iso) | Out-Null
  & curl.exe -L --fail --retry 5 -C - -o "$Iso.part" 'https://go.microsoft.com/fwlink/?LinkId=2382600&clcid=0x409&culture=en-us&country=us'
  if ($LASTEXITCODE) { throw 'ISO download failed' }
  Move-Item "$Iso.part" $Iso
}
$vhd = Join-Path $Dir "$VmName.vhdx"
if (Test-Path $vhd) { Remove-Item $vhd -Force }

Write-Host '[1/5] Creating VHDX and applying Windows image (takes several minutes)...'
New-VHD -Path $vhd -SizeBytes ($DiskGB * 1GB) -Dynamic | Out-Null
$disk = Mount-VHD -Path $vhd -Passthru | Get-Disk
try {
  Initialize-Disk -Number $disk.Number -PartitionStyle GPT
  $efi = New-Partition -DiskNumber $disk.Number -Size 260MB -GptType '{c12a7328-f81f-11d2-ba4b-00a0c93ec93b}'
  Format-Volume -Partition $efi -FileSystem FAT32 -NewFileSystemLabel 'System' -Confirm:$false | Out-Null
  New-Partition -DiskNumber $disk.Number -Size 16MB -GptType '{e3c9e316-0b5c-4db8-817d-f92df00215ae}' | Out-Null
  $win = New-Partition -DiskNumber $disk.Number -UseMaximumSize -GptType '{ebd0a0a2-b9e5-4433-87c0-68b6b72699c7}'
  Format-Volume -Partition $win -FileSystem NTFS -NewFileSystemLabel 'Windows' -Confirm:$false | Out-Null
  $efi | Add-PartitionAccessPath -AssignDriveLetter
  $win | Add-PartitionAccessPath -AssignDriveLetter
  $efiL = (Get-Partition -DiskNumber $disk.Number -PartitionNumber $efi.PartitionNumber).DriveLetter
  $winL = (Get-Partition -DiskNumber $disk.Number -PartitionNumber $win.PartitionNumber).DriveLetter

  # Note: PowerShell variables are case-insensitive, so don't reuse the name $iso here.
  $isoImg = Mount-DiskImage -ImagePath $Iso -PassThru
  $isoL = (Get-Volume -DiskImage $isoImg).DriveLetter
  try {
    $wim = "${isoL}:\sources\install.wim"
    if (-not (Test-Path $wim)) { $wim = "${isoL}:\sources\install.esd" }
    dism /Apply-Image /ImageFile:$wim /Index:1 /ApplyDir:"${winL}:\" | Out-Host
    if ($LASTEXITCODE) { throw 'DISM apply failed' }
  } finally { Dismount-DiskImage -ImagePath $Iso | Out-Null }

  Write-Host '[2/5] Making the disk bootable and enabling test signing in the VM BCD...'
  & "$env:SystemRoot\System32\bcdboot.exe" "${winL}:\Windows" /s "${efiL}:" /f UEFI | Out-Host
  $bcd = "${efiL}:\EFI\Microsoft\Boot\BCD"
  & bcdedit.exe /store $bcd /set '{default}' testsigning on | Out-Host
  & bcdedit.exe /store $bcd /set '{default}' nointegritychecks on | Out-Host

  Write-Host '[3/5] Injecting unattend.xml and first-boot script...'
  New-Item -ItemType Directory -Force "${winL}:\Windows\Panther" | Out-Null
  # git holds only a placeholder; the password comes from the machine-local secret (vmcommon.ps1).
  $unattend = (Get-Content (Join-Path $PSScriptRoot 'unattend.xml') -Raw).Replace('@PASSWORD@', [Security.SecurityElement]::Escape((Get-TestVMPassword)))
  Set-Content -Path "${winL}:\Windows\Panther\unattend.xml" -Value $unattend -Encoding utf8 -NoNewline
  New-Item -ItemType Directory -Force "${winL}:\Windows\Setup\Scripts" | Out-Null
  Copy-Item (Join-Path $PSScriptRoot 'SetupComplete.cmd') "${winL}:\Windows\Setup\Scripts\SetupComplete.cmd" -Force
} finally {
  Dismount-VHD -Path $vhd
}

Write-Host '[4/5] Creating the Hyper-V VM (Gen2, Secure Boot off)...'
$sw = (Get-VMSwitch -Name 'Default Switch' -ErrorAction SilentlyContinue)
$vm = New-VM -Name $VmName -Generation 2 -MemoryStartupBytes $MemoryBytes -VHDPath $vhd -Path $Dir
if ($sw) { Connect-VMNetworkAdapter -VMName $VmName -SwitchName $sw.Name }
Set-VMFirmware -VMName $VmName -EnableSecureBoot Off
Set-VMProcessor -VMName $VmName -Count $Cpus
Set-VMMemory -VMName $VmName -DynamicMemoryEnabled $false
Set-VM -Name $VmName -CheckpointType Standard -AutomaticCheckpointsEnabled $false
Enable-VMIntegrationService -VMName $VmName -Name 'Guest Service Interface'
Start-VM -Name $VmName

Write-Host '[5/5] Waiting for Windows to finish first boot (PowerShell Direct)...'
Wait-TestVM -VmName $VmName -TimeoutMinutes 45
Invoke-Command -VMName $VmName -Credential (Get-TestVMCredential) -ScriptBlock {
  bcdedit | Select-String testsigning
  Get-Service Audiosrv, AudioEndpointBuilder | Select-Object Name, Status
}
# First sign-in still has a restart pending (CloudExperienceHost "reconfiguration"). Let it happen
# before the checkpoint, or every Restore-TestVM reboots the VM a minute later, mid-test.
Wait-TestVMSettled -VmName $VmName
Stop-VM -Name $VmName -Force
Checkpoint-VM -Name $VmName -SnapshotName 'clean'
Start-VM -Name $VmName
Wait-TestVM -VmName $VmName -TimeoutMinutes 15
Write-Host "VM '$VmName' is ready (checkpoint 'clean' created)."
