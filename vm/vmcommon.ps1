# Shared helpers for the ToneSphere test VM.
$script:TestVmUser = 'tonesphere'
# The VM's local admin password is generated on this machine and kept outside the repository.
$script:TestVmSecretDir = Join-Path $env:LOCALAPPDATA 'ToneSphereVM'

function Get-TestVMPassword {
  $file = Join-Path $script:TestVmSecretDir 'vm-password.txt'
  if (-not (Test-Path $file)) {
    New-Item -ItemType Directory -Force $script:TestVmSecretDir | Out-Null
    $chars = [char[]]'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789'
    $bytes = [byte[]]::new(20)
    [Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
    # A fixed upper/lower/digit/symbol prefix keeps Windows' complexity rules satisfied.
    $pw = 'Ts-9a' + (-join ($bytes | ForEach-Object { $chars[$_ % $chars.Length] }))
    Set-Content -Path $file -Value $pw -NoNewline -Encoding ascii
  }
  (Get-Content $file -Raw).Trim()
}

function Get-TestVMCredential {
  $sec = ConvertTo-SecureString (Get-TestVMPassword) -AsPlainText -Force
  New-Object System.Management.Automation.PSCredential(".\$script:TestVmUser", $sec)
}

function Wait-TestVM {
  param([string]$VmName = 'ToneSphere-DriverTest', [int]$TimeoutMinutes = 20)
  $cred = Get-TestVMCredential
  $deadline = (Get-Date).AddMinutes($TimeoutMinutes)
  while ((Get-Date) -lt $deadline) {
    try {
      $ok = Invoke-Command -VMName $VmName -Credential $cred -ScriptBlock { 'ready' } -ErrorAction Stop
      if ($ok -eq 'ready') { return }
    } catch { Start-Sleep -Seconds 15 }
  }
  throw "VM $VmName did not become reachable within $TimeoutMinutes minutes"
}

function Restore-TestVM {
  param([string]$VmName = 'ToneSphere-DriverTest', [string]$Checkpoint = 'clean')
  Stop-VM -Name $VmName -TurnOff -Force -ErrorAction SilentlyContinue
  Restore-VMSnapshot -VMName $VmName -Name $Checkpoint -Confirm:$false
  Start-VM -Name $VmName
  Wait-TestVM -VmName $VmName
}
