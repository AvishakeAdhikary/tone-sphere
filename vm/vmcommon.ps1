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

# Waits until Windows in the VM has been up for $Minutes without restarting, riding out a
# restart it still had pending (first sign-in reconfiguration, updates).
function Wait-TestVMSettled {
  param([string]$VmName = 'ToneSphere-DriverTest', [int]$Minutes = 4, [int]$TimeoutMinutes = 30)
  $cred = Get-TestVMCredential
  $deadline = (Get-Date).AddMinutes($TimeoutMinutes)
  while ((Get-Date) -lt $deadline) {
    Wait-TestVM -VmName $VmName
    try {
      $up = Invoke-Command -VMName $VmName -Credential $cred -ErrorAction Stop -ScriptBlock {
        ((Get-Date) - (Get-CimInstance Win32_OperatingSystem).LastBootUpTime).TotalMinutes
      }
      if ($up -ge $Minutes) { return }
    } catch {}
    Start-Sleep -Seconds 30
  }
  throw "VM $VmName kept restarting for $TimeoutMinutes minutes"
}

function Restore-TestVM {
  param([string]$VmName = 'ToneSphere-DriverTest', [string]$Checkpoint = 'clean')
  Stop-VM -Name $VmName -TurnOff -Force -ErrorAction SilentlyContinue
  Restore-VMSnapshot -VMName $VmName -Name $Checkpoint -Confirm:$false
  Start-VM -Name $VmName
  Wait-TestVM -VmName $VmName
}
