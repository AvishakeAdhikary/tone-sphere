# Shared helpers for the ToneSphere test VM.
$script:TestVmUser = 'tonesphere'
$script:TestVmPassword = 'ToneSphere-Test-2026!'  # local test VM only, never used elsewhere

function Get-TestVMCredential {
  $sec = ConvertTo-SecureString $script:TestVmPassword -AsPlainText -Force
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
