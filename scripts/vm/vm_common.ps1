# Shared by new_driver_vm.ps1 and run_driver_tests.ps1.

function Get-GuestCredential([string]$Directory, [switch]$Create) {
    # The guest's local administrator. Its password is generated per VM and kept beside the
    # VM's disk, not in the repository: it guards nothing but a disposable test VM, and the
    # host needs it to reach the guest over PowerShell Direct.
    $file = Join-Path $Directory 'guest-credential.txt'
    if (-not (Test-Path $file)) {
        if (-not $Create) { throw "No guest credential at $file; was the VM built by new_driver_vm.ps1?" }
        $bytes = New-Object byte[] 18
        [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
        $generated = 'Ts1!' + [Convert]::ToBase64String($bytes).Replace('/', 'x').Replace('+', 'y')
        Set-Content -Path $file -Value "tsadmin`n$generated"
    }
    $user, $password = Get-Content $file
    New-Object System.Management.Automation.PSCredential ($user, (ConvertTo-SecureString $password -AsPlainText -Force))
}

function Wait-Guest([string]$Name, [pscredential]$Credential, [int]$TimeoutMinutes = 20) {
    $deadline = (Get-Date).AddMinutes($TimeoutMinutes)
    while ((Get-Date) -lt $deadline) {
        try {
            $null = Invoke-Command -VMName $Name -Credential $Credential -ScriptBlock { $env:COMPUTERNAME } -ErrorAction Stop
            return
        } catch {
            Start-Sleep -Seconds 15
        }
    }
    throw "The guest $Name did not answer PowerShell Direct within $TimeoutMinutes minutes."
}
