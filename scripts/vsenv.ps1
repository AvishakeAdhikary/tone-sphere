# Imports the MSVC developer environment into the current PowerShell session.
#   . .\scripts\vsenv.ps1 [-Arch x64|x86|arm64]
param([ValidateSet('x64', 'x86', 'arm64')] [string]$Arch = 'x64')
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -products * -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'Visual Studio C++ build tools not found' }
$target = if ($Arch -eq 'x64') { 'x64' } elseif ($Arch -eq 'x86') { 'x64_x86' } else { 'x64_arm64' }
cmd /c "`"$(Join-Path $vs 'VC\Auxiliary\Build\vcvarsall.bat')`" $target >nul && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] }
}
Remove-Item env:ELECTRON_RUN_AS_NODE -ErrorAction SilentlyContinue
