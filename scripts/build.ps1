# Configures and builds ToneSphere (engine, ASIO driver, tools) with MSVC + Ninja.
#   .\scripts\build.ps1                 # Release x64
#   .\scripts\build.ps1 -Config Debug
#   .\scripts\build.ps1 -Arch x86       # 32-bit (ASIO driver for 32-bit DAWs)
#   .\scripts\build.ps1 -Test           # also runs unit tests
param(
  [ValidateSet('Release', 'Debug', 'RelWithDebInfo')] [string]$Config = 'Release',
  [ValidateSet('x64', 'x86', 'arm64')] [string]$Arch = 'x64',
  [switch]$Test,
  [switch]$Clean
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -products * -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'Visual Studio C++ build tools not found' }
$hostArch = 'x64'
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvarsall.bat'
$target = if ($Arch -eq 'x64') { 'x64' } elseif ($Arch -eq 'x86') { 'x64_x86' } else { 'x64_arm64' }
$build = Join-Path $root "build\$Arch-$($Config.ToLower())"
if ($Clean -and (Test-Path $build)) { Remove-Item -Recurse -Force $build }

# Import the MSVC environment into this PowerShell session.
cmd /c "`"$vcvars`" $target >nul && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] }
}
Remove-Item env:ELECTRON_RUN_AS_NODE -ErrorAction SilentlyContinue

$tests = if ($Arch -eq 'x64') { 'ON' } else { 'OFF' }
$cmakeArgs = @("-DCMAKE_BUILD_TYPE=$Config", "-DTONESPHERE_BUILD_TESTS=$tests")
# CI keeps the downloaded SDKs (VST3, Catch2, RNNoise, json) in one cached folder.
if ($env:TS_DEPS_DIR) { $cmakeArgs += "-DFETCHCONTENT_BASE_DIR=$env:TS_DEPS_DIR\$Arch" }
cmake -S $root -B $build -G Ninja @cmakeArgs
if ($LASTEXITCODE) { throw 'CMake configure failed' }
cmake --build $build --parallel
if ($LASTEXITCODE) { throw 'Build failed' }
if ($Test) {
  ctest --test-dir $build --output-on-failure
  if ($LASTEXITCODE) { throw 'Tests failed' }
}
Write-Host "Build output: $build\bin"
