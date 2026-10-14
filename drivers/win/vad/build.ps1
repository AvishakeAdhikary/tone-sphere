# Builds, catalogs and test-signs the NexTone virtual audio driver (x64).
#   .\drivers\win\vad\build.ps1 [-Config Release|Debug]
# Output: drivers\win\vad\build\package\{nextonevad.sys, nextonevad.inf, nextonevad.cat, NexToneTest.cer}
# The driver is test-signed with a self-signed certificate: it loads only on machines in
# test-signing mode (used here exclusively inside the Hyper-V test VM).
param([ValidateSet('Release', 'Debug')] [string]$Config = 'Release')
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$root = Resolve-Path (Join-Path $here '..\..\..')
. (Join-Path $root 'scripts\vsenv.ps1') -Arch x64

$kits = "${env:ProgramFiles(x86)}\Windows Kits\10"
$ver = '10.0.26100.0'
$inc = @("$kits\Include\$ver\km", "$kits\Include\$ver\km\crt", "$kits\Include\$ver\shared")
$lib = "$kits\Lib\$ver\km\x64"
$out = Join-Path $here 'build'
$obj = Join-Path $out "obj-$Config"
$pkg = Join-Path $out 'package'
New-Item -ItemType Directory -Force $obj, $pkg | Out-Null

$sources = @(
  'adapter.cpp', 'basetopo.cpp', 'common.cpp', 'hw.cpp', 'kshelper.cpp', 'micarraytopo.cpp', 'mintopo.cpp',
  'minwavert.cpp', 'minwavertstream.cpp', 'NewDelete.cpp', 'speakertopo.cpp', 'nextone_ring.cpp'
) | ForEach-Object { Join-Path $here "src\$_" }

$defines = @('_WIN64', '_AMD64_', 'AMD64', 'DEPRECATE_DDK_FUNCTIONS=1', 'MSC_NOOPT', '_WIN32_WINNT=0x0A00',
  'WINVER=0x0A00', 'WINNT=1', 'NTDDI_VERSION=0x0A000008', 'POOL_NX_OPTIN=1', 'POOL_ZERO_DOWN_LEVEL_SUPPORT=1',
  '_NT_TARGET_VERSION=0xA000008', 'NTSTRSAFE_LIB', '_NEW_DELETE_OPERATORS_')
if ($Config -eq 'Debug') { $defines += @('DBG=1') }
$opt = if ($Config -eq 'Debug') { @('/Od') } else { @('/O2', '/Oi') }

$cflags = @('/nologo', '/c', '/Zi', '/W4', '/WX-', '/GF', '/Gm-', '/Zp8', '/GS', '/Gy', '/fp:precise', '/Zc:wchar_t-',
  '/Zc:forScope', '/Zc:inline', '/GR-', '/kernel', '/wd4748', '/wd4603', '/wd4627', '/wd4986', '/wd4987', '/wd4996',
  '/wd4100', '/wd4127', "/Fd$obj\nextonevad.pdb") + $opt
$cflags += $defines | ForEach-Object { "/D$_" }
$cflags += $inc | ForEach-Object { "/I$_" }
$cflags += @("/I$here\src", "/I$here\include")

Write-Host '== Compiling'
$objs = @()
foreach ($s in $sources) {
  $o = Join-Path $obj ([IO.Path]::GetFileNameWithoutExtension($s) + '.obj')
  & cl.exe @cflags "/Fo$o" $s
  if ($LASTEXITCODE) { throw "Compile failed: $s" }
  $objs += $o
}

Write-Host '== Linking'
$sys = Join-Path $pkg 'nextonevad.sys'
$lflags = @('/nologo', '/DRIVER', '/SUBSYSTEM:NATIVE,10.00', '/ENTRY:GsDriverEntry', '/NODEFAULTLIB', '/MANIFEST:NO',
  '/RELEASE', '/OPT:REF', '/OPT:ICF', '/INCREMENTAL:NO', '/DEBUG', '/MERGE:_TEXT=.text;_PAGE=PAGE', '/SECTION:INIT,d',
  '/IGNORE:4198,4010,4037,4039,4065,4070,4078,4087,4089,4221', '/MACHINE:X64', "/LIBPATH:$lib", "/OUT:$sys",
  "/PDB:$out\nextonevad.pdb")
$libs = @('BufferOverflowFastFailK.lib', 'ntoskrnl.lib', 'hal.lib', 'wmilib.lib', 'portcls.lib', 'stdunk.lib', 'ks.lib',
  'ksguid.lib', 'libcntpr.lib', 'ntstrsafe.lib')
& link.exe @lflags @objs @libs
if ($LASTEXITCODE) { throw 'Link failed' }

Copy-Item (Join-Path $here 'nextonevad.inf') $pkg -Force

Write-Host '== Test certificate'
$certName = 'NexTone Test Driver Signing'
$cert = Get-ChildItem Cert:\CurrentUser\My | Where-Object { $_.Subject -eq "CN=$certName" } | Select-Object -First 1
if (-not $cert) {
  $cert = New-SelfSignedCertificate -Type CodeSigningCert -Subject "CN=$certName" -CertStoreLocation Cert:\CurrentUser\My `
    -KeyUsage DigitalSignature -KeyAlgorithm RSA -KeyLength 3072 -HashAlgorithm SHA256 -NotAfter (Get-Date).AddYears(5)
}
Export-Certificate -Cert $cert -FilePath (Join-Path $pkg 'NexToneTest.cer') | Out-Null

Write-Host '== Catalog + signing'
$signtool = "$kits\bin\$ver\x64\signtool.exe"
& $signtool sign /q /fd sha256 /sha1 $cert.Thumbprint /s My $sys
if ($LASTEXITCODE) { throw 'Signing the driver failed' }
Remove-Item (Join-Path $pkg 'nextonevad.cat') -ErrorAction SilentlyContinue
& "$kits\bin\$ver\x86\Inf2Cat.exe" /driver:$pkg /os:10_VB_X64,10_CO_X64 /uselocaltime | Out-Host
if ($LASTEXITCODE) { throw 'Inf2Cat failed' }
& $signtool sign /q /fd sha256 /sha1 $cert.Thumbprint /s My (Join-Path $pkg 'nextonevad.cat')
if ($LASTEXITCODE) { throw 'Signing the catalog failed' }
Write-Host "Driver package: $pkg"
