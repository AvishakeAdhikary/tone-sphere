; The Windows installer: what a user downloads and double-clicks.
;
; Built in CI from PyInstaller's one-folder output (dist\ToneSphere) with Inno Setup:
;   iscc /DAppVersion=0.2.1 /DSourceDir=dist\ToneSphere /DOutputDir=release packaging\windows\ToneSphere.iss
;
; Installed for the current user only, under %LOCALAPPDATA%\Programs: no administrator rights,
; so no UAC prompt, and nothing written outside the user's own profile. Uninstalling leaves the
; user's settings, presets and logs (%LOCALAPPDATA%\Neural Nexus Studios\ToneSphere) in place.

#ifndef AppVersion
  #error AppVersion must be defined (/DAppVersion=x.y.z)
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\ToneSphere"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\release"
#endif

[Setup]
AppId={{6F7B0C2E-6A3D-4C4B-9E55-2D7B1A0E9C31}
AppName=ToneSphere
AppVersion={#AppVersion}
AppVerName=ToneSphere {#AppVersion}
AppPublisher=Neural Nexus Studios
AppPublisherURL=https://github.com/AvishakeAdhikary/tone-sphere
AppSupportURL=https://github.com/AvishakeAdhikary/tone-sphere/issues
VersionInfoVersion={#AppVersion}
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\ToneSphere
DisableProgramGroupPage=yes
DisableDirPage=auto
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
OutputDir={#OutputDir}
OutputBaseFilename=ToneSphere-{#AppVersion}-Setup
SetupIconFile=..\..\assets\images\ToneSphere.ico
UninstallDisplayIcon={app}\ToneSphere.exe
UninstallDisplayName=ToneSphere
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; A new version's files replace the old ones wholesale: a stale DLL left from an older build
; would be loaded beside the new ones.
Type: filesandordirs; Name: "{app}\_internal"

[Icons]
Name: "{userprograms}\ToneSphere"; Filename: "{app}\ToneSphere.exe"
Name: "{userdesktop}\ToneSphere"; Filename: "{app}\ToneSphere.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\ToneSphere.exe"; Description: "{cm:LaunchProgram,ToneSphere}"; Flags: nowait postinstall skipifsilent
