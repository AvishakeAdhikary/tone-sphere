; ToneSphere desktop installer hooks (electron-builder NSIS).
; Installs the Driver Pack components that ship with the desktop build:
;  - ToneSphere ASIO driver (64- and 32-bit), registered for every DAW
;  - ToneSphere virtual audio driver (only where Windows can load it; see tonesphere-setup)

!macro customInit
  ; Stop a running engine so its files can be replaced.
  nsExec::Exec 'taskkill /f /im tonesphere-engine.exe'
!macroend

!macro customInstall
  DetailPrint "Registering the ToneSphere ASIO driver..."
  nsExec::ExecToLog '"$INSTDIR\resources\drivers\tonesphere-setup.exe" install-asio "$INSTDIR\resources\drivers" --engine "$INSTDIR\resources\engine\tonesphere-engine.exe" --ui "$INSTDIR\ToneSphere.exe"'
  DetailPrint "Installing ToneSphere virtual audio devices..."
  nsExec::ExecToLog '"$INSTDIR\resources\drivers\tonesphere-setup.exe" install-vad "$INSTDIR\resources\drivers\vad"'
!macroend

!macro customUnInstall
  nsExec::Exec 'taskkill /f /im tonesphere-engine.exe'
  nsExec::ExecToLog '"$INSTDIR\resources\drivers\tonesphere-setup.exe" uninstall-asio "$INSTDIR\resources\drivers"'
  nsExec::ExecToLog '"$INSTDIR\resources\drivers\tonesphere-setup.exe" uninstall-vad'
!macroend
