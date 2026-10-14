; NexTone desktop installer hooks (electron-builder NSIS).
; Installs the Driver Pack components that ship with the desktop build:
;  - NexTone ASIO driver (64- and 32-bit), registered for every DAW
;  - NexTone virtual audio driver (only where Windows can load it; see nextone-setup)

!macro customInit
  ; Stop a running engine so its files can be replaced.
  nsExec::Exec 'taskkill /f /im nextone-engine.exe'
!macroend

!macro customInstall
  DetailPrint "Registering the NexTone ASIO driver..."
  nsExec::ExecToLog '"$INSTDIR\resources\drivers\nextone-setup.exe" install-asio "$INSTDIR\resources\drivers" --engine "$INSTDIR\resources\engine\nextone-engine.exe" --ui "$INSTDIR\NexTone.exe"'
  DetailPrint "Installing NexTone virtual audio devices..."
  nsExec::ExecToLog '"$INSTDIR\resources\drivers\nextone-setup.exe" install-vad "$INSTDIR\resources\drivers\vad"'
!macroend

!macro customUnInstall
  nsExec::Exec 'taskkill /f /im nextone-engine.exe'
  nsExec::ExecToLog '"$INSTDIR\resources\drivers\nextone-setup.exe" uninstall-asio "$INSTDIR\resources\drivers"'
  nsExec::ExecToLog '"$INSTDIR\resources\drivers\nextone-setup.exe" uninstall-vad'
!macroend
