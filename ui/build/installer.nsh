; ToneSphere desktop installer hooks (electron-builder NSIS).
; Installs the drivers that ship with the desktop build:
;  - ToneSphere ASIO driver (64- and 32-bit), registered for every DAW; needs no certificate
;  - ToneSphere virtual audio driver, only in builds that carry it (it needs Microsoft's signature,
;    so public builds leave it out; virtual devices then come from VB-CABLE, which the app installs
;    from Settings > Devices > Install drivers).

!macro customInit
  ; Stop a running engine so its files can be replaced.
  nsExec::Exec 'taskkill /f /im tonesphere-engine.exe'
!macroend

!macro customInstall
  DetailPrint "Registering the ToneSphere ASIO driver..."
  nsExec::ExecToLog '"$INSTDIR\resources\drivers\tonesphere-setup.exe" install-asio "$INSTDIR\resources\drivers" --engine "$INSTDIR\resources\engine\tonesphere-engine.exe" --ui "$INSTDIR\ToneSphere.exe"'
  IfFileExists "$INSTDIR\resources\drivers\vad\tonespherevad.inf" 0 +3
    DetailPrint "Installing ToneSphere virtual audio devices..."
    nsExec::ExecToLog '"$INSTDIR\resources\drivers\tonesphere-setup.exe" install-vad "$INSTDIR\resources\drivers\vad"'
!macroend

!macro customUnInstall
  nsExec::Exec 'taskkill /f /im tonesphere-engine.exe'
  nsExec::ExecToLog '"$INSTDIR\resources\drivers\tonesphere-setup.exe" uninstall-asio "$INSTDIR\resources\drivers"'
  nsExec::ExecToLog '"$INSTDIR\resources\drivers\tonesphere-setup.exe" uninstall-vad'
!macroend
