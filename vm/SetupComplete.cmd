@echo off
rem Runs once as SYSTEM at the end of first boot in the NexTone test VM.
bcdedit /set {current} testsigning on
rem Kernel memory dumps for driver debugging.
reg add "HKLM\SYSTEM\CurrentControlSet\Control\CrashControl" /v CrashDumpEnabled /t REG_DWORD /d 2 /f
reg add "HKLM\SYSTEM\CurrentControlSet\Control\CrashControl" /v AutoReboot /t REG_DWORD /d 1 /f
rem Make sure the audio stack runs even without sound hardware.
sc config Audiosrv start= auto
sc config AudioEndpointBuilder start= auto
rem Keep the test VM awake.
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
