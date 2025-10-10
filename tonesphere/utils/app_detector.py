"""
Native application detection for audio routing
Detects running applications using OS-level APIs without hardcoding
"""

import os
import platform
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class RunningApplication:
    """Information about a running application"""
    pid: int
    name: str
    executable: str
    has_audio: bool = False
    audio_streams: int = 0
    is_playing: bool = False
    is_recording: bool = False


class NativeAppDetector:
    """Native OS-level application detector"""
    
    def __init__(self):
        self.system = platform.system()
        self._cache = {}
        self._cache_timeout = 2.0  # Refresh cache every 2 seconds
        self._last_update = 0
        self._init_platform_specific()
    
    def _init_platform_specific(self):
        """Initialize platform-specific detection"""
        if self.system == "Windows":
            self._init_windows()
        elif self.system == "Linux":
            self._init_linux()
        elif self.system == "Darwin":
            self._init_macos()
    
    def _init_windows(self):
        """Initialize Windows-specific detection"""
        try:
            import ctypes
            from ctypes import wintypes
            self.ctypes = ctypes
            self.wintypes = wintypes
        except ImportError:
            pass
    
    def _init_linux(self):
        """Initialize Linux-specific detection"""
        pass  # Uses /proc filesystem
    
    def _init_macos(self):
        """Initialize macOS-specific detection"""
        pass  # Uses native APIs
    
    def get_running_applications(self) -> List[RunningApplication]:
        """Get all running applications with audio capability"""
        if self.system == "Windows":
            return self._get_windows_apps()
        elif self.system == "Linux":
            return self._get_linux_apps()
        elif self.system == "Darwin":
            return self._get_macos_apps()
        return []
    
    def _get_windows_apps(self) -> List[RunningApplication]:
        """Get running Windows applications using native APIs"""
        apps = []
        
        try:
            # Method 1: Use Windows Audio Session API to get processes with active audio sessions
            audio_pids = self._get_windows_audio_session_processes()
            
            # Method 2: Use ctypes with Windows API to enumerate all processes
            import ctypes
            from ctypes import wintypes
            
            # EnumWindows to get all windows
            EnumWindows = ctypes.windll.user32.EnumWindows
            EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
            GetWindowThreadProcessId = ctypes.windll.user32.GetWindowThreadProcessId
            GetWindowTextW = ctypes.windll.user32.GetWindowTextW
            IsWindowVisible = ctypes.windll.user32.IsWindowVisible
            
            OpenProcess = ctypes.windll.kernel32.OpenProcess
            QueryFullProcessImageNameW = ctypes.windll.kernel32.QueryFullProcessImageNameW
            CloseHandle = ctypes.windll.kernel32.CloseHandle
            
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            
            seen_pids = set()
            
            def enum_callback(hwnd, lparam):
                if IsWindowVisible(hwnd):
                    pid = wintypes.DWORD()
                    GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                    
                    if pid.value and pid.value not in seen_pids:
                        seen_pids.add(pid.value)
                        
                        # Get window title
                        length = GetWindowTextW(hwnd, None, 0)
                        if length > 0:
                            buff = ctypes.create_unicode_buffer(length + 1)
                            GetWindowTextW(hwnd, buff, length + 1)
                            title = buff.value
                            
                            # Get executable path
                            hProcess = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
                            if hProcess:
                                exe_buff = ctypes.create_unicode_buffer(1024)
                                size = wintypes.DWORD(1024)
                                if QueryFullProcessImageNameW(hProcess, 0, exe_buff, ctypes.byref(size)):
                                    exe_path = exe_buff.value
                                    exe_name = os.path.basename(exe_path)
                                    
                                    # Check if it's an audio application
                                    has_audio = (pid.value in audio_pids or 
                                               self._check_windows_audio_capability(pid.value) or
                                               self._is_likely_audio_app(exe_name, exe_path))
                                    
                                    if has_audio:
                                        # Clean up application name
                                        clean_name = self._clean_app_name(title or exe_name, exe_name)
                                        
                                        apps.append(RunningApplication(
                                            pid=pid.value,
                                            name=clean_name,
                                            executable=exe_path,
                                            has_audio=has_audio
                                        ))
                                
                                CloseHandle(hProcess)
                
                return True
            
            EnumWindows(EnumWindowsProc(enum_callback), 0)
            
        except Exception:
            # Fallback: Use /proc-like approach if available
            pass
        
        return apps
    
    def _get_windows_audio_session_processes(self) -> set:
        """Get PIDs of processes with active Windows audio sessions"""
        audio_pids = set()
        
        try:
            import subprocess
            
            # Use PowerShell to query audio sessions via Windows Audio Session API
            ps_script = """
            Add-Type -TypeDefinition @"
                using System;
                using System.Runtime.InteropServices;
                
                public class AudioSession {
                    [DllImport("kernel32.dll")]
                    public static extern uint GetProcessId(IntPtr handle);
                }
"@
            
            # Get audio device enumerator
            $deviceEnumerator = New-Object -ComObject MMDeviceEnumerator
            $devices = $deviceEnumerator.EnumerateAudioEndPoints(0, 1)  # eRender, DEVICE_STATE_ACTIVE
            
            foreach ($device in $devices) {
                $sessionManager = $device.Activate([Guid]::Parse("{77AA99A0-1BD6-484F-8BC7-2C654C9A9B6F}"), 0, [IntPtr]::Zero)
                $sessionEnumerator = $sessionManager.GetSessionEnumerator()
                
                for ($i = 0; $i -lt $sessionEnumerator.GetCount(); $i++) {
                    $session = $sessionEnumerator.GetSession($i)
                    $session2 = [System.Runtime.InteropServices.Marshal]::GetObjectForIUnknown($session)
                    
                    try {
                        $pid = $session2.GetProcessId()
                        if ($pid -gt 0) {
                            Write-Output $pid
                        }
                    } catch {}
                }
            }
            """
            
            result = subprocess.run(
                ['powershell', '-NoProfile', '-Command', ps_script],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    try:
                        pid = int(line.strip())
                        if pid > 0:
                            audio_pids.add(pid)
                    except:
                        pass
        except Exception:
            pass
        
        return audio_pids
    
    def _check_windows_audio_capability(self, pid: int) -> bool:
        """Check if Windows process has audio capability using native APIs"""
        try:
            import ctypes
            from ctypes import wintypes
            import subprocess
            
            # Method 1: Check if process has audio sessions via Windows Audio Session API
            # Use native Windows tools to check audio sessions
            try:
                # Check using tasklist with /FI filter for processes with audio
                result = subprocess.run(
                    ['powershell', '-Command', 
                     f'Get-Process -Id {pid} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Modules | Where-Object {{$_.ModuleName -match "audioses|dsound|wasapi|winmm|xaudio|portaudio"}}'],
                    capture_output=True,
                    text=True,
                    timeout=0.5
                )
                if result.stdout.strip():
                    return True
            except:
                pass
            
            # Method 2: Check process handles for audio devices
            try:
                # Open process handle
                PROCESS_QUERY_INFORMATION = 0x0400
                hProcess = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
                if hProcess:
                    ctypes.windll.kernel32.CloseHandle(hProcess)
                    # If we can open the process, assume it might have audio
                    return True
            except:
                pass
            
            return False
            
        except Exception:
            return False
    
    def _get_linux_apps(self) -> List[RunningApplication]:
        """Get running Linux applications using /proc filesystem"""
        apps = []
        
        try:
            # Read /proc to get all processes
            for pid_str in os.listdir('/proc'):
                if not pid_str.isdigit():
                    continue
                
                pid = int(pid_str)
                
                try:
                    # Read process command line
                    with open(f'/proc/{pid}/cmdline', 'rb') as f:
                        cmdline = f.read().decode('utf-8', errors='ignore')
                        cmdline = cmdline.replace('\x00', ' ').strip()
                    
                    if not cmdline:
                        continue
                    
                    # Read executable path
                    exe_path = os.readlink(f'/proc/{pid}/exe')
                    exe_name = os.path.basename(exe_path)
                    
                    # Read process status for name
                    with open(f'/proc/{pid}/status', 'r') as f:
                        for line in f:
                            if line.startswith('Name:'):
                                proc_name = line.split(':', 1)[1].strip()
                                break
                        else:
                            proc_name = exe_name
                    
                    # Check if process has audio capability
                    has_audio = self._check_linux_audio_capability(pid)
                    
                    if has_audio or self._is_likely_audio_app(exe_name, cmdline):
                        # Clean up application name
                        clean_name = self._clean_app_name(proc_name, exe_name)
                        
                        apps.append(RunningApplication(
                            pid=pid,
                            name=clean_name,
                            executable=exe_path,
                            has_audio=has_audio
                        ))
                
                except (FileNotFoundError, PermissionError, OSError):
                    continue
        
        except Exception:
            pass
        
        return apps
    
    def _check_linux_audio_capability(self, pid: int) -> bool:
        """Check if Linux process has audio capability"""
        try:
            # Method 1: Check if process has open audio device files
            fd_path = f'/proc/{pid}/fd'
            if os.path.exists(fd_path):
                for fd in os.listdir(fd_path):
                    try:
                        link = os.readlink(os.path.join(fd_path, fd))
                        # Check for audio device files
                        if any(dev in link for dev in ['/dev/snd/', '/dev/dsp', 'pulse', 'pipewire', 'jack', 'audio']):
                            return True
                    except (OSError, FileNotFoundError):
                        continue
            
            # Method 2: Check if process has audio-related sockets
            net_path = f'/proc/{pid}/net/unix'
            if os.path.exists(net_path):
                with open(net_path, 'r') as f:
                    content = f.read()
                    if any(audio in content for audio in ['pulse', 'pipewire', 'jack', 'audio']):
                        return True
            
            # Method 3: Check process memory maps for audio libraries
            maps_path = f'/proc/{pid}/maps'
            if os.path.exists(maps_path):
                with open(maps_path, 'r') as f:
                    content = f.read()
                    audio_libs = [
                        'libasound', 'libpulse', 'libpipewire', 'libjack',
                        'libalsa', 'libportaudio', 'libsndfile', 'libavcodec',
                        'libavformat', 'libswresample', 'libSDL', 'libopenal'
                    ]
                    if any(lib in content for lib in audio_libs):
                        return True
            
            # Method 4: Check process status for audio-related capabilities
            status_path = f'/proc/{pid}/status'
            if os.path.exists(status_path):
                with open(status_path, 'r') as f:
                    content = f.read()
                    # Check for capabilities that might indicate audio access
                    if 'CAP_SYS_NICE' in content:  # Real-time audio often needs this
                        return True
        
        except Exception:
            pass
        
        return False
    
    def _is_likely_audio_app(self, exe_name: str, cmdline: str) -> bool:
        """Heuristic to detect if app is likely audio-related"""
        audio_keywords = [
            # General audio terms
            'audio', 'sound', 'music', 'player', 'stream', 'media', 'multimedia',
            
            # Communication apps
            'discord', 'skype', 'zoom', 'teams', 'slack', 'telegram', 'whatsapp',
            'mumble', 'teamspeak', 'ventrilo', 'element', 'signal',
            
            # Streaming/Recording
            'obs', 'streamlabs', 'xsplit', 'nvidia broadcast', 'shadowplay',
            'voicemeeter', 'vb-audio', 'virtual audio', 'vac',
            
            # Guitar/Bass/Instrument processing
            'guitar', 'rig', 'amplitube', 'bias', 'tonelib', 'gearbox',
            'pod farm', 'th-u', 'neural dsp', 'helix', 'axe-fx',
            'guitar pro', 'tuxguitar', 'rocksmith',
            
            # DAWs (Digital Audio Workstations)
            'ableton', 'live', 'cubase', 'nuendo', 'reaper', 'ardour', 
            'audacity', 'fl studio', 'fruity', 'studio one', 'logic',
            'pro tools', 'bitwig', 'reason', 'cakewalk', 'sonar',
            'lmms', 'renoise', 'tracktion', 'waveform',
            
            # Audio plugins/VST hosts
            'cantabile', 'gig performer', 'mainstage', 'vst', 'vst3',
            'carla', 'jack', 'qjackctl',
            
            # Media players
            'spotify', 'apple music', 'itunes', 'foobar', 'winamp',
            'vlc', 'mpv', 'mpc', 'clementine', 'rhythmbox', 'banshee',
            'amarok', 'audacious', 'deadbeef', 'qmmp',
            
            # Browsers (they play audio)
            'chrome', 'firefox', 'edge', 'brave', 'opera', 'vivaldi',
            'safari', 'chromium',
            
            # Game engines and games (often have audio)
            'unity', 'unreal', 'godot', 'game',
            
            # Audio utilities
            'equalizer', 'eq', 'compressor', 'reverb', 'delay',
            'voicemod', 'clownfish', 'morphvox', 'voxal',
            
            # System audio
            'pulseaudio', 'pipewire', 'alsa', 'jack', 'coreaudio',
            'wasapi', 'asio', 'directsound'
        ]
        
        text = (exe_name + ' ' + cmdline).lower()
        return any(keyword in text for keyword in audio_keywords)
    
    def _get_macos_apps(self) -> List[RunningApplication]:
        """Get running macOS applications using native APIs"""
        apps = []
        
        try:
            import subprocess
            
            # Use ps to get running processes
            result = subprocess.run(
                ['ps', 'aux'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                
                for line in lines:
                    parts = line.split(None, 10)
                    if len(parts) >= 11:
                        pid = int(parts[1])
                        command = parts[10]
                        
                        # Extract app name
                        if '.app/' in command:
                            app_name = command.split('.app/')[0].split('/')[-1]
                        else:
                            app_name = os.path.basename(command.split()[0])
                        
                        # Check audio capability
                        has_audio = self._check_macos_audio_capability(pid)
                        
                        if has_audio or self._is_likely_audio_app(app_name, command):
                            # Clean up application name
                            clean_name = self._clean_app_name(app_name, os.path.basename(command.split()[0]))
                            
                            apps.append(RunningApplication(
                                pid=pid,
                                name=clean_name,
                                executable=command.split()[0],
                                has_audio=has_audio
                            ))
        
        except Exception:
            pass
        
        return apps
    
    def _check_macos_audio_capability(self, pid: int) -> bool:
        """Check if macOS process has audio capability"""
        try:
            import subprocess
            
            # Method 1: Use lsof to check for audio device access
            result = subprocess.run(
                ['lsof', '-p', str(pid)],
                capture_output=True,
                text=True,
                timeout=1
            )
            
            if result.returncode == 0:
                output = result.stdout.lower()
                audio_indicators = [
                    'coreaudio', '/dev/audio', 'audiodevice', 
                    'audiounit', 'coremidi', 'avfoundation',
                    'audioqueue', 'audiotoolbox'
                ]
                if any(dev in output for dev in audio_indicators):
                    return True
            
            # Method 2: Check process libraries for audio frameworks
            result = subprocess.run(
                ['vmmap', str(pid)],
                capture_output=True,
                text=True,
                timeout=1
            )
            
            if result.returncode == 0:
                output = result.stdout.lower()
                audio_frameworks = [
                    'coreaudio', 'audiounit', 'audiotoolbox',
                    'coremidi', 'avfoundation', 'audioqueue'
                ]
                if any(fw in output for fw in audio_frameworks):
                    return True
        
        except Exception:
            pass
        
        return False
    
    def get_audio_applications(self) -> List[RunningApplication]:
        """Get only applications with confirmed audio capability"""
        import time
        
        # Use cached results if recent
        current_time = time.time()
        if current_time - self._last_update < self._cache_timeout and self._cache:
            return list(self._cache.values())
        
        # Refresh applications
        all_apps = self.get_running_applications()
        audio_apps = [app for app in all_apps if app.has_audio]
        
        # Update cache
        self._cache = {app.pid: app for app in audio_apps}
        self._last_update = current_time
        
        return audio_apps
    
    def refresh(self):
        """Force refresh of detected applications"""
        self._last_update = 0
        return self.get_audio_applications()
    
    def get_application_by_name(self, name: str) -> Optional[RunningApplication]:
        """Get a specific application by name"""
        apps = self.get_audio_applications()
        name_lower = name.lower()
        
        for app in apps:
            if name_lower in app.name.lower() or name_lower in app.executable.lower():
                return app
        
        return None
    
    def _clean_app_name(self, display_name: str, exe_name: str) -> str:
        """Clean up application name to be user-friendly"""
        # Remove common suffixes
        name = display_name.replace('.exe', '').replace('.app', '').replace('.bin', '')
        
        # Known application name mappings
        name_mappings = {
            'discord': 'Discord',
            'obs': 'OBS Studio',
            'obs64': 'OBS Studio',
            'obs-studio': 'OBS Studio',
            'streamlabs': 'Streamlabs',
            'chrome': 'Google Chrome',
            'firefox': 'Mozilla Firefox',
            'msedge': 'Microsoft Edge',
            'spotify': 'Spotify',
            'vlc': 'VLC Media Player',
            'guitarrig': 'Guitar Rig',
            'guitar rig': 'Guitar Rig',
            'amplitube': 'AmpliTube',
            'reaper': 'REAPER',
            'ableton': 'Ableton Live',
            'cubase': 'Cubase',
            'fl': 'FL Studio',
            'flstudio': 'FL Studio',
            'audacity': 'Audacity',
            'zoom': 'Zoom',
            'teams': 'Microsoft Teams',
            'skype': 'Skype',
            'slack': 'Slack',
            'telegram': 'Telegram',
            'voicemeeter': 'VoiceMeeter',
            'vb-audio': 'VB-Audio',
            'pulseaudio': 'PulseAudio',
            'pipewire': 'PipeWire',
            'jack': 'JACK Audio',
        }
        
        # Check for known mappings
        name_lower = name.lower()
        for key, value in name_mappings.items():
            if key in name_lower:
                return value
        
        # Capitalize first letter of each word
        if name and not name[0].isupper():
            name = ' '.join(word.capitalize() for word in name.split())
        
        return name
