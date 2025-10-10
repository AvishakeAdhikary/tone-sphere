"""
System Tray Integration for ToneSphere
Provides native system tray icon with context menu
"""

import platform
import threading
import os
from pathlib import Path
from typing import Callable, Optional


class SystemTrayIcon:
    """Cross-platform system tray icon manager"""
    
    def __init__(self, on_show_gui: Callable, on_exit: Callable, engine=None):
        self.on_show_gui = on_show_gui
        self.on_exit = on_exit
        self.engine = engine
        self.system = platform.system()
        self.tray_thread = None
        self.running = False
        self.icon = None
        
        # Get icon path
        self.icon_path = self._get_icon_path()
        
    def _get_icon_path(self) -> Optional[str]:
        """Get the path to the icon file"""
        # Try to find the icon in assets/images
        possible_paths = [
            Path(__file__).parent.parent.parent / "assets" / "images" / "ToneSphere.png",
            Path(__file__).parent.parent.parent / "assets" / "images" / "ToneSphere.ico",
            Path("assets/images/ToneSphere.png"),
            Path("assets/images/ToneSphere.ico"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def start(self):
        """Start the system tray icon"""
        if self.system == "Windows":
            self._start_windows_tray()
        elif self.system == "Linux":
            self._start_linux_tray()
        elif self.system == "Darwin":
            self._start_macos_tray()
    
    def stop(self):
        """Stop the system tray icon"""
        self.running = False
        if self.icon:
            try:
                self.icon.stop()
            except:
                pass
    
    def _start_windows_tray(self):
        """Start Windows system tray using pystray"""
        try:
            import pystray
            from PIL import Image
            
            # Load icon image
            def create_image():
                if self.icon_path and os.path.exists(self.icon_path):
                    try:
                        return Image.open(self.icon_path)
                    except:
                        pass
                
                # Fallback: create simple icon
                from PIL import ImageDraw
                width = 64
                height = 64
                image = Image.new('RGB', (width, height), color='#ff8844')
                dc = ImageDraw.Draw(image)
                dc.ellipse([16, 16, 48, 48], fill='#ffffff')
                return image
            
            # Create menu with engine controls
            menu_items = [
                pystray.MenuItem('Show ToneSphere', self.on_show_gui, default=True),
                pystray.Menu.SEPARATOR,
            ]
            
            # Add engine controls if engine is available
            if self.engine:
                menu_items.extend([
                    pystray.MenuItem('Start Engine', self._start_engine, enabled=lambda item: not self._is_engine_running()),
                    pystray.MenuItem('Stop Engine', self._stop_engine, enabled=lambda item: self._is_engine_running()),
                    pystray.Menu.SEPARATOR,
                    pystray.MenuItem('Refresh Devices', self._refresh_devices, enabled=lambda item: self._is_engine_running()),
                    pystray.Menu.SEPARATOR,
                ])
            
            menu_items.append(pystray.MenuItem('Exit', self.on_exit))
            
            menu = pystray.Menu(*menu_items)
            
            # Create and run icon
            self.icon = pystray.Icon('ToneSphere', create_image(), 'ToneSphere Audio', menu)
            self.running = True
            
            # Run in separate thread
            self.tray_thread = threading.Thread(target=self.icon.run, daemon=True)
            self.tray_thread.start()
            
        except ImportError:
            # Fallback: use native Windows API
            self._start_windows_native_tray()
    
    def _start_windows_native_tray(self):
        """Start Windows system tray using native Win32 API"""
        try:
            import ctypes
            from ctypes import wintypes
            import win32gui
            import win32con
            
            class NOTIFYICONDATA(ctypes.Structure):
                _fields_ = [
                    ('cbSize', wintypes.DWORD),
                    ('hWnd', wintypes.HWND),
                    ('uID', wintypes.UINT),
                    ('uFlags', wintypes.UINT),
                    ('uCallbackMessage', wintypes.UINT),
                    ('hIcon', wintypes.HICON),
                    ('szTip', wintypes.WCHAR * 128),
                ]
            
            def wndproc(hwnd, msg, wparam, lparam):
                if msg == win32con.WM_USER + 20:
                    if lparam == win32con.WM_LBUTTONDBLCLK:
                        self.on_show_gui()
                    elif lparam == win32con.WM_RBUTTONUP:
                        # Show context menu
                        pass
                return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)
            
            # Create window class
            wc = win32gui.WNDCLASS()
            wc.lpfnWndProc = wndproc
            wc.lpszClassName = 'ToneSphere'
            wc.hInstance = win32gui.GetModuleHandle(None)
            
            class_atom = win32gui.RegisterClass(wc)
            hwnd = win32gui.CreateWindow(class_atom, 'ToneSphere', 0, 0, 0, 0, 0, 0, 0, wc.hInstance, None)
            
            # Add tray icon
            nid = NOTIFYICONDATA()
            nid.cbSize = ctypes.sizeof(NOTIFYICONDATA)
            nid.hWnd = hwnd
            nid.uID = 1
            nid.uFlags = win32con.NIF_ICON | win32con.NIF_MESSAGE | win32con.NIF_TIP
            nid.uCallbackMessage = win32con.WM_USER + 20
            nid.szTip = 'ToneSphere Audio'
            
            # Load icon (use default for now)
            nid.hIcon = win32gui.LoadIcon(0, win32con.IDI_APPLICATION)
            
            shell32 = ctypes.windll.shell32
            shell32.Shell_NotifyIconW(1, ctypes.byref(nid))  # NIM_ADD
            
            self.running = True
            
        except Exception:
            pass
    
    def _start_linux_tray(self):
        """Start Linux system tray using pystray or AppIndicator"""
        try:
            import pystray
            from PIL import Image
            
            # Load icon image
            def create_image():
                if self.icon_path and os.path.exists(self.icon_path):
                    try:
                        return Image.open(self.icon_path)
                    except:
                        pass
                
                # Fallback: create simple icon
                from PIL import ImageDraw
                width = 64
                height = 64
                image = Image.new('RGBA', (width, height), color=(255, 136, 68, 255))
                dc = ImageDraw.Draw(image)
                dc.ellipse([16, 16, 48, 48], fill=(255, 255, 255, 255))
                return image
            
            # Create menu with engine controls
            menu_items = [
                pystray.MenuItem('Show ToneSphere', self.on_show_gui, default=True),
                pystray.Menu.SEPARATOR,
            ]
            
            # Add engine controls if engine is available
            if self.engine:
                menu_items.extend([
                    pystray.MenuItem('Start Engine', self._start_engine, enabled=lambda item: not self._is_engine_running()),
                    pystray.MenuItem('Stop Engine', self._stop_engine, enabled=lambda item: self._is_engine_running()),
                    pystray.Menu.SEPARATOR,
                    pystray.MenuItem('Refresh Devices', self._refresh_devices, enabled=lambda item: self._is_engine_running()),
                    pystray.Menu.SEPARATOR,
                ])
            
            menu_items.append(pystray.MenuItem('Exit', self.on_exit))
            
            menu = pystray.Menu(*menu_items)
            
            # Create and run icon
            self.icon = pystray.Icon('ToneSphere', create_image(), 'ToneSphere Audio', menu)
            self.running = True
            
            # Run in separate thread
            self.tray_thread = threading.Thread(target=self.icon.run, daemon=True)
            self.tray_thread.start()
            
        except ImportError:
            # Try AppIndicator3 for better Linux integration
            try:
                import gi
                gi.require_version('Gtk', '3.0')
                gi.require_version('AppIndicator3', '0.1')
                from gi.repository import Gtk, AppIndicator3
                
                # Use icon file if available
                icon_name = 'audio-card'
                if self.icon_path and os.path.exists(self.icon_path):
                    icon_name = self.icon_path
                
                indicator = AppIndicator3.Indicator.new(
                    'tonesphere',
                    icon_name,
                    AppIndicator3.IndicatorCategory.APPLICATION_STATUS
                )
                indicator.set_status(AppIndicator3.IndicatorStatus.ACTIVE)
                
                menu = Gtk.Menu()
                
                show_item = Gtk.MenuItem(label='Show ToneSphere')
                show_item.connect('activate', lambda x: self.on_show_gui())
                menu.append(show_item)
                
                # Add engine controls if engine is available
                if self.engine:
                    menu.append(Gtk.SeparatorMenuItem())
                    
                    start_item = Gtk.MenuItem(label='Start Engine')
                    start_item.connect('activate', lambda x: self._start_engine())
                    menu.append(start_item)
                    
                    stop_item = Gtk.MenuItem(label='Stop Engine')
                    stop_item.connect('activate', lambda x: self._stop_engine())
                    menu.append(stop_item)
                    
                    menu.append(Gtk.SeparatorMenuItem())
                    
                    refresh_item = Gtk.MenuItem(label='Refresh Devices')
                    refresh_item.connect('activate', lambda x: self._refresh_devices())
                    menu.append(refresh_item)
                    
                    menu.append(Gtk.SeparatorMenuItem())
                
                exit_item = Gtk.MenuItem(label='Exit')
                exit_item.connect('activate', lambda x: self.on_exit())
                menu.append(exit_item)
                
                menu.show_all()
                indicator.set_menu(menu)
                
                self.running = True
                
            except Exception:
                pass
    
    def _start_macos_tray(self):
        """Start macOS system tray using pystray or rumps"""
        try:
            import pystray
            from PIL import Image
            
            # Load icon image
            def create_image():
                if self.icon_path and os.path.exists(self.icon_path):
                    try:
                        return Image.open(self.icon_path)
                    except:
                        pass
                
                # Fallback: create simple icon
                from PIL import ImageDraw
                width = 64
                height = 64
                image = Image.new('RGBA', (width, height), color=(255, 136, 68, 255))
                dc = ImageDraw.Draw(image)
                dc.ellipse([16, 16, 48, 48], fill=(255, 255, 255, 255))
                return image
            
            # Create menu with engine controls
            menu_items = [
                pystray.MenuItem('Show ToneSphere', self.on_show_gui, default=True),
                pystray.Menu.SEPARATOR,
            ]
            
            # Add engine controls if engine is available
            if self.engine:
                menu_items.extend([
                    pystray.MenuItem('Start Engine', self._start_engine, enabled=lambda item: not self._is_engine_running()),
                    pystray.MenuItem('Stop Engine', self._stop_engine, enabled=lambda item: self._is_engine_running()),
                    pystray.Menu.SEPARATOR,
                    pystray.MenuItem('Refresh Devices', self._refresh_devices, enabled=lambda item: self._is_engine_running()),
                    pystray.Menu.SEPARATOR,
                ])
            
            menu_items.append(pystray.MenuItem('Exit', self.on_exit))
            
            menu = pystray.Menu(*menu_items)
            
            # Create and run icon
            self.icon = pystray.Icon('ToneSphere', create_image(), 'ToneSphere Audio', menu)
            self.running = True
            
            # Run in separate thread
            self.tray_thread = threading.Thread(target=self.icon.run, daemon=True)
            self.tray_thread.start()
            
        except ImportError:
            # Try rumps for better macOS integration
            try:
                import rumps
                
                class ToneSphereApp(rumps.App):
                    def __init__(self, on_show, on_exit, engine, icon_path):
                        # Use icon if available
                        icon = '🎵'
                        if icon_path and os.path.exists(icon_path):
                            icon = icon_path
                        
                        super(ToneSphereApp, self).__init__('ToneSphere', icon)
                        self.on_show = on_show
                        self.on_exit_callback = on_exit
                        self.engine = engine
                        
                        # Build menu
                        menu_items = ['Show ToneSphere']
                        if engine:
                            menu_items.extend([None, 'Start Engine', 'Stop Engine', None, 'Refresh Devices', None])
                        menu_items.append('Exit')
                        self.menu = menu_items
                    
                    @rumps.clicked('Show ToneSphere')
                    def show(self, _):
                        self.on_show()
                    
                    @rumps.clicked('Start Engine')
                    def start_engine(self, _):
                        if self.engine and not self.engine.is_running:
                            try:
                                self.engine.start_engine()
                            except:
                                pass
                    
                    @rumps.clicked('Stop Engine')
                    def stop_engine(self, _):
                        if self.engine and self.engine.is_running:
                            try:
                                self.engine.stop_engine()
                            except:
                                pass
                    
                    @rumps.clicked('Refresh Devices')
                    def refresh_devices(self, _):
                        if self.engine and self.engine.is_running:
                            try:
                                self.engine.refresh_devices()
                            except:
                                pass
                    
                    @rumps.clicked('Exit')
                    def exit(self, _):
                        self.on_exit_callback()
                        rumps.quit_application()
                
                app = ToneSphereApp(self.on_show_gui, self.on_exit, self.engine, self.icon_path)
                self.running = True
                
                # Run in separate thread
                self.tray_thread = threading.Thread(target=app.run, daemon=True)
                self.tray_thread.start()
                
            except Exception:
                pass
    
    def _is_engine_running(self) -> bool:
        """Check if engine is running"""
        if self.engine:
            return getattr(self.engine, 'is_running', False)
        return False
    
    def _start_engine(self):
        """Start the audio engine"""
        if self.engine and not self._is_engine_running():
            try:
                if not hasattr(self.engine, 'initialize') or getattr(self.engine, '_initialized', False):
                    self.engine.start_engine()
                else:
                    self.engine.initialize()
                    self.engine.start_engine()
            except Exception as e:
                print(f"Failed to start engine: {e}")
    
    def _stop_engine(self):
        """Stop the audio engine"""
        if self.engine and self._is_engine_running():
            try:
                self.engine.stop_engine()
            except Exception as e:
                print(f"Failed to stop engine: {e}")
    
    def _refresh_devices(self):
        """Refresh audio devices"""
        if self.engine and self._is_engine_running():
            try:
                if hasattr(self.engine, 'refresh_devices'):
                    self.engine.refresh_devices()
            except Exception as e:
                print(f"Failed to refresh devices: {e}")
