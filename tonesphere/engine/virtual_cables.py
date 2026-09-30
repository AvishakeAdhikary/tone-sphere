"""
The ToneSphere virtual cables on Windows, as the user manages them.

Every cable is its own PnP device — an instance of `ROOT\\ToneSphereVirtualAudio`, driven by
`driver/windows_virtual_audio` — with a render endpoint and a capture endpoint joined inside
the driver, and a buffer of its own. So Windows' own device operations act on exactly one
cable: disabling one takes its two endpoints away and leaves the others playing.

Reading needs no privilege: SetupAPI and CfgMgr32, through ctypes, list the instances with
their names and state. Changing anything — adding, renaming, disabling, enabling, removing a
cable, removing the driver — changes the machine's device configuration, which Windows
reserves to administrators. Those run in a child process started with the `runas` verb, so
Windows asks the user once (its own prompt) and nothing else in ToneSphere ever runs
elevated; the child writes its result to a file the parent reads back. A process that is
already elevated (the driver test VM) does them in-process.

A cable a program has open cannot be taken away from it: Windows' audio engine (audiodg.exe)
vetoes the removal, and a plain `pnputil /disable-device` then leaves the cable "pending a
restart" — still working, and refusing every later change until Windows restarts. So every
change to an existing cable first asks for the device with the "no UI" flags, which turn a
veto into a clean refusal (`CableInUse`) with nothing changed.

The driver must be test-signed or attestation-signed to load at all; see
docs/VIRTUAL_AUDIO_DRIVER.md.
"""

import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
from ctypes import wintypes
from dataclasses import asdict, dataclass

HARDWARE_ID = 'ROOT\\ToneSphereVirtualAudio'
CLASS_MEDIA = '{4d36e96c-e325-11ce-bfc1-08002be10318}'
MAX_CABLES = 8
DEFAULT_NAMES = ('ToneSphere Cable 1', 'ToneSphere Cable 2')

DIGCF_PRESENT = 0x2
DIGCF_ALLCLASSES = 0x4
SPDRP_DEVICEDESC = 0x0
SPDRP_HARDWAREID = 0x1
SPDRP_FRIENDLYNAME = 0xC
DICD_GENERATE_ID = 0x1
DIF_REGISTERDEVICE = 0x19
DIF_REMOVE = 0x5
CR_SUCCESS = 0
CR_REMOVE_VETOED = 0x17
CM_DISABLE_UI_NOT_OK = 0x4
CM_DISABLE_PERSIST = 0x8
# The audio engine also opens a cable by itself for a moment — when one arrives, or when it
# becomes the default device because another left — and vetoes a stop meanwhile. Only a
# veto that outlasts this is a program holding the cable.
VETO_PATIENCE_S = 10.0
CM_PROB_DISABLED = 22
DN_STARTED = 0x8
INVALID_HANDLE = ctypes.c_void_p(-1).value


class CableError(RuntimeError):
    """A cable operation Windows refused, with its reason."""


class ElevationRefused(CableError):
    """The user declined Windows' administrator prompt."""


class CableInUse(CableError):
    """A program has the cable open, and Windows will not take it away from under it."""


IN_USE = ("a program has this cable open, and Windows' audio engine will not take a device away from a "
          "program using it; close that program, or stop it using the cable, and try again")


@dataclass(frozen=True)
class Cable:
    instance_id: str
    name: str
    enabled: bool
    problem: int | None       # a CM_PROB_* code while the device is not working; None when it is
    started: bool

    @property
    def state(self) -> str:
        if not self.enabled:
            return 'disabled'
        return 'working' if self.started and self.problem is None else f'problem {self.problem}'


class GUID(ctypes.Structure):
    _fields_ = [('Data1', wintypes.DWORD), ('Data2', wintypes.WORD), ('Data3', wintypes.WORD),
                ('Data4', ctypes.c_ubyte * 8)]

    @classmethod
    def parse(cls, text: str) -> 'GUID':
        guid = cls()
        ctypes.oledll.ole32.CLSIDFromString(text, ctypes.byref(guid))
        return guid


class SP_DEVINFO_DATA(ctypes.Structure):
    _fields_ = [('cbSize', wintypes.DWORD), ('ClassGuid', GUID), ('DevInst', wintypes.DWORD),
                ('Reserved', ctypes.c_size_t)]

    def __init__(self):
        super().__init__()
        self.cbSize = ctypes.sizeof(SP_DEVINFO_DATA)


def supported() -> bool:
    return sys.platform == 'win32'


def _setupapi():
    api = ctypes.WinDLL('setupapi', use_last_error=True)
    api.SetupDiGetClassDevsW.restype = ctypes.c_void_p
    api.SetupDiGetClassDevsW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, wintypes.HWND, wintypes.DWORD]
    api.SetupDiEnumDeviceInfo.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(SP_DEVINFO_DATA)]
    api.SetupDiGetDeviceRegistryPropertyW.argtypes = [ctypes.c_void_p, ctypes.POINTER(SP_DEVINFO_DATA), wintypes.DWORD,
                                                      ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p, wintypes.DWORD,
                                                      ctypes.POINTER(wintypes.DWORD)]
    api.SetupDiSetDeviceRegistryPropertyW.argtypes = [ctypes.c_void_p, ctypes.POINTER(SP_DEVINFO_DATA), wintypes.DWORD,
                                                      ctypes.c_void_p, wintypes.DWORD]
    api.SetupDiGetDeviceInstanceIdW.argtypes = [ctypes.c_void_p, ctypes.POINTER(SP_DEVINFO_DATA), wintypes.LPWSTR,
                                                wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    api.SetupDiDestroyDeviceInfoList.argtypes = [ctypes.c_void_p]
    api.SetupDiCreateDeviceInfoList.restype = ctypes.c_void_p
    api.SetupDiCreateDeviceInfoList.argtypes = [ctypes.POINTER(GUID), wintypes.HWND]
    api.SetupDiCreateDeviceInfoW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, ctypes.POINTER(GUID), wintypes.LPCWSTR,
                                             wintypes.HWND, wintypes.DWORD, ctypes.POINTER(SP_DEVINFO_DATA)]
    api.SetupDiCallClassInstaller.argtypes = [wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(SP_DEVINFO_DATA)]
    api.SetupDiOpenDeviceInfoW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, wintypes.HWND, wintypes.DWORD,
                                           ctypes.POINTER(SP_DEVINFO_DATA)]
    return api


def _property(api, devs, data, which: int) -> str | None:
    kind, size = wintypes.DWORD(), wintypes.DWORD()
    buffer = ctypes.create_unicode_buffer(1024)
    if not api.SetupDiGetDeviceRegistryPropertyW(devs, ctypes.byref(data), which, ctypes.byref(kind), buffer,
                                                 ctypes.sizeof(buffer), ctypes.byref(size)):
        return None
    return buffer[:max(0, size.value // 2 - 1)]


def _status(devinst: int) -> tuple[bool, int | None]:
    status, problem = wintypes.ULONG(), wintypes.ULONG()
    cfgmgr = ctypes.windll.cfgmgr32
    if cfgmgr.CM_Get_DevNode_Status(ctypes.byref(status), ctypes.byref(problem), devinst, 0) != CR_SUCCESS:
        return False, None
    return bool(status.value & DN_STARTED), problem.value or None


def cables() -> list[Cable]:
    """Every ToneSphere cable Windows has, working or disabled."""
    if not supported():
        return []
    api = _setupapi()
    devs = api.SetupDiGetClassDevsW(None, 'ROOT', None, DIGCF_ALLCLASSES | DIGCF_PRESENT)
    if devs in (None, INVALID_HANDLE):
        raise CableError(f"SetupDiGetClassDevs failed ({ctypes.get_last_error()})")
    found = []
    try:
        index, data = 0, SP_DEVINFO_DATA()
        while api.SetupDiEnumDeviceInfo(devs, index, ctypes.byref(data)):
            index += 1
            ids = _property(api, devs, data, SPDRP_HARDWAREID) or ''
            if HARDWARE_ID.lower() not in ids.lower().split('\x00'):
                continue
            instance = ctypes.create_unicode_buffer(512)
            api.SetupDiGetDeviceInstanceIdW(devs, ctypes.byref(data), instance, 512, None)
            started, problem = _status(data.DevInst)
            name = _property(api, devs, data, SPDRP_FRIENDLYNAME) or _property(api, devs, data, SPDRP_DEVICEDESC) or ''
            found.append(Cable(instance.value, name, problem != CM_PROB_DISABLED, problem, started))
    finally:
        api.SetupDiDestroyDeviceInfoList(devs)
    return sorted(found, key=lambda c: c.instance_id)


def driver_package() -> str | None:
    """The published name (oemNN.inf) of the ToneSphere driver in the driver store, if it is there."""
    if not supported():
        return None
    listing = subprocess.run(['pnputil', '/enum-drivers'], capture_output=True, text=True,
                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).stdout
    block: dict[str, str] = {}
    for line in listing.splitlines() + ['']:
        if ':' in line:
            key, _, value = line.partition(':')
            block[key.strip().lower()] = value.strip()
        elif block:
            if block.get('original name', '').lower() == 'simpleaudiosample.inf' and \
                    'neural nexus' in block.get('provider name', '').lower():
                return block.get('published name')
            block = {}
    return None


def is_elevated() -> bool:
    return supported() and bool(ctypes.windll.shell32.IsUserAnAdmin())


# --- The privileged operations (run elevated) ---

def _open(api, instance_id: str):
    devs = api.SetupDiCreateDeviceInfoList(None, None)
    data = SP_DEVINFO_DATA()
    if not api.SetupDiOpenDeviceInfoW(devs, instance_id, None, 0, ctypes.byref(data)):
        api.SetupDiDestroyDeviceInfoList(devs)
        raise CableError(f"no device {instance_id} ({ctypes.get_last_error()})")
    return devs, data


def _set_name(api, devs, data, name: str):
    value = ctypes.create_unicode_buffer(name)
    if not api.SetupDiSetDeviceRegistryPropertyW(devs, ctypes.byref(data), SPDRP_FRIENDLYNAME, value,
                                                 ctypes.sizeof(value)):
        raise CableError(f"could not name the cable ({ctypes.get_last_error()})")


class RestartRequired(CableError):
    """Windows accepted the change but will make it only at the next restart."""


def _pnputil(*args: str):
    result = subprocess.run(['pnputil', *args], capture_output=True, text=True,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    output = (result.stdout + result.stderr).strip()
    # 3010, or "pending system reboot": the device is still there, working, until Windows
    # restarts, so the change is reported as not made rather than as done.
    if result.returncode == 3010 or 'pending system reboot' in output:
        raise RestartRequired(f"Windows will finish this at the next restart: {output[-300:]}")
    if result.returncode != 0:
        raise CableError(f"pnputil {' '.join(args)}: {output[-400:]}")


def _devinst(instance_id: str) -> int:
    devinst = wintypes.DWORD()
    cr = ctypes.windll.cfgmgr32.CM_Locate_DevNodeW(ctypes.byref(devinst), ctypes.c_wchar_p(instance_id), 0)
    if cr != CR_SUCCESS:
        raise CableError(f"no device {instance_id} (CONFIGRET {cr:#x})")
    return devinst.value


def _stop(instance_id: str, persist: bool):
    """Disable the device, or refuse with CableInUse, changing nothing, if a program holds it."""
    # Stopped first without persisting: a vetoed persistent disable still records the
    # disable for the next restart, and the device then refuses to be enabled until one.
    deadline = time.monotonic() + VETO_PATIENCE_S
    while True:
        cr = ctypes.windll.cfgmgr32.CM_Disable_DevNode(_devinst(instance_id), CM_DISABLE_UI_NOT_OK)
        if cr != CR_REMOVE_VETOED:
            break
        if time.monotonic() >= deadline:
            raise CableInUse(IN_USE)
        time.sleep(0.5)
    if cr == CR_SUCCESS and persist:
        cr = ctypes.windll.cfgmgr32.CM_Disable_DevNode(_devinst(instance_id), CM_DISABLE_UI_NOT_OK | CM_DISABLE_PERSIST)
    if cr != CR_SUCCESS:
        raise CableError(f"could not stop {instance_id} (CONFIGRET {cr:#x})")


def _admin_add(name: str) -> dict:
    """What devcon install does, for one new device only: register it, then install its driver."""
    if len(cables()) >= MAX_CABLES:
        raise CableError(f"at most {MAX_CABLES} cables")
    if driver_package() is None:
        raise CableError("the ToneSphere driver is not installed")
    api = _setupapi()
    guid = GUID.parse(CLASS_MEDIA)
    devs = api.SetupDiCreateDeviceInfoList(ctypes.byref(guid), None)
    try:
        data = SP_DEVINFO_DATA()
        if not api.SetupDiCreateDeviceInfoW(devs, 'MEDIA', ctypes.byref(guid), None, None, DICD_GENERATE_ID,
                                            ctypes.byref(data)):
            raise CableError(f"SetupDiCreateDeviceInfo failed ({ctypes.get_last_error()})")
        ids = ctypes.create_unicode_buffer(HARDWARE_ID + '\x00\x00')
        if not api.SetupDiSetDeviceRegistryPropertyW(devs, ctypes.byref(data), SPDRP_HARDWAREID, ids,
                                                     ctypes.sizeof(ids)):
            raise CableError(f"could not set the hardware ID ({ctypes.get_last_error()})")
        if not api.SetupDiCallClassInstaller(DIF_REGISTERDEVICE, devs, ctypes.byref(data)):
            raise CableError(f"could not register the device ({ctypes.get_last_error()})")
        newdev = ctypes.WinDLL('newdev', use_last_error=True)
        newdev.DiInstallDevice.argtypes = [wintypes.HWND, ctypes.c_void_p, ctypes.POINTER(SP_DEVINFO_DATA),
                                           ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.BOOL)]
        reboot = wintypes.BOOL()
        if not newdev.DiInstallDevice(None, devs, ctypes.byref(data), None, 0, ctypes.byref(reboot)):
            error = ctypes.get_last_error()
            api.SetupDiCallClassInstaller(DIF_REMOVE, devs, ctypes.byref(data))
            raise CableError(f"could not install the driver on the new cable ({error})")
        instance = ctypes.create_unicode_buffer(512)
        api.SetupDiGetDeviceInstanceIdW(devs, ctypes.byref(data), instance, 512, None)
    finally:
        api.SetupDiDestroyDeviceInfoList(devs)
    # Named after the driver is installed, which would otherwise replace the name with the
    # INF's description; the restart has the endpoints built again under it.
    return _admin_rename(instance.value, name)


def _admin_rename(instance_id: str, name: str) -> dict:
    # The endpoints take the device's name when they are built, so the device is stopped
    # first — refused cleanly if a program holds it — and started again under the new name.
    _stop(instance_id, persist=False)
    api = _setupapi()
    devs, data = _open(api, instance_id)
    try:
        _set_name(api, devs, data, name)
    finally:
        api.SetupDiDestroyDeviceInfoList(devs)
    _pnputil('/enable-device', instance_id)
    return {'instance_id': instance_id, 'name': name}


def _admin_disable(instance_id: str) -> dict:
    _stop(instance_id, persist=True)
    return {'instance_id': instance_id}


def _admin_enable(instance_id: str) -> dict:
    _pnputil('/enable-device', instance_id)
    return {'instance_id': instance_id}


def _admin_remove(instance_id: str) -> dict:
    cable = next((c for c in cables() if c.instance_id == instance_id), None)
    if cable is None:
        raise CableError(f"no cable {instance_id}")
    if cable.enabled:
        _stop(instance_id, persist=False)
    _pnputil('/remove-device', instance_id)
    return {'instance_id': instance_id}


def _admin_install_cables() -> dict:
    """A driver just installed gets the two default cables; one that already has cables, none."""
    if cables():
        return {'created': []}
    return {'created': [_admin_add(name) for name in DEFAULT_NAMES]}


def _admin_remove_driver() -> dict:
    for cable in cables():
        _admin_remove(cable.instance_id)
    package = driver_package()
    if package:
        _pnputil('/delete-driver', package, '/uninstall', '/force')
    return {'removed_package': package}


ADMIN_OPERATIONS = {
    'add': _admin_add, 'rename': _admin_rename, 'disable': _admin_disable, 'enable': _admin_enable,
    'remove': _admin_remove, 'remove-driver': _admin_remove_driver, 'install-cables': _admin_install_cables,
}


def run_admin(operation: str, args: list[str]) -> dict:
    """The elevated side: one operation, its result as a dict (raises CableError)."""
    if operation not in ADMIN_OPERATIONS:
        raise CableError(f"unknown operation {operation}")
    return ADMIN_OPERATIONS[operation](*args)


# --- The unprivileged side ---

class _SHELLEXECUTEINFOW(ctypes.Structure):
    _fields_ = [('cbSize', wintypes.DWORD), ('fMask', wintypes.ULONG), ('hwnd', wintypes.HWND),
                ('lpVerb', wintypes.LPCWSTR), ('lpFile', wintypes.LPCWSTR), ('lpParameters', wintypes.LPCWSTR),
                ('lpDirectory', wintypes.LPCWSTR), ('nShow', ctypes.c_int), ('hInstApp', wintypes.HINSTANCE),
                ('lpIDList', ctypes.c_void_p), ('lpClass', wintypes.LPCWSTR), ('hkeyClass', wintypes.HKEY),
                ('dwHotKey', wintypes.DWORD), ('hIconOrMonitor', wintypes.HANDLE), ('hProcess', wintypes.HANDLE)]


def _command_line() -> tuple[str, list[str]]:
    """How to run ToneSphere's `cable-admin` entry point: the frozen exe itself, or main.py."""
    if getattr(sys, 'frozen', False):
        return sys.executable, ['cable-admin']
    main = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'main.py')
    return sys.executable, [main, 'cable-admin']


def _quote(arg: str) -> str:
    return subprocess.list2cmdline([arg])


def elevate(operation: str, *args: str, timeout_s: float = 120.0) -> dict:
    """
    Run one privileged operation: in-process when already elevated, otherwise in a child
    started through Windows' administrator prompt. Raises ElevationRefused if the user says
    no, CableError with Windows' reason if the operation fails.
    """
    if is_elevated():
        return run_admin(operation, list(args))
    program, prefix = _command_line()
    with tempfile.TemporaryDirectory(prefix='tonesphere-cable-') as folder:
        result_file = os.path.join(folder, 'result.json')
        parameters = ' '.join(_quote(a) for a in prefix + [operation, *args, '--result', result_file])
        info = _SHELLEXECUTEINFOW()
        info.cbSize = ctypes.sizeof(info)
        info.fMask = 0x40   # SEE_MASK_NOCLOSEPROCESS
        info.lpVerb, info.lpFile, info.lpParameters, info.nShow = 'runas', program, parameters, 0
        if not ctypes.windll.shell32.ShellExecuteExW(ctypes.byref(info)):
            if ctypes.GetLastError() == 1223:   # ERROR_CANCELLED
                raise ElevationRefused("the administrator prompt was declined")
            raise CableError(f"could not start the elevated helper ({ctypes.GetLastError()})")
        ctypes.windll.kernel32.WaitForSingleObject(info.hProcess, int(timeout_s * 1000))
        ctypes.windll.kernel32.CloseHandle(info.hProcess)
        if not os.path.exists(result_file):
            raise CableError("the elevated helper ended without a result")
        with open(result_file, encoding='utf-8') as handle:
            outcome = json.load(handle)
    if not outcome.get('ok'):
        raise CableError(outcome.get('error', 'failed'))
    return outcome['result']


def admin_main(argv: list[str]) -> int:
    """`main.py cable-admin <operation> [args...] [--result file]`: the elevated child."""
    result_file = None
    if '--result' in argv:
        i = argv.index('--result')
        result_file = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    try:
        outcome = {'ok': True, 'result': run_admin(argv[0], argv[1:])}
    except (CableError, IndexError, TypeError) as e:
        outcome = {'ok': False, 'error': str(e)}
    text = json.dumps(outcome)
    if result_file:
        with open(result_file, 'w', encoding='utf-8') as handle:
            handle.write(text)
    else:
        print(text)
    return 0 if outcome['ok'] else 1


def as_dicts(items: list[Cable]) -> list[dict]:
    return [{**asdict(c), 'state': c.state} for c in items]
