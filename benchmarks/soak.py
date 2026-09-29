"""
Run the whole application path on a real device for a long time, and restart it many times.

    uv run python benchmarks/soak.py --minutes 30 --json benchmarks/results/soak.json

The path is the one the UI drives: `AudioEngine` -> native host -> WASAPI on the default
output, with a bus fed by a Python thread (pink noise, in real time) routed to the speaker,
and VST3 plugins on the speaker's output side: Surge XT Effects if it is installed, then
ToneSphere's test plugin at gain 0. The test plugin is last so the chain does real work and
the room hears nothing; the output meter proves the silence, the bus meter proves the
signal reached the chain.

Every sample is the engine's own measurement (callback timing on the audio thread, the
device's xrun count, allocations on the audio thread). The feeder thread is ordinary Python,
so ring underruns on the bus say how well Python kept up, not how the engine did; they are
reported separately for that reason.
"""

import argparse
import ctypes
import ctypes.wintypes
import json
import platform
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.signals import pink_noise  # noqa: E402
from tonesphere.core.engine import AudioEngine  # noqa: E402
from tonesphere.native import build_info  # noqa: E402
from tonesphere.plugins import classes_in  # noqa: E402

RATE = 48000
BLOCK = 480


class _Memory(ctypes.Structure):
    _fields_ = [("cb", ctypes.wintypes.DWORD), ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


_kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
_kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
_kernel32.K32GetProcessMemoryInfo.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(_Memory), ctypes.wintypes.DWORD]
_kernel32.GetProcessHandleCount.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(ctypes.wintypes.DWORD)]


def private_bytes() -> int:
    m = _Memory()
    m.cb = ctypes.sizeof(m)
    if not _kernel32.K32GetProcessMemoryInfo(_kernel32.GetCurrentProcess(), ctypes.byref(m), m.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return m.PagefileUsage


def handle_count() -> int:
    count = ctypes.wintypes.DWORD()
    if not _kernel32.GetProcessHandleCount(_kernel32.GetCurrentProcess(), ctypes.byref(count)):
        raise ctypes.WinError(ctypes.get_last_error())
    return count.value


class Feeder(threading.Thread):
    def __init__(self, engine, bus):
        super().__init__(daemon=True)
        self.engine, self.bus = engine, bus
        self.noise = pink_noise(RATE * 10, amplitude=0.25)
        self.stop_event = threading.Event()
        self.written = 0

    def run(self):
        position, start = 0, time.perf_counter()
        while not self.stop_event.is_set():
            due = int((time.perf_counter() - start) * RATE) + BLOCK * 8
            while self.written < due:
                chunk = self.noise[position:position + BLOCK * 4]
                taken = self.engine.write_to_bus(self.bus, chunk)
                if not taken:
                    break
                self.written += taken
                position = (position + taken) % (len(self.noise) - BLOCK * 4)
            time.sleep(0.005)


def plugins():
    found = []
    surge = sorted(Path.home().glob("AppData/Local/Programs/Common/VST3/Surge XT Effects.vst3")) + \
        sorted(Path("C:/Program Files/Common Files/VST3").glob("Surge XT Effects.vst3"))
    if surge:
        found += [c for c in classes_in(surge[0]) if c.is_audio_effect and not c.is_instrument][:1]
    test = sorted(ROOT.glob("native/build/*/VST3/Release/tonesphere_test_gain.vst3"))
    if not test:
        sys.exit("build the test plugin first: uv run python scripts/build_native.py")
    found.append(next(c for c in classes_in(test[0]) if c.name == "ToneSphere Test Gain"))
    return found


def sample(engine, speaker, bus, t0) -> dict:
    s = engine.get_performance_stats()
    rings = engine.get_ring_statistics().get('native', {})
    meters = engine.get_meters()
    out = meters.get(speaker, {}).get('sides', {}).get('output', {})
    feed = meters.get(bus, {})
    return {
        't_s': round(time.perf_counter() - t0, 1),
        'running': s['running'],
        'xruns': s['xruns'],
        'callbacks': s['callback_count'],
        'callback_mean_us': s['callback_mean_ms'] * 1000 if s['callback_mean_ms'] is not None else None,
        'callback_p99_us': s['callback_p99_ms'] * 1000 if s['callback_p99_ms'] is not None else None,
        'callback_max_us': s['callback_max_ms'] * 1000 if s['callback_max_ms'] is not None else None,
        'worst_load': s['processing_load'],
        'rt_allocations': s['audio_thread_allocations'],
        'ring_underruns': rings.get('underruns'),
        'ring_overruns': rings.get('overruns'),
        'bus_peak_db': feed.get('peak_db'),
        'output_peak_db': out.get('peak_db'),
        'private_mb': round(private_bytes() / 2**20, 1),
        'handles': handle_count(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--minutes', type=float, default=30.0)
    parser.add_argument('--restarts', type=int, default=25)
    parser.add_argument('--interval', type=float, default=10.0)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()

    engine = AudioEngine(sample_rate=RATE, buffer_size=BLOCK, exclusive=False)
    engine.initialize()
    speaker = engine.default_output_id()
    bus = engine.create_virtual_input("soak", channels=2)
    assert engine.create_routing(bus, speaker)[0]
    chain = plugins()
    for info in chain:
        ok, message = engine.add_plugin(speaker, info, is_input=False)
        assert ok, message
    engine.set_plugin_parameter(speaker, len(chain) - 1, 0, 0.0, is_input=False)  # the test plugin at gain 0

    report = {
        'machine': {'cpu': platform.processor(), 'os': platform.platform(), 'engine': build_info()},
        'config': {'rate': RATE, 'block': BLOCK, 'device': engine.get_device_info(speaker).name,
                   'plugins': [p.name for p in chain], 'minutes': args.minutes, 'restarts': args.restarts},
        'samples': [], 'restarts': [],
    }

    engine.start_engine()
    feeder = Feeder(engine, bus)
    feeder.start()
    t0 = time.perf_counter()
    try:
        while time.perf_counter() - t0 < args.minutes * 60:
            time.sleep(args.interval)
            s = sample(engine, speaker, bus, t0)
            report['samples'].append(s)
            print(json.dumps(s), flush=True)
        feeder.stop_event.set()
        feeder.join()

        for i in range(args.restarts):
            engine.stop_engine()
            began = time.perf_counter()
            engine.start_engine()
            came_up = engine.host.is_running
            time.sleep(0.5)
            s = engine.get_performance_stats()
            report['restarts'].append({'cycle': i, 'running': came_up, 'start_ms': round((time.perf_counter() - began
                                                                                            - 0.5) * 1000, 1),
                                       'xruns_after': s['xruns'], 'rt_allocations': s['audio_thread_allocations'],
                                       'private_mb': round(private_bytes() / 2**20, 1), 'handles': handle_count()})
            print(json.dumps(report['restarts'][-1]), flush=True)
    finally:
        feeder.stop_event.set()
        engine.cleanup()

    samples = report['samples']
    live = [s for s in samples if s['running']]
    report['summary'] = {
        'duration_s': samples[-1]['t_s'] if samples else 0,
        'always_running': all(s['running'] for s in samples),
        'xruns_total': samples[-1]['xruns'] if samples else None,
        'callbacks_total': samples[-1]['callbacks'] if samples else None,
        'callback_max_us': max((s['callback_max_us'] for s in live if s['callback_max_us'] is not None), default=None),
        'worst_load': max((s['worst_load'] for s in live if s['worst_load'] is not None), default=None),
        'rt_allocations': samples[-1]['rt_allocations'] if samples else None,
        'output_silent': all(s['output_peak_db'] is not None and s['output_peak_db'] <= -99 for s in live),
        'bus_carried_signal': all(s['bus_peak_db'] is not None and s['bus_peak_db'] > -30 for s in live),
        'private_mb_first_last': [samples[0]['private_mb'], samples[-1]['private_mb']] if samples else None,
        # The first sample includes the feeder's start, before which the bus was empty by design.
        'ring_underruns_after_first_sample': samples[-1]['ring_underruns'] - samples[0]['ring_underruns']
        if len(samples) > 1 else None,
        'restarts_all_running': all(r['running'] for r in report['restarts']),
        'handles_first_last_restart': [report['restarts'][0]['handles'], report['restarts'][-1]['handles']]
        if report['restarts'] else None,
    }
    print(json.dumps(report['summary'], indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
