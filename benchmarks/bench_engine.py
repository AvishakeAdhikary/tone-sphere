"""
Measure the native engine's processing cost per block, offline.

    uv run python benchmarks/bench_engine.py            # table to stdout
    uv run python benchmarks/bench_engine.py --json out.json

The figures are the engine's own timing, taken on the audio thread inside `run_block`
(the same code a device callback runs), so the ctypes call that drives it offline is not
in them. What they do not include, and a device run would: the driver's own work, format
conversion at the device boundary, and scheduling jitter from running in a real-time
callback rather than a Python loop. They bound what ToneSphere itself adds; they are not
a latency measurement and are never reported as one.
"""

import argparse
import ctypes
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.signals import sine  # noqa: E402
from tonesphere.native import COMPRESSOR, DELAY, EQ, Insert, NativeEngine, Node, Route, _abi, build_info  # noqa: E402

RATES = (44100, 48000, 96000)
BLOCKS = (32, 64, 128, 256, 512)


def guitar_chain(engine):
    """Interface input -> EQ -> compressor -> bus -> delay -> output with safety limiter."""
    nodes = [Node.source(1, 2), Node.bus(10, 2), Node.sink(20, 2, limiter=True)]
    inserts = [Insert(1, 0, EQ), Insert(1, 1, COMPRESSOR), Insert(10, 0, DELAY)]
    engine.apply_plan(nodes, [Route(1, 10), Route(10, 20)], inserts)
    for band, (kind, freq, q, gain) in enumerate([(_abi.EQ_HIGHPASS, 80, 0.707, 0), (_abi.EQ_PEAKING, 2500, 1.0, 3),
                                                   (_abi.EQ_HIGH_SHELF, 8000, 0.8, -4)]):
        for p, v in enumerate((kind, freq, q, gain)):
            engine.set_insert_param(1, 0, band * 4 + p, v)
    return {1: 2}, {20: 2}


def mixer_16(engine):
    """Sixteen stereo sources into a mix bus and a monitor bus, each to its own output."""
    sources = list(range(1, 17))
    nodes = [Node.source(s, 2) for s in sources] + [Node.bus(100, 2), Node.bus(101, 2),
                                                    Node.sink(200, 2, limiter=True), Node.sink(201, 2, limiter=True)]
    routes = [Route(s, 100, gain=0.5, pan=(s - 8) / 8) for s in sources] + \
             [Route(s, 101, gain=0.25) for s in sources] + [Route(100, 200), Route(101, 201)]
    engine.apply_plan(nodes, routes, [Insert(s, 0, EQ) for s in sources])
    return {s: 2 for s in sources}, {200: 2, 201: 2}


SCENARIOS = {'guitar chain': guitar_chain, '16-channel mixer': mixer_16}


def join_pro_audio() -> bool:
    """
    Register this thread with MMCSS as "Pro Audio", as the device backends do for their
    audio threads. Without it the worst case is dominated by the OS scheduling something
    else onto this core mid-block, which says nothing about the engine.
    """
    if sys.platform != 'win32':
        return False
    avrt = ctypes.WinDLL('avrt')
    avrt.AvSetMmThreadCharacteristicsW.restype = ctypes.c_void_p
    task = ctypes.c_uint32(0)
    return bool(avrt.AvSetMmThreadCharacteristicsW('Pro Audio', ctypes.byref(task)))


def measure(rate, block, scenario, seconds):
    with NativeEngine(rate, block) as engine:
        inputs, outputs = SCENARIOS[scenario](engine)
        blocks = max(200, int(seconds * rate / block))
        feed = {node: sine(block, 440.0, rate=rate, amplitude=0.5, channels=ch) for node, ch in inputs.items()}
        for _ in range(50):
            engine.process(feed, outputs)
        engine.reset_stats()
        for _ in range(blocks):
            engine.process(feed, outputs)
        s = engine.stats()
    return {
        'sample_rate': rate,
        'block': block,
        'scenario': scenario,
        'blocks': s['blocks'],
        'period_us': s['period_ns'] / 1000,
        'mean_us': s['callback_ns_mean'] / 1000,
        'p99_us': s['callback_ns_p99'] / 1000,
        'max_us': s['callback_ns_max'] / 1000,
        'worst_load': s['processing_load'],
        'mean_load': s['callback_ns_mean'] / s['period_ns'],
        'rt_allocations': s['rt_allocations'],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seconds', type=float, default=5.0, help='audio time simulated per configuration')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()

    mmcss = join_pro_audio()
    machine = {
        'cpu': platform.processor(),
        'os': platform.platform(),
        'python': platform.python_version(),
        'engine': build_info(),
        'mmcss_pro_audio': mmcss,
    }
    print(f"# {machine['engine']} on {machine['os']} ({machine['cpu']})")
    print(f"# {args.seconds:g} s of audio per configuration; times from inside run_block; MMCSS Pro Audio: "
          f"{'yes' if mmcss else 'NO - worst case includes ordinary scheduling'}")
    print()
    print(f"{'scenario':<18}{'rate':>7}{'block':>6}{'period µs':>11}{'mean µs':>9}{'p99 µs':>9}"
          f"{'max µs':>9}{'worst load':>11}{'allocs':>7}")

    results = []
    for scenario in SCENARIOS:
        for rate in RATES:
            for block in BLOCKS:
                r = measure(rate, block, scenario, args.seconds)
                results.append(r)
                print(f"{scenario:<18}{rate:>7}{block:>6}{r['period_us']:>11.1f}{r['mean_us']:>9.2f}"
                      f"{r['p99_us']:>9.1f}{r['max_us']:>9.1f}{r['worst_load']:>10.1%}{r['rt_allocations']:>7}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({'machine': machine, 'results': results}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
