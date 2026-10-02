"""
What a listener on the AI-04's headphones would hear, measured from outside ToneSphere: this
process captures the interface's input (the guitar, ToneSphere's source) and its output's
loopback (exactly what ToneSphere sends to the headphones), in shared mode beside the app.

    python tests/hardware/first_run_probe.py <seconds> <result.json>

Used by tests/hardware/test_first_run.py, which drives the installed app's own window.
"""

import json
import sys
import time

import numpy as np

from tonesphere.native import NativeEngine, Node, Route
from tonesphere.native.wasapi import StreamSpec, endpoints

RATE = 48000
GUITAR, HEADPHONES, GUITAR_TAP, HEADPHONES_TAP, CLOCK = 1, 2, 10, 20, 30


def measure(seconds: float, interface: str = 'AI-04') -> dict:
    found = {e.flow: e for e in endpoints() if interface in e.name}
    with NativeEngine(RATE, 480) as engine:
        engine.apply_plan([Node.sink(CLOCK, 2), Node.source(GUITAR, 2), Node.source(HEADPHONES, 2),
                           Node.sink(GUITAR_TAP, 2, ring_frames=int(RATE * (seconds + 2))),
                           Node.sink(HEADPHONES_TAP, 2, ring_frames=int(RATE * (seconds + 2)))],
                          [Route(GUITAR, GUITAR_TAP), Route(HEADPHONES, HEADPHONES_TAP)])
        engine.start_wasapi([StreamSpec(CLOCK, 'render', 2, found['render'].id),
                             StreamSpec(GUITAR, 'capture', 2, found['capture'].id),
                             StreamSpec(HEADPHONES, 'loopback', 2, found['render'].id)])
        time.sleep(seconds)
        guitar = engine.port_read(GUITAR_TAP, int(RATE * (seconds + 2)))
        heard = engine.port_read(HEADPHONES_TAP, int(RATE * (seconds + 2)))
        engine.stop_backend()
    n = min(len(guitar), len(heard))
    guitar, heard = guitar[RATE // 2:n].astype(np.float64), heard[RATE // 2:n].astype(np.float64)

    def db(x):
        return float(20 * np.log10(max(np.sqrt(np.mean(x * x)), 1e-12)))

    def dominant(x):
        spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        spectrum[:int(20 * len(x) / RATE)] = 0
        return float(np.argmax(spectrum) * RATE / len(x))

    # How much of what the headphones carry is the guitar: the best normalised correlation
    # of input 1 with each ear, over the lags a monitoring path can have.
    def coherence(a, b, max_lag=RATE // 5):
        a = a - a.mean()
        b = b - b.mean()
        corr = np.fft.irfft(np.fft.rfft(b, 2 * len(b)) * np.conj(np.fft.rfft(a, 2 * len(a))))
        corr = np.concatenate([corr[-max_lag:], corr[:max_lag + 1]]) if max_lag else corr
        best = int(np.argmax(np.abs(corr)))
        return float(abs(corr[best]) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)), best - max_lag

    left_coherence, lag = coherence(guitar[:, 0], heard[:, 0])
    right_coherence, _ = coherence(guitar[:, 0], heard[:, 1])
    return {'seconds': len(guitar) / RATE, 'input_1_db': db(guitar[:, 0]), 'input_2_db': db(guitar[:, 1]),
            'left_db': db(heard[:, 0]), 'right_db': db(heard[:, 1]),
            'input_1_hz': dominant(guitar[:, 0]), 'left_hz': dominant(heard[:, 0]), 'right_hz': dominant(heard[:, 1]),
            'left_coherence': left_coherence, 'right_coherence': right_coherence, 'lag_ms': lag / RATE * 1000}


if __name__ == '__main__':
    result = measure(float(sys.argv[1]))
    with open(sys.argv[2], 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
